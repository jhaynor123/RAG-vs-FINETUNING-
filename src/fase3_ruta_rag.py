"""
Fase 3 — Ruta A (RAG): recuperación híbrida y generación.

Dada una consulta, recupera los fragmentos más relevantes combinando similitud
vectorial (índice de ChromaDB, fase2_orquestacion.py) y BM25 (coincidencia
exacta de códigos), y llama a uno de los tres generadores comerciales (Claude,
GPT o Gemini, según `proveedor`), restringido a responder solo con esa
evidencia. Los tres reciben el mismo prompt y el mismo límite de tokens, así
que el generador es lo único que cambia entre las configuraciones de la Ruta A.

Como librería:
    from fase3_ruta_rag import responder_rag
    resultado = responder_rag("¿Qué código de error...?", proveedor="claude")  # o "gpt", "gemini"

Como script (prueba rápida):
    python src/fase3_ruta_rag.py "tu pregunta aquí"
    python src/fase3_ruta_rag.py --proveedor gpt "tu pregunta aquí"
"""
import sys
import threading
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from busqueda_lexica import BM25
from embeddings_utils import embed_textos

import chromadb
import numpy as np
from chromadb.config import Settings

PROMPT_GENERADOR = """Eres un asistente técnico de una planta textil. Responde la consulta del usuario basándote ÚNICAMENTE en los fragmentos de contexto que se muestran abajo. Si el contexto no contiene la información necesaria, dilo explícitamente en vez de inventar una respuesta.

Contexto recuperado:
{contexto}

Consulta: {consulta}

Responde de forma directa y precisa, citando los códigos o cifras exactas cuando el contexto los mencione."""


def _coleccion():
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR), settings=Settings(anonymized_telemetry=False))
    return client.get_or_create_collection(name="corpus", metadata={"hnsw:space": "cosine"})


_indice_lexico = None
_candado_indice = threading.Lock()


def _indice():
    """(ids, documentos, metadatos, BM25, embeddings) del índice completo; se arma una vez."""
    global _indice_lexico
    if _indice_lexico is None:
        with _candado_indice:
            if _indice_lexico is None:
                datos = _coleccion().get(include=["documents", "metadatas", "embeddings"])
                _indice_lexico = (
                    datos["ids"], datos["documents"], datos["metadatas"],
                    BM25(datos["documents"]), np.array(datos["embeddings"], dtype=np.float32),
                )
    return _indice_lexico


def _normalizar(puntajes: dict) -> dict:
    """Min-max a [0, 1] dentro de la consulta."""
    bajo, alto = min(puntajes.values()), max(puntajes.values())
    if alto == bajo:
        return {k: 0.0 for k in puntajes}
    return {k: (v - bajo) / (alto - bajo) for k, v in puntajes.items()}


def recuperar(consulta: str, top_k: int = None, top_k_final: int = None, subset: str = None) -> list[dict]:
    """Recupera los fragmentos más relevantes para la consulta.

    Recuperación híbrida: puntaje = w * BM25 + (1 - w) * similitud coseno de
    embeddings (w = config.RETRIEVER_PESO_LEXICO), cada uno normalizado a
    [0, 1] por consulta. BM25 aporta la coincidencia exacta de códigos
    alfanuméricos, que los embeddings densos no distinguen (ver
    busqueda_lexica.py).

    Se toman los top_k mejores candidatos por puntaje híbrido y se recorta a
    top_k_final. El re-ranking usa ese mismo puntaje; un re-ranker distinto
    (por ejemplo, un cross-encoder) se enchufaría aquí.

    `subset` ("A" o "B") restringe la búsqueda a un subconjunto del corpus; el
    sistema no lo usa al responder, solo enlazar_fragmento_fuente.py.
    """
    top_k = top_k or config.RETRIEVER_TOP_K
    top_k_final = top_k_final or config.RETRIEVER_TOP_K_FINAL

    coleccion = _coleccion()
    if coleccion.count() == 0:
        raise RuntimeError(
            "El índice de ChromaDB está vacío. Corre primero fase2_orquestacion.py."
        )

    ids, documentos, metadatos, bm25, embeddings = _indice()
    texto = dict(zip(ids, documentos))
    meta = dict(zip(ids, metadatos))

    [vector_consulta] = embed_textos([consulta])
    # similitud coseno EXACTA contra todos los fragmentos (los embeddings están
    # normalizados); el índice aproximado de ChromaDB falla al pedirle todos
    # los resultados y con unos pocos cientos de fragmentos no hace falta.
    similitud = embeddings @ np.array(vector_consulta, dtype=np.float32)
    denso = _normalizar(dict(zip(ids, map(float, similitud))))
    lexico = _normalizar(dict(zip(ids, bm25.puntajes(consulta))))

    w = config.RETRIEVER_PESO_LEXICO
    puntaje = {i: w * lexico[i] + (1 - w) * denso[i] for i in ids}
    orden = sorted(ids, key=lambda i: -puntaje[i])
    if subset:
        orden = [i for i in orden if meta[i]["subset"] == subset]

    candidatos = [
        {"chunk_id": i, "text": texto[i], "metadata": meta[i], "score": puntaje[i],
         "score_denso": denso[i], "score_lexico": lexico[i]}
        for i in orden[:top_k]
    ]
    return candidatos[:top_k_final]


def _clave_configurada(proveedor: str) -> bool:
    return bool(
        {
            "claude": config.ANTHROPIC_API_KEY,
            "gpt": config.OPENAI_API_KEY,
            "gemini": config.GOOGLE_API_KEY,
        }.get(proveedor)
    )


# Tokens realmente facturados por proveedor en esta ejecución (para el costo real).
USO = defaultdict(lambda: {"llamadas": 0, "tokens_entrada": 0, "tokens_salida": 0})
_candado_uso = threading.Lock()


def _registrar_uso(proveedor: str, tokens_entrada: int, tokens_salida: int):
    with _candado_uso:
        u = USO[proveedor]
        u["llamadas"] += 1
        u["tokens_entrada"] += tokens_entrada or 0
        u["tokens_salida"] += tokens_salida or 0


def _kwargs_temperatura_generador() -> dict:
    """Vacío mientras config.GENERADORES_EXPONEN_TEMPERATURE sea False (ver config.py)."""
    return {"temperature": config.GENERACION_TEMPERATURE} if config.GENERADORES_EXPONEN_TEMPERATURE else {}


def _generar_claude(prompt: str, cliente=None) -> str:
    from anthropic import Anthropic

    cliente = cliente or Anthropic(api_key=config.ANTHROPIC_API_KEY)
    resp = cliente.messages.create(
        model=config.CLAUDE_MODEL_GENERADOR,
        max_tokens=config.MAX_TOKENS_API_GENERADOR,
        **_kwargs_temperatura_generador(),
        messages=[{"role": "user", "content": prompt}],
    )
    _registrar_uso("claude", resp.usage.input_tokens, resp.usage.output_tokens)
    return config.extraer_texto(resp)


def _generar_gpt(prompt: str) -> str:
    from openai import OpenAI

    cliente = OpenAI(api_key=config.OPENAI_API_KEY)
    resp = cliente.responses.create(
        model=config.GPT_MODEL_GENERADOR,
        input=prompt,
        max_output_tokens=config.MAX_TOKENS_API_GENERADOR,
        **_kwargs_temperatura_generador(),
    )
    _registrar_uso("gpt", resp.usage.input_tokens, resp.usage.output_tokens)
    return resp.output_text


def _generar_gemini(prompt: str) -> str:
    from google import genai
    from google.genai import types

    cliente = genai.Client(api_key=config.GOOGLE_API_KEY)
    resp = cliente.models.generate_content(
        model=config.GEMINI_MODEL_GENERADOR,
        contents=prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=config.MAX_TOKENS_API_GENERADOR,
            **_kwargs_temperatura_generador(),
        ),
    )
    um = resp.usage_metadata  # los tokens de pensamiento se facturan como salida
    _registrar_uso("gemini", um.prompt_token_count, (um.candidates_token_count or 0) + (um.thoughts_token_count or 0))
    return config.extraer_texto_gemini(resp)


_GENERADORES = {"claude": _generar_claude, "gpt": _generar_gpt, "gemini": _generar_gemini}


def responder_rag(consulta: str, proveedor: str = "claude", cliente=None, devolver_contexto: bool = False) -> dict:
    """proveedor: "claude", "gpt" o "gemini" -- los tres generadores de la
    Ruta A. `cliente` solo se usa (y solo tiene efecto) con proveedor="claude",
    para reutilizar una instancia ya creada)."""
    if proveedor not in _GENERADORES:
        raise ValueError(f"proveedor debe ser uno de {list(_GENERADORES)}, no {proveedor!r}")

    fragmentos = recuperar(consulta)
    contexto = "\n\n---\n\n".join(
        f"[Fuente: {f['metadata']['source_file']}]\n{f['text']}" for f in fragmentos
    )
    prompt = PROMPT_GENERADOR.format(contexto=contexto, consulta=consulta)

    if cliente is None and not _clave_configurada(proveedor):
        return {
            "respuesta": None,
            "fragmentos": fragmentos,
            "prompt_dry_run": prompt,
            "aviso": f"clave de API para {proveedor!r} no configurada: no se llamó al generador. "
            "Se devuelve el prompt que se habría enviado.",
        }

    if proveedor == "claude":
        texto = _generar_claude(prompt, cliente=cliente)
    else:
        texto = _GENERADORES[proveedor](prompt)

    salida = {"respuesta": texto, "fragmentos": fragmentos}
    if devolver_contexto:
        salida["prompt"] = prompt
    return salida


if __name__ == "__main__":
    argv = sys.argv[1:]
    proveedor = "claude"
    if argv and argv[0] == "--proveedor":
        proveedor = argv[1]
        argv = argv[2:]
    consulta = " ".join(argv) or "¿Qué código de error indica sobrecalentamiento del motor principal?"
    print(f"Consulta ({proveedor}): {consulta}\n")
    resultado = responder_rag(consulta, proveedor=proveedor)

    print("Fragmentos recuperados:")
    for f in resultado["fragmentos"]:
        print(f"  - {f['chunk_id']}  (score={f['score']:.3f})  {f['text'][:100]}...")

    print()
    if resultado.get("respuesta"):
        print(f"Respuesta ({proveedor}):")
        print(resultado["respuesta"])
    else:
        print(resultado.get("aviso"))
        print("\n--- prompt que se habría enviado ---")
        print(resultado["prompt_dry_run"][:800])
