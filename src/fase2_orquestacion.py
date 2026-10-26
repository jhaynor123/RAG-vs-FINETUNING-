"""
Fase 2 — Orquestación y representación del corpus.

Toma los chunks de la Fase 1 (corpus_A.jsonl, corpus_B.jsonl) y produce dos
representaciones del mismo corpus (Figuras 1 y 2 del artículo):

  1) Vectores densos indexados en ChromaDB -> Ruta A (RAG, fase3_ruta_rag.py).
  2) Dataset sintético de preguntas y respuestas (JSONL) -> Ruta B
     (Fine-Tuning, fase3_ruta_finetuning/).

El dataset sintético lo genera Claude: a partir de cada chunk redacta
preguntas realistas de un operario o técnico y su respuesta, basada solo en
ese chunk. Se descartan los pares cuya pregunta coincide con un caso de prueba.

Uso:
    python src/fase2_orquestacion.py                 # indexa + genera QA
    python src/fase2_orquestacion.py --solo-indexar   # solo Ruta A
    python src/fase2_orquestacion.py --solo-qa        # solo Ruta B
"""
import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from embeddings_utils import embed_textos

import chromadb
from chromadb.config import Settings


def cargar_chunks() -> list[dict]:
    chunks = []
    for fname in ("corpus_A.jsonl", "corpus_B.jsonl"):
        path = config.DATA_PROCESSED / fname
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line))
    return chunks


def construir_indice(chunks: list[dict]):
    """Ruta A: genera embeddings densos y los guarda en ChromaDB (persistente)."""
    print(f"Generando embeddings para {len(chunks)} chunks ...")
    textos = [c["text"] for c in chunks]
    vectores = embed_textos(textos)

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR), settings=Settings(anonymized_telemetry=False))
    coleccion = client.get_or_create_collection(
        name="corpus", metadata={"hnsw:space": "cosine"}
    )
    # Reindexar limpio para evitar duplicados en corridas repetidas
    ids_existentes = coleccion.get()["ids"]
    if ids_existentes:
        coleccion.delete(ids=ids_existentes)

    coleccion.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=vectores,
        documents=textos,
        metadatas=[{"source_file": c["source_file"], "subset": c["subset"]} for c in chunks],
    )
    print(f"  -> {coleccion.count()} vectores indexados en {config.CHROMA_DIR}")


PROMPT_QA = """Eres un generador de datos de entrenamiento para un sistema de soporte técnico industrial (planta textil).

A partir del siguiente fragmento de documentación técnica, redacta {n_preguntas} pares de pregunta-respuesta realistas, como los que un operario o técnico de planta preguntaría en su trabajo diario. La respuesta debe basarse ÚNICAMENTE en la información del fragmento (no inventes datos que no estén ahí).

{estilo}

Fragmento (fuente: {source_file}):
\"\"\"
{texto}
\"\"\"

Si el fragmento no contiene información técnica útil (índices, encabezados, números de página, texto cortado sin sentido), responde con una lista vacía: [].
Responde ÚNICAMENTE con un JSON válido, una lista de objetos con las claves "pregunta" y "respuesta". No agregues texto antes ni después del JSON."""

# Estilo de respuesta por subconjunto: el mismo que el rol de sistema con el
# que se entrena y se evalúa cada uno (config.SYSTEM_PROMPTS).
ESTILO_POR_SUBSET = {
    "A": (
        "Estilo de las respuestas: como un ingeniero textil experto en mantenimiento, con explicación "
        "detallada y paso a paso (usa viñetas cuando haya varios pasos). Las {n_preguntas} preguntas deben "
        "ser distintas entre sí y de redacción natural."
    ),
    "B": (
        "Estilo de las respuestas: SIEMPRE en el formato exacto «Pieza: ..., Fallo: ..., Acción: ...», técnico y "
        "ultra conciso, sin texto adicional. Las preguntas deben mencionar el código del ticket (por ejemplo "
        "ERR-XXX-NN) y variar en su redacción (¿qué pieza falla con...?, ¿qué hago si aparece...?, el operario "
        "reporta..., etc.); NO uses la fórmula «Interpreta el ticket ...»."
    ),
}


# 3 respuestas paso a paso sobre un fragmento denso superan los 1.536 tokens (8 de 288 chunks se truncaron).
MAX_TOKENS_QA = 4096


def _normalizar(texto: str) -> str:
    return " ".join(re.findall(r"\w+", texto.lower()))


def filtrar_fuga_con_casos_de_prueba(dataset: list[dict]) -> tuple[list[dict], int]:
    """El corpus de tickets ES la fuente de los casos de prueba, así que el
    dataset sintético no puede contener ninguno de ellos (el artículo declara
    que excluye los casos de prueba). Elimina los pares cuya pregunta, una vez
    normalizada, coincide con la de un caso de prueba."""
    path = config.CASOS_PRUEBA_DIR / "casos_de_prueba.jsonl"
    if not path.exists():
        return dataset, 0
    prohibidas = {_normalizar(json.loads(l)["pregunta"]) for l in open(path, encoding="utf-8") if l.strip()}
    limpio = [d for d in dataset if _normalizar(d["pregunta"]) not in prohibidas]
    return limpio, len(dataset) - len(limpio)


def _extraer_pares(bruto: str) -> list[dict]:
    bruto = bruto.strip()
    if bruto.startswith("```"):  # tolerar ```json ... ```
        bruto = bruto.strip("`")
    return json.loads(bruto[bruto.find("[") : bruto.rfind("]") + 1])


def generar_qa_sinteticas(chunks: list[dict], n_por_chunk: int = 3, cliente=None, hilos: int = 8) -> tuple[list[dict], dict]:
    """Ruta B: usa Claude para generar pares QA sintéticos por cada chunk
    (en paralelo). Devuelve (dataset, resumen) con el conteo de tokens y el
    costo en USD. Requiere ANTHROPIC_API_KEY. Si no hay clave, imprime el
    prompt que se enviaría (dry-run)."""
    if cliente is None and not config.ANTHROPIC_API_KEY:
        print("[fase2] ANTHROPIC_API_KEY no configurada -> modo dry-run: no se llama a la API.")
        if chunks:
            c = chunks[0]
            print(PROMPT_QA.format(n_preguntas=n_por_chunk, estilo=ESTILO_POR_SUBSET[c["subset"]].format(n_preguntas=n_por_chunk),
                                   source_file=c["source_file"], texto=c["text"][:500]))
        return [], {}

    from anthropic import Anthropic

    cliente = cliente or Anthropic(api_key=config.ANTHROPIC_API_KEY)
    candado = threading.Lock()
    uso = {"llamadas": 0, "tokens_entrada": 0, "tokens_salida": 0}

    def procesar(c: dict):
        prompt = PROMPT_QA.format(
            n_preguntas=n_por_chunk,
            estilo=ESTILO_POR_SUBSET[c["subset"]].format(n_preguntas=n_por_chunk),
            source_file=c["source_file"],
            texto=c["text"],
        )
        ultimo_error = None
        for intento in range(3):
            try:
                resp = cliente.messages.create(
                    model=config.CLAUDE_MODEL_GENERADOR,
                    max_tokens=MAX_TOKENS_QA,
                    messages=[{"role": "user", "content": prompt}],
                )
                with candado:
                    uso["llamadas"] += 1
                    uso["tokens_entrada"] += resp.usage.input_tokens
                    uso["tokens_salida"] += resp.usage.output_tokens
                if resp.stop_reason == "max_tokens":
                    raise ValueError("Respuesta truncada por max_tokens al generar QA sintéticas.")
                pares = _extraer_pares(config.extraer_texto(resp))
                return c, [
                    {
                        "chunk_id": c["chunk_id"],
                        "source_file": c["source_file"],
                        "subset": c["subset"],
                        "pregunta": p["pregunta"].strip(),
                        "respuesta": p["respuesta"].strip(),
                    }
                    for p in pares
                ], None
            except Exception as e:
                ultimo_error = f"{type(e).__name__}: {e}"
                time.sleep(2**intento)
        return c, [], ultimo_error

    dataset, fallidos, sin_pares = [], [], 0
    with ThreadPoolExecutor(max_workers=hilos) as pool:
        for c, pares, error in tqdm(pool.map(procesar, chunks), total=len(chunks), desc="Generando QA"):
            if error:
                fallidos.append({"chunk_id": c["chunk_id"], "error": error})
            elif not pares:
                sin_pares += 1
            dataset.extend(pares)

    dataset, n_fuga = filtrar_fuga_con_casos_de_prueba(dataset)
    resumen = {
        "modelo": config.CLAUDE_MODEL_GENERADOR,
        "chunks_procesados": len(chunks),
        "chunks_sin_pares_utiles": sin_pares,
        "chunks_fallidos": len(fallidos),
        "pares_eliminados_por_igualar_un_caso_de_prueba": n_fuga,
        "pares_finales": len(dataset),
        "pares_por_subset": {s: sum(1 for d in dataset if d["subset"] == s) for s in ("A", "B")},
        **uso,
        "costo_usd": round(config.costo_usd(config.CLAUDE_MODEL_GENERADOR, uso["tokens_entrada"], uso["tokens_salida"]), 4),
        "chunks_fallidos_detalle": fallidos,
    }
    return dataset, resumen


def guardar_dataset_qa(dataset: list[dict], resumen: dict):
    destino = config.DATA_PROCESSED / "dataset_qa_sintetico.jsonl"
    with open(destino, "w", encoding="utf-8") as f:
        for d in dataset:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"  guardado: {destino} ({len(dataset)} pares QA)")
    with open(config.RESULTADOS_DIR / "costo_dataset_qa.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo-indexar", action="store_true")
    ap.add_argument("--solo-qa", action="store_true")
    ap.add_argument("--n-por-chunk", type=int, default=3)
    ap.add_argument("--limite-chunks", type=int, default=None, help="Prueba rápida: solo los primeros N chunks")
    args = ap.parse_args()

    print("=== Fase 2: Orquestación y representación ===\n")
    chunks = cargar_chunks()
    if not chunks:
        print("No hay chunks en data/processed/. Corre primero fase1_ingesta.py")
        return
    print(f"Chunks cargados: {len(chunks)}\n")

    if not args.solo_qa:
        print("-- Representación 1: vectores densos (Ruta A) --")
        construir_indice(chunks)

    if not args.solo_indexar:
        print("\n-- Representación 2: dataset sintético QA (Ruta B) --")
        if args.limite_chunks:
            chunks = chunks[: args.limite_chunks]
        dataset, resumen = generar_qa_sinteticas(chunks, n_por_chunk=args.n_por_chunk)
        if dataset:
            guardar_dataset_qa(dataset, resumen)
            print(f"\n  COSTO DE ESTA PARTE: ${resumen['costo_usd']} USD "
                  f"({resumen['llamadas']} llamadas, {resumen['tokens_entrada']:,} tokens de entrada, "
                  f"{resumen['tokens_salida']:,} de salida) -> resultados/costo_dataset_qa.json")


if __name__ == "__main__":
    main()
