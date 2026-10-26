"""
Las tres métricas del artículo (Faithfulness, Answer Correctness y Answer
Relevance), con DeepSeek como juez (LLM-as-a-Judge).

Cada función hace una llamada con salida JSON y devuelve un puntaje en [0, 1]
junto con el detalle que lo sustenta, para poder auditar el juicio.

`cliente` es un `openai.OpenAI` apuntando a config.DEEPSEEK_BASE_URL (la API
de DeepSeek es compatible con el SDK de OpenAI); ver juez_llm.py.
"""
import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from embeddings_utils import embed_textos

import numpy as np


_INTENTOS_JSON = 3

# Tokens realmente facturados al juez en esta ejecución (para el costo real).
USO_JUEZ = {"llamadas": 0, "tokens_entrada": 0, "tokens_salida": 0}
_candado_uso = threading.Lock()


def registrar_uso_juez(resp):
    with _candado_uso:
        USO_JUEZ["llamadas"] += 1
        USO_JUEZ["tokens_entrada"] += resp.usage.prompt_tokens
        USO_JUEZ["tokens_salida"] += resp.usage.completion_tokens


def _llamar_json(cliente, prompt: str, max_tokens: int = 4096) -> dict | list:
    for intento in range(1, _INTENTOS_JSON + 1):
        resp = cliente.chat.completions.create(
            model=config.DEEPSEEK_MODEL_JUEZ,
            max_tokens=config.max_tokens_juez(max_tokens),
            **config.kwargs_juez(),
            messages=[{"role": "user", "content": prompt}],
        )
        registrar_uso_juez(resp)
        if resp.choices[0].finish_reason == "length":
            # evita parsear un JSON truncado
            raise ValueError(
                f"Respuesta del juez truncada por max_tokens={max_tokens} (finish_reason=length). "
                "Sube max_tokens en la llamada."
            )
        bruto = (resp.choices[0].message.content or "").strip()
        if bruto.startswith("```"):
            bruto = bruto.strip("`")
        inicio = min((i for i in (bruto.find("["), bruto.find("{")) if i != -1), default=0)
        fin = max(bruto.rfind("]"), bruto.rfind("}")) + 1
        try:
            return json.loads(bruto[inicio:fin])
        except json.JSONDecodeError:
            if intento == _INTENTOS_JSON:
                raise


PROMPT_FAITHFULNESS = """Analiza la siguiente respuesta y descompónla en afirmaciones atómicas (claims). Para cada afirmación, indica si está respaldada por el contexto proporcionado.

Contexto:
\"\"\"
{contexto}
\"\"\"

Respuesta a analizar:
\"\"\"
{respuesta}
\"\"\"

Responde SOLO con JSON: una lista de objetos {{"afirmacion": "...", "soportada_por_contexto": true/false, "justificacion": "..."}}."""


def faithfulness(pregunta: str, respuesta: str, contexto: str, cliente) -> dict:
    if not respuesta or not respuesta.strip():
        return {"score": 0.0, "detalle": [], "nota": "respuesta vacía"}
    afirmaciones = _llamar_json(
        cliente, PROMPT_FAITHFULNESS.format(contexto=contexto, respuesta=respuesta)
    )
    if not afirmaciones:
        return {"score": 1.0, "detalle": [], "nota": "sin afirmaciones verificables (respuesta trivial)"}
    soportadas = sum(1 for a in afirmaciones if a.get("soportada_por_contexto"))
    return {"score": soportadas / len(afirmaciones), "detalle": afirmaciones}


PROMPT_CORRECTNESS = """Compara la respuesta generada contra la respuesta de referencia (considerada correcta). Clasifica los hechos de la respuesta generada en:
- verdaderos_positivos: hechos correctos que coinciden con la referencia
- falsos_positivos: hechos inventados o incorrectos que NO están en la referencia
- falsos_negativos: hechos importantes de la referencia que la respuesta generada omitió

Pregunta: {pregunta}

Respuesta de referencia:
\"\"\"
{referencia}
\"\"\"

Respuesta generada:
\"\"\"
{respuesta}
\"\"\"

Responde SOLO con JSON: {{"verdaderos_positivos": [...], "falsos_positivos": [...], "falsos_negativos": [...]}}"""


def answer_correctness(pregunta: str, respuesta: str, referencia: str, cliente) -> dict:
    if not respuesta or not respuesta.strip():
        return {"score": 0.0, "tp": 0, "fp": 0, "fn": 1, "nota": "respuesta vacía"}
    d = _llamar_json(
        cliente,
        PROMPT_CORRECTNESS.format(pregunta=pregunta, referencia=referencia, respuesta=respuesta),
    )
    tp = len(d.get("verdaderos_positivos", []))
    fp = len(d.get("falsos_positivos", []))
    fn = len(d.get("falsos_negativos", []))
    denom = 2 * tp + fp + fn
    score = (2 * tp / denom) if denom > 0 else 1.0
    return {"score": score, "tp": tp, "fp": fp, "fn": fn, "detalle": d}


PROMPT_PREGUNTAS_SINTETICAS = """A partir de la siguiente respuesta, genera {n} preguntas distintas que esta respuesta contestaría directamente (ingeniería inversa de la pregunta original).

Respuesta:
\"\"\"
{respuesta}
\"\"\"

Responde SOLO con JSON: una lista de {n} strings (las preguntas)."""


def answer_relevance(pregunta_original: str, respuesta: str, cliente, n: int = 3) -> dict:
    if not respuesta or not respuesta.strip():
        return {"score": 0.0, "preguntas_sinteticas": [], "nota": "respuesta vacía"}
    preguntas = _llamar_json(
        cliente, PROMPT_PREGUNTAS_SINTETICAS.format(respuesta=respuesta, n=n), max_tokens=1024
    )
    if not preguntas:
        return {"score": 0.0, "preguntas_sinteticas": []}

    vectores = embed_textos([pregunta_original] + list(preguntas))
    v_original = np.array(vectores[0])
    similitudes = []
    for v in vectores[1:]:
        v = np.array(v)
        denom = (np.linalg.norm(v_original) * np.linalg.norm(v)) or 1e-9
        similitudes.append(float(np.dot(v_original, v) / denom))
    return {
        "score": sum(similitudes) / len(similitudes),
        "preguntas_sinteticas": list(preguntas),
        "similitudes": similitudes,
    }


def evaluar_respuesta(pregunta: str, respuesta: str, referencia: str, contexto: str, cliente) -> dict:
    """Calcula las tres métricas para una sola respuesta (una arquitectura, un caso)."""
    return {
        "faithfulness": faithfulness(pregunta, respuesta, contexto, cliente),
        "answer_correctness": answer_correctness(pregunta, respuesta, referencia, cliente),
        "answer_relevance": answer_relevance(pregunta, respuesta, cliente),
    }
