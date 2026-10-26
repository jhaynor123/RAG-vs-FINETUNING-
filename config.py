"""
Configuración central del experimento RAG vs. Fine-Tuning.

Todos los scripts toman de aquí los modelos, rutas, hiperparámetros y
parámetros del protocolo de evaluación (Tabla 1 y Tabla 3 del artículo).
Las claves de API se leen del entorno (archivo .env, ver .env.example).
"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")  # ChromaDB: sin telemetría

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
DATA_RAW_MANUALES = ROOT_DIR / "data" / "raw" / "manuales"   # subconjunto A (lenguaje natural)
DATA_RAW_TICKETS = ROOT_DIR / "data" / "raw" / "tickets"     # subconjunto B (semiestructurado)
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
CHROMA_DIR = DATA_PROCESSED / "chroma_db"
CASOS_PRUEBA_DIR = ROOT_DIR / "casos_prueba"
RESULTADOS_DIR = ROOT_DIR / "resultados"

for _d in (DATA_PROCESSED, CASOS_PRUEBA_DIR, RESULTADOS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Claves de API
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")  # solo para el juez
DEEPSEEK_BASE_URL = "https://api.deepseek.com"             # compatible con el SDK de OpenAI

# ---------------------------------------------------------------------------
# Generadores de la Ruta A (RAG): tres proveedores con tarifas distintas.
# Los IDs cambian con el tiempo; los vigentes durante la campaña son los de
# la Tabla 1 del artículo.
# ---------------------------------------------------------------------------
CLAUDE_MODEL_GENERADOR = os.environ.get("CLAUDE_MODEL_GENERADOR", "claude-sonnet-5-5")
GPT_MODEL_GENERADOR = os.environ.get("GPT_MODEL_GENERADOR", "gpt-6.1-sol")
GEMINI_MODEL_GENERADOR = os.environ.get("GEMINI_MODEL_GENERADOR", "gemini-3.8-flash")

# claude-sonnet-5-5 y gpt-6.1-sol rechazan `temperature` (error 400). Para que
# ningún generador tenga ventaja por exponer ese control, tampoco se envía a
# gemini-3.8-flash, que sí la acepta.
GENERADORES_EXPONEN_TEMPERATURE = False

# Solo la inferencia local de la Ruta B (Qwen) usa esta temperatura.
GENERACION_TEMPERATURE = 0.1

# Longitud máxima de la respuesta visible, común a las cuatro configuraciones.
MAX_TOKENS_RESPUESTA = 512
# Techo que se envía a la API: los modelos de razonamiento gastan parte de
# max_tokens en "pensamiento" oculto antes de escribir la respuesta.
MAX_TOKENS_API_GENERADOR = MAX_TOKENS_RESPUESTA + 1024

# Rol de sistema de cada subconjunto (A = manuales, B = registros de incidencia);
# son los mismos de los 200 casos reales y del entrenamiento de la Ruta B.
SYSTEM_PROMPTS = {
    "A": "Eres un ingeniero textil experto en mantenimiento de maquinaria. Responde con explicaciones detalladas y paso a paso.",
    "B": "Eres un sistema automatizado de diagnóstico de incidencias textiles. Responde de forma técnica, estructurada y ultra concisa.",
}

# ---------------------------------------------------------------------------
# Juez (Fase 5): DeepSeek, que no comparte proveedor con ningún generador, así
# que no hace falta excluirlo de ninguna comparación (control de
# auto-preferencia, Tabla 3).
# ---------------------------------------------------------------------------
DEEPSEEK_MODEL_JUEZ = os.environ.get("DEEPSEEK_MODEL_JUEZ", "deepseek-v4-pro")

_t_juez = os.environ.get("JUEZ_TEMPERATURE", "0").strip().lower()
JUEZ_TEMPERATURE = None if _t_juez in ("", "none") else float(_t_juez)  # "none": no enviar el parámetro

JUEZ_CONCURRENCIA = int(os.environ.get("JUEZ_CONCURRENCIA", "8"))  # llamadas simultáneas

# Con el razonamiento activado la API ignora `temperature` y descuenta los
# tokens de razonamiento de max_tokens (cortando el JSON del juez). Por eso el
# modo por defecto es "disabled"; el razonamiento explícito ya lo pide el prompt.
DEEPSEEK_THINKING = os.environ.get("DEEPSEEK_THINKING", "disabled").strip().lower()
assert DEEPSEEK_THINKING in ("disabled", "enabled"), "DEEPSEEK_THINKING debe ser 'disabled' o 'enabled'"
_MARGEN_RAZONAMIENTO_JUEZ = 8192


def kwargs_juez() -> dict:
    """Argumentos extra de chat.completions.create() para el juez."""
    kw = {} if JUEZ_TEMPERATURE is None else {"temperature": JUEZ_TEMPERATURE}
    kw["extra_body"] = {"thinking": {"type": DEEPSEEK_THINKING}}
    return kw


def max_tokens_juez(base: int) -> int:
    """`base` para la respuesta visible, más margen si el razonamiento está activado."""
    return base + (_MARGEN_RAZONAMIENTO_JUEZ if DEEPSEEK_THINKING == "enabled" else 0)


# ---------------------------------------------------------------------------
# Lectura de respuestas de los SDK
# ---------------------------------------------------------------------------
def extraer_texto(resp) -> str:
    """Primer bloque de texto de una respuesta de Claude (content[0] puede ser
    un bloque "thinking" sin .text)."""
    for bloque in resp.content:
        if bloque.type == "text":
            return bloque.text
    raise ValueError(f"La respuesta no tiene ningún bloque de texto (stop_reason={resp.stop_reason!r})")


def extraer_texto_gemini(resp) -> str:
    """Texto de una respuesta de Gemini sin las partes de "pensamiento"
    (resp.text devuelve None cuando las hay)."""
    candidato = resp.candidates[0]
    partes = [p.text for p in candidato.content.parts if not getattr(p, "thought", False) and p.text]
    if not partes:
        raise ValueError(f"La respuesta no tiene ninguna parte de texto (finish_reason={candidato.finish_reason!r})")
    return "".join(partes)


# ---------------------------------------------------------------------------
# Costos: USD por millón de tokens (entrada, salida), según la documentación
# pública de cada proveedor (septiembre-octubre de 2026).
# ---------------------------------------------------------------------------
PRECIOS_USD_POR_MTOK = {
    "claude-sonnet-5-5": (2.00, 10.00),
    "gpt-6.1-sol": (2.00, 10.00),
    "gemini-3.8-flash": (0.75, 3.75),
    "deepseek-v4-pro": (1.32, 3.96),  # tarifa de hora pico; la de fuera de pico es la mitad
}


def costo_usd(modelo: str, tokens_entrada: int, tokens_salida: int) -> float:
    entrada, salida = PRECIOS_USD_POR_MTOK[modelo]
    return (tokens_entrada * entrada + tokens_salida * salida) / 1_000_000


# ---------------------------------------------------------------------------
# Ruta B (Fine-Tuning): modelo base de pesos abiertos, QLoRA 4-bit
# ---------------------------------------------------------------------------
FT_MODELO_BASE = os.environ.get("FT_MODELO_BASE", "unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
FT_MAX_SEQ_LENGTH = 2048
FT_LORA_R = 16
FT_LORA_ALPHA = 16
FT_LORA_DROPOUT = 0.0
FT_LEARNING_RATE = 2e-4
FT_NUM_EPOCHS = 3
FT_BATCH_SIZE = 2
FT_GRAD_ACCUM = 4

# Servidor local del modelo (src/fase3_ruta_finetuning/servidor_ft.py, en WSL
# con la GPU). Vacío = el modelo se ejecuta en el mismo proceso. No se define
# en .env porque WSL lee el mismo archivo; la interfaz de validación lo activa sola.
FT_SERVIDOR_URL = os.environ.get("FT_SERVIDOR_URL", "")

# ---------------------------------------------------------------------------
# Preprocesamiento, embeddings y recuperación (Ruta A)
# ---------------------------------------------------------------------------
CHUNK_SIZE_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 64

EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

RETRIEVER_TOP_K = 5          # candidatos iniciales
RETRIEVER_TOP_K_FINAL = 3    # fragmentos que llegan al generador

# Recuperación híbrida: puntaje = w * BM25 + (1 - w) * similitud vectorial, ambos
# normalizados a [0, 1] por consulta. Pesos iguales, sin ajuste manual: sobre los
# 100 casos de registros el acierto de recuperación (top-3) pasa de 12 % a 100 %
# frente a la búsqueda solo vectorial (ver src/busqueda_lexica.py).
RETRIEVER_PESO_LEXICO = 0.5

# ---------------------------------------------------------------------------
# Protocolo de evaluación (Fase 5, Tabla 3 del artículo)
# ---------------------------------------------------------------------------
K_REPETICIONES_JUEZ = 3
MAX_STD_ACEPTABLE = 0.05          # dispersión máxima entre repeticiones del juez
MAX_DESFASE_POSICION_PCT = 2.0    # sesgo de posición máximo aceptado (%)
UMBRAL_ALUCINACION_FAITHFULNESS = 0.50
ALPHA_SIGNIFICANCIA = 0.05
