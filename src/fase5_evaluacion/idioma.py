"""
Idioma de las figuras. Con la variable de entorno IDIOMA=en los scripts de figuras escriben los rótulos en inglés,
con punto decimal, y agregan el sufijo _en al nombre del archivo; por defecto (es) son los del manuscrito en español.
"""
import os

IDIOMA = os.environ.get("IDIOMA", "es").lower()
SUFIJO = "" if IDIOMA == "es" else f"_{IDIOMA}"

_EN = {
    "Lenguaje natural": "Natural language",
    "Semiestructurado": "Semi-structured",
    "Fine-Tuning": "Fine-tuning",
    "Qwen\n+ RAG": "Qwen\n+ RAG",
    "Qwen\nsin ajustar": "Qwen\nunadapted",
    "Qwen\najustado": "Fine-tuned\nQwen",
    "RAG comercial\n− ajustado": "Commercial RAG\n− fine-tuned",
    "Qwen+RAG\n− ajustado": "Qwen+RAG\n− fine-tuned",
    "Ajustado − Qwen\nsin ajustar": "Fine-tuned −\nunadapted Qwen",
    "Media con IC 95 %": "Mean with 95% CI",
    "Diferencia por caso": "Per-case difference",
    "Abajo: ": "Bottom: ",
    ". Los tres primeros cuadros de cada panel son de manuales y los tres últimos de registros. *** p < 0,001, ** p < 0,01, * p < 0,05, n.s. no significativo (Holm).":
        ". The first three boxes of each panel are manuals and the last three records. *** p < 0.001, ** p < 0.01, * p < 0.05, n.s. not significant (Holm).",
    "Costo real (USD)": "Actual cost (USD)",
    "Total de la campaña (API): $": "Campaign total (API): $",
    "llamadas": "calls",
    "Juez DeepSeek-V4-Pro (2 400 juicios)": "DeepSeek-V4-Pro judge (2,400 judgments)",
    "Dataset de entrenamiento (Claude Sonnet 5.5)": "Training dataset (Claude Sonnet 5.5)",
    "Generador RAG · Claude": "RAG generator · Claude",
    "Generador RAG · Gemini": "RAG generator · Gemini",
    "Generador RAG · ChatGPT": "RAG generator · ChatGPT",
    "Juez: condiciones de control": "Judge: control conditions",
    "Juez: fidelidad frente a fragmentos recibidos": "Judge: faithfulness vs. received fragments",
    "Juez: auditoría del dataset sintético": "Judge: synthetic dataset audit",
    "Fine-Tuning y controles con Qwen2.5-7B\nGPU local, sin costo de API": "Fine-tuning and Qwen2.5-7B controls\nlocal GPU, no API cost",
    "Proporción de respuestas que el participante\nmarcó como coincidentes (IC de Wilson 95 %)":
        "Proportion of responses the participant\nmarked as matching (95% Wilson CI)",
    "(a) Coincidencia con el manual": "(a) Agreement with the manual",
    "Confianza para\nejecutar la acción": "Confidence to\nact on the answer",
    "Advertencia ante\ninfo. ausente": "Warning when\ninformation is missing",
    "Tiempo de espera\npercibido": "Perceived\nwaiting time",
    "Media (1 a 5) con IC 95 %": "Mean (1 to 5) with 95% CI",
    "(b) Escala Likert": "(b) Likert scale",
    "Consultas en que fue la preferida": "Queries where it was preferred",
    "(c) Preferencia de uso": "(c) Preferred for use",
}


def T(texto: str) -> str:
    """Traduce un rótulo (si IDIOMA es en) y deja intacto el texto sin traducción."""
    return _EN.get(texto, texto) if IDIOMA == "en" else texto


def dec(texto: str) -> str:
    """Separador decimal del idioma (coma en español, punto en inglés)."""
    return texto.replace(".", ",") if IDIOMA == "es" else texto


def miles(n: int) -> str:
    """Separador de miles (espacio en español, coma en inglés)."""
    return f"{n:,}".replace(",", " " if IDIOMA == "es" else ",")
