"""Prompts compartidos por los generadores de la Ruta A y por sus condiciones de control."""

PROMPT_GENERADOR = """Eres un asistente técnico de una planta textil. Responde la consulta del usuario basándote ÚNICAMENTE en los fragmentos de contexto que se muestran abajo. Si el contexto no contiene la información necesaria, dilo explícitamente en vez de inventar una respuesta.

Contexto recuperado:
{contexto}

Consulta: {consulta}

Responde de forma directa y precisa, citando los códigos o cifras exactas cuando el contexto los mencione."""


def formatear_contexto(fragmentos: list[dict]) -> str:
    """Contexto tal como se envía al generador: cada fragmento con su archivo de origen."""
    return "\n\n---\n\n".join(f"[Fuente: {f['source_file']}]\n{f['text']}" for f in fragmentos)
