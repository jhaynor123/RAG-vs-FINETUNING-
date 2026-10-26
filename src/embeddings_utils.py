"""
Utilidad compartida para generar embeddings (Fase 2, Ruta A y métrica
Answer Relevance de la Fase 5).

Carga el modelo definido en config.EMBEDDING_MODEL (sentence-transformers).
Si el modelo no se puede cargar, falla con un error explícito: NO existe un
embedding de respaldo, porque uno sin entrenamiento semántico produciría
una recuperación sin sentido sin que el experimento avisara.

La carga está protegida con un candado: si varios hilos piden embeddings a
la vez, el modelo se carga una sola vez (cargarlo en paralelo falla con
"Cannot copy out of meta tensor").
"""
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

_modelo = None
_candado = threading.Lock()


def _cargar_modelo():
    global _modelo
    if _modelo is None:
        with _candado:
            if _modelo is None:
                from sentence_transformers import SentenceTransformer

                _modelo = SentenceTransformer(config.EMBEDDING_MODEL)
                print(f"[embeddings_utils] modelo cargado: {config.EMBEDDING_MODEL}")
    return _modelo


def embed_textos(textos: list[str]) -> list[list[float]]:
    """Devuelve un embedding por cada texto de la lista."""
    return _cargar_modelo().encode(textos, normalize_embeddings=True).tolist()
