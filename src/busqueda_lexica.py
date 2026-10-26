"""
Búsqueda léxica BM25 (Okapi) para la recuperación híbrida de la Ruta A.

Los embeddings densos pierden los identificadores exactos: en una prueba
sobre los 100 casos de tickets, la recuperación puramente vectorial
encontraba el ticket correcto solo en el 12 % de los casos (top-3), porque
"ERR-LUB-01" y "ERR-STC-04" quedan casi igual de cerca en el espacio
vectorial. BM25 compara palabras exactas, así que los códigos
alfanuméricos sí discriminan.

El tokenizador conserva los códigos con guion completos ("err-lub-01") y
además sus partes ("err", "lub", "01").
"""
import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9áéíóúüñ]+(?:-[a-z0-9áéíóúüñ]+)*")


def tokenizar(texto: str) -> list[str]:
    salida = []
    for palabra in _TOKEN.findall(texto.lower()):
        salida.append(palabra)
        if "-" in palabra:
            salida.extend(palabra.split("-"))
    return salida


class BM25:
    def __init__(self, documentos: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self._tokens = [tokenizar(d) for d in documentos]
        self._tf = [Counter(t) for t in self._tokens]
        n = len(documentos)
        self._largo_medio = sum(len(t) for t in self._tokens) / n
        df = Counter()
        for t in self._tokens:
            df.update(set(t))
        self._idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}

    def puntajes(self, consulta: str) -> list[float]:
        """Puntaje BM25 de la consulta contra cada documento (mismo orden que el índice)."""
        consulta_tokens = tokenizar(consulta)
        salida = []
        for tf, tokens in zip(self._tf, self._tokens):
            largo = len(tokens)
            s = 0.0
            for w in consulta_tokens:
                f = tf.get(w)
                if f:
                    s += self._idf[w] * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * largo / self._largo_medio)
                    )
            salida.append(s)
        return salida
