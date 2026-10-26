"""
Filtro de idioma + limpieza específica por documento, para la Fase 1.

Los manuales pueden traer varios idiomas mezclados (típico en manuales
de maquinaria importada: español/inglés/francés/portugués repetidos por
sección, o páginas completas en otro idioma). Este módulo:

  1) Divide el texto extraído en bloques (párrafos).
  2) Detecta el idioma de cada bloque y descarta los que no sean español.
  3) Aplica una limpieza específica por archivo (cada manual puede traer
     un tipo de "ruido" distinto: pies de página repetidos, numeración,
     tablas de contenido, marcas de agua, etc.) — configurable en
     LIMPIADORES_POR_ARCHIVO más abajo.

Los bloques muy cortos (menos de MIN_CHARS_DETECCION) no son confiables
para detección de idioma automática; se conservan por defecto en vez de
descartarlos a ciegas (ej. un título corto, un código de pieza), pero se
puede ajustar el comportamiento con `conservar_bloques_cortos`.
"""
import re
from langdetect import DetectorFactory, LangDetectException, detect

DetectorFactory.seed = 0  # resultados reproducibles

MIN_CHARS_DETECCION = 25


# ---------------------------------------------------------------------------
# Reparación de glifos rotos tipo "(cid:N)".
#
# Algunos PDF (ej. el manual JUKI LU-2216/2266N-7, que trae 6 idiomas en el
# mismo archivo) incrustan una fuente cuyo subconjunto de letras latinas
# simples (a-z, A-Z, dígitos, puntuación básica) quedó desplazado por un
# offset fijo respecto al código ASCII real: el glifo "N" del PDF equivale
# al carácter chr(N + 29), NO al carácter N. Esto se verificó de forma
# objetiva (no a ojo): se tomaron ~9.500 apariciones de "(cid:N)" dentro de
# palabras, se sustituyó cada una por chr(N+29) y se comprobó qué fracción
# de las palabras reconstruidas existe en un diccionario real (español,
# tomado del propio corpus limpio, o inglés vía nltk) -- ~68-92% de acierto
# según la letra, y el margen de error restante es sobre todo el propio
# diccionario de referencia (es incompleto), no el offset.
#
# El resto de los códigos "(cid:N)" que aparecen en este mismo documento
# (más de 1.000 códigos distintos) NO siguen este patrón: corresponden a
# las secciones en japonés/chino/otros alfabetos del mismo manual
# multi-idioma, que el filtro de idioma de este módulo debe descartar de
# todos modos. Para esos casos no se adivina ningún carácter -- se elimina
# el marcador "(cid:N)" sin más, en vez de arriesgar insertar un carácter
# incorrecto en lo que se presenta como corpus en español.
# ---------------------------------------------------------------------------
_PATRON_CID = re.compile(r"\(cid:(\d+)\)")

# Los pocos códigos "(cid:N)" que SÍ aparecen dentro de texto en español y no
# siguen el offset anterior: letras con tilde y ligaturas. Cada uno se
# verificó contra el diccionario español (wordfreq) y contra el contexto:
#   126 -> ú  "Aseg(cid:126)rese"        105 -> á  "m(cid:105)quina"
#   121 -> ó  "indicaci(cid:121)n"        112 -> é  "el(cid:112)ctricas"
#   191 -> fi "signi(cid:191)ca"          192/193 -> fl "in(cid:192)amaci(ó)n", "a(cid:193)oje"
# Sin esto, esas letras se perdían ("asegrese", "elctricas", "signica").
_CID_ESPECIALES = {126: "ú", 121: "ó", 105: "á", 112: "é", 191: "fi", 192: "fl", 193: "fl"}


def _offset_29(num: int) -> str | None:
    codigo = num + 29
    return chr(codigo) if 32 <= codigo <= 126 else None


def reparar_cid_rotos(texto: str) -> str:
    """Corrige los glifos "(cid:N)" reconstruibles (offset verificado o
    tabla de acentos/ligaturas) y elimina, sin adivinar, los que no lo son."""

    def _reemplazo(m: re.Match) -> str:
        num = int(m.group(1))
        if num in _CID_ESPECIALES:
            return _CID_ESPECIALES[num]
        caracter = _offset_29(num)
        return caracter if caracter is not None else ""

    return _PATRON_CID.sub(_reemplazo, texto)


def _es_bloque_español(bloque: str) -> bool:
    if len(bloque) < MIN_CHARS_DETECCION:
        return True  # se decide en la capa de arriba (conservar_bloques_cortos)
    try:
        return detect(bloque) == "es"
    except LangDetectException:
        return True  # no se pudo determinar -> se conserva, mejor pecar de incluir


def filtrar_bloques_español(bloques: list[str], conservar_bloques_cortos: bool = True) -> tuple[list[str], dict]:
    """Descarta los bloques que no están en español. Recibe la lista de
    elementos que entrega Unstructured.io (título, párrafo, fila de tabla...),
    sin depender de que el texto conserve líneas en blanco entre párrafos."""
    conservados = []
    descartados = 0
    for b in bloques:
        b = b.strip()
        if not b:
            continue
        if len(b) < MIN_CHARS_DETECCION:
            if conservar_bloques_cortos:
                conservados.append(b)
            else:
                descartados += 1
            continue
        if _es_bloque_español(b):
            conservados.append(b)
        else:
            descartados += 1

    stats = {
        "bloques_totales": len(conservados) + descartados,
        "bloques_conservados": len(conservados),
        "bloques_descartados_por_idioma": descartados,
    }
    return conservados, stats


# ---------------------------------------------------------------------------
# Limpieza específica por archivo.
#
# Cada manual puede traer un ruido distinto (pies de página, numeración, tablas
# de contenido). Aquí se registra, por nombre de archivo (coincidencia parcial,
# sin distinguir mayúsculas), una lista de expresiones regulares a eliminar,
# además de la limpieza genérica de abajo. En este estudio no se usó ninguno
# (solo la limpieza genérica y la reparación de glifos), por eso está vacío.
# ---------------------------------------------------------------------------
LIMPIADORES_POR_ARCHIVO: dict[str, list[str]] = {}

_LIMPIEZA_GENERICA = [
    r"^\s*Page\s+\d+\s*$",
    r"^\s*Página\s+\d+\s*$",
    r"^\s*\d+\s*$",  # líneas que son solo un número de página
]


def limpiar_especifico(texto: str, nombre_archivo: str) -> str:
    patrones = list(_LIMPIEZA_GENERICA)
    nombre_lower = nombre_archivo.lower()
    for clave, extra in LIMPIADORES_POR_ARCHIVO.items():
        if clave.lower() in nombre_lower:
            patrones.extend(extra)

    lineas = texto.splitlines()
    resultado = []
    for linea in lineas:
        if any(re.match(p, linea.strip(), flags=re.IGNORECASE) for p in patrones):
            continue
        resultado.append(linea)
    return "\n".join(resultado)
