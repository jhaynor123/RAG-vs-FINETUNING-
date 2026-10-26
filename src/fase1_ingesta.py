"""
Fase 1 — Adquisición y preprocesamiento de datos.

Ingiere los dos subconjuntos documentales (manuales técnicos en
lenguaje natural, y tickets semiestructurados), limpia el ruido con
Unstructured.io, y segmenta el contenido en chunks semánticos con
LlamaIndex (512 tokens, con solapamiento).

Salida: dos archivos JSONL en data/processed/
  - corpus_A.jsonl  (Subconjunto A: lenguaje natural)
  - corpus_B.jsonl  (Subconjunto B: semiestructurado)

Cada línea: {"chunk_id", "source_file", "subset", "text"}

Uso:
    python src/fase1_ingesta.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from limpieza_idioma import filtrar_bloques_español, limpiar_especifico, reparar_cid_rotos

from unstructured.partition.auto import partition
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.schema import Document


def limpiar_texto(texto: str) -> str:
    """Elimina líneas vacías repetidas y espacios redundantes que suele
    dejar la extracción, sin tocar el contenido real (códigos, cifras)."""
    lineas = [l.strip() for l in texto.splitlines()]
    lineas = [l for l in lineas if l]
    return "\n".join(lineas)


def ingerir_archivo(path: Path, filtrar_idioma: bool = False) -> str:
    """Usa Unstructured.io para extraer texto limpio de un PDF/Word/Excel/CSV,
    descartando encabezados/pies de página repetidos y marcas de agua
    cuando el formato lo permite.

    Si filtrar_idioma=True (se usa para los 5 manuales, que pueden traer
    varios idiomas mezclados), además: 1) aplica la limpieza específica de
    ese archivo (ver limpieza_idioma.LIMPIADORES_POR_ARCHIVO), y 2) descarta
    los bloques que no estén en español.
    """
    # strategy="fast" evita el pipeline de detección de layout (hi_res),
    # que en esta máquina termina en segmentation fault; no hace falta de
    # todos modos porque estos PDFs traen texto real embebido, no son
    # escaneos que necesiten OCR.
    elementos = partition(filename=str(path), strategy="fast")
    bloques = [el.text.strip() for el in elementos if getattr(el, "text", "").strip()]
    # Repara glifos rotos tipo "(cid:N)" (ver limpieza_idioma.reparar_cid_rotos)
    # antes de cualquier otro paso, para que la detección de idioma de abajo
    # también trabaje sobre texto limpio.
    bloques = [reparar_cid_rotos(b) for b in bloques]

    if filtrar_idioma:
        # limpieza específica del archivo, bloque por bloque (antes de decidir idioma,
        # por si el ruido -numeración, pies de página- confunde al detector)
        bloques = [limpiar_especifico(b, path.name) for b in bloques]
        bloques = [b for b in bloques if b.strip()]
        bloques, stats = filtrar_bloques_español(bloques)
        print(
            f"    [idioma] {stats['bloques_conservados']}/{stats['bloques_totales']} bloques en "
            f"español ({stats['bloques_descartados_por_idioma']} descartados por otro idioma)"
        )

    texto = limpiar_texto("\n".join(bloques))
    return texto


def ingerir_json_chat(path: Path) -> list[dict]:
    """Ingesta específica para datos semiestructurados en formato chat JSONL
    ({"messages": [...]})), como el que ya tienes en
    data/raw/qa_reales/DATA_SEMIESTRUCTURADA.json.

    Cada línea se convierte en UN documento fuente para el índice RAG: el
    contenido del mensaje "assistant" (la interpretación estructurada del
    ticket), con el código de ticket -tomado del mensaje "user"- como
    encabezado, para que quede recuperable por código exacto.

    Devuelve una lista de {"texto", "origen"} -- uno por línea del JSON --
    en vez de un solo texto largo, porque cada ticket es una unidad de
    información independiente y NO conviene fusionarlos antes de chunkear
    (evita mezclar dos tickets distintos en un mismo chunk por casualidad
    de tamaño).
    """
    documentos = []
    with open(path, encoding="utf-8") as f:
        for i, linea in enumerate(f):
            linea = linea.strip()
            if not linea:
                continue
            d = json.loads(linea)
            mensajes = {m["role"]: m["content"] for m in d["messages"]}
            pregunta = mensajes.get("user", "")
            respuesta = mensajes.get("assistant", "")
            documentos.append(
                {
                    "texto": f"{pregunta}\n{respuesta}",
                    "origen": f"{path.name}#{i:04d}",
                }
            )
    return documentos


def chunkear(texto: str, source_file: str, subset: str) -> list[dict]:
    splitter = TokenTextSplitter(
        chunk_size=config.CHUNK_SIZE_TOKENS,
        chunk_overlap=config.CHUNK_OVERLAP_TOKENS,
    )
    doc = Document(text=texto, metadata={"source_file": source_file, "subset": subset})
    nodes = splitter.get_nodes_from_documents([doc])
    return [
        {
            "chunk_id": f"{subset}_{source_file}_{i:04d}",
            "source_file": source_file,
            "subset": subset,
            "text": n.get_content(),
        }
        for i, n in enumerate(nodes)
    ]


def procesar_carpeta_manuales(carpeta: Path) -> list[dict]:
    """Subconjunto A: los 5 manuales. Aplica filtro de idioma (español) y
    limpieza específica por archivo, ya que suelen traer varios idiomas
    mezclados."""
    chunks = []
    archivos = sorted([p for p in carpeta.glob("*") if p.is_file() and not p.name.startswith(".")])
    if not archivos:
        print(f"  (sin archivos en {carpeta} — coloca aquí tus 5 manuales; ver data/raw/LEEME.txt)")
        return chunks
    for path in archivos:
        print(f"  procesando {path.name} ...")
        texto = ingerir_archivo(path, filtrar_idioma=True)
        nuevos = chunkear(texto, path.name, "A")
        chunks.extend(nuevos)
        print(f"    -> {len(nuevos)} chunks")
    return chunks


def procesar_carpeta_semiestructurada(carpeta: Path) -> list[dict]:
    """Subconjunto B: datos semiestructurados. Soporta tanto JSON en
    formato chat ({"messages": [...]})-- como DATA_SEMIESTRUCTURADA.json --
    (un chunk por ticket, sin fusionar), como CSV/Excel de tickets."""
    chunks = []
    archivos = sorted([p for p in carpeta.glob("*") if p.is_file() and not p.name.startswith(".")])
    if not archivos:
        print(f"  (sin archivos en {carpeta} — coloca aquí tu data semiestructurada; ver data/raw/LEEME.txt)")
        return chunks

    for path in archivos:
        print(f"  procesando {path.name} ...")
        if path.suffix.lower() == ".json":
            documentos = ingerir_json_chat(path)
            for doc in documentos:
                nuevos = chunkear(doc["texto"], doc["origen"], "B")
                chunks.extend(nuevos)
            print(f"    -> {len(documentos)} registros -> {len(chunks)} chunks acumulados")
        else:
            texto = ingerir_archivo(path, filtrar_idioma=False)
            nuevos = chunkear(texto, path.name, "B")
            chunks.extend(nuevos)
            print(f"    -> {len(nuevos)} chunks")
    return chunks


def guardar_jsonl(chunks: list[dict], destino: Path):
    with open(destino, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"  guardado: {destino} ({len(chunks)} chunks)")


def main():
    print("=== Fase 1: Adquisición y preprocesamiento ===\n")

    print("Subconjunto A (lenguaje natural / 'estructurados' — 5 manuales):")
    corpus_a = procesar_carpeta_manuales(config.DATA_RAW_MANUALES)
    guardar_jsonl(corpus_a, config.DATA_PROCESSED / "corpus_A.jsonl")

    print("\nSubconjunto B (semiestructurado — tickets):")
    corpus_b = procesar_carpeta_semiestructurada(config.DATA_RAW_TICKETS)
    guardar_jsonl(corpus_b, config.DATA_PROCESSED / "corpus_B.jsonl")

    print(f"\nTotal: {len(corpus_a)} chunks (A) + {len(corpus_b)} chunks (B) = {len(corpus_a) + len(corpus_b)}")


if __name__ == "__main__":
    main()
