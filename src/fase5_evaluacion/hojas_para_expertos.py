"""
Genera las hojas de Excel que completan personas expertas del dominio (no se pueden automatizar):

  1. concordancia_expertos_plantilla.xlsx   40 casos (20 por tipo de dato) x 4 respuestas, sin indicar de qué
                                             configuración viene cada una y en orden mezclado. Cada experto marca
                                             si la respuesta es correcta y si inventa información.
  2. referencia_manuales_plantilla.xlsx      Para los 100 casos de manuales: los 5 fragmentos que recupera el
                                             sistema; la persona marca cuáles respaldan la respuesta de referencia.
  3. auditoria_dataset_plantilla.xlsx        60 pares de pregunta y respuesta del dataset de entrenamiento
                                             (30 por tipo de dato) con su fragmento fuente.

También escribe clave_concordancia.json, que NO debe entregarse a los expertos (relaciona cada fila con su
configuración). acuerdo_expertos.py calcula los resultados a partir de las hojas completadas.

Salida: resultados/hojas_para_expertos/. Semilla fija: 42.

Uso:
    python src/fase5_evaluacion/hojas_para_expertos.py
"""
import json
import random
import sys
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

DESTINO = config.RESULTADOS_DIR / "hojas_para_expertos"
SEMILLA = 42
CAMPOS = {"RAG_Claude": "respuesta_rag_claude", "RAG_GPT": "respuesta_rag_gpt",
          "RAG_Gemini": "respuesta_rag_gemini", "FT_Qwen": "respuesta_ft"}
NOMBRE_SUBSET = {"A": "Manual (lenguaje natural)", "B": "Registro de incidencia"}
VACIO = PatternFill("solid", fgColor="FFF2CC")


def libro(encabezados: list[str], filas: list[list], anchos: list[int], columnas_vacias: list[int]) -> Workbook:
    wb = Workbook()
    ws = wb.active
    ws.append(encabezados)
    for c in ws[1]:
        c.font = Font(bold=True)
        c.alignment = Alignment(wrap_text=True, vertical="top")
    for fila in filas:
        ws.append(fila)
    for i, ancho in enumerate(anchos, start=1):
        ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = ancho
    for fila in ws.iter_rows(min_row=2):
        for c in fila:
            c.alignment = Alignment(wrap_text=True, vertical="top")
        for i in columnas_vacias:
            fila[i].fill = VACIO
    ws.freeze_panes = "A2"
    return wb


def hoja_concordancia(casos: list[dict], rng: random.Random):
    completos = [c for c in casos if all(c.get(v) for v in CAMPOS.values())]
    muestra = []
    for s in ("A", "B"):
        muestra += rng.sample([c for c in completos if c["subset"] == s], 20)
    filas, clave = [], {}
    for c in muestra:
        for arq, campo in CAMPOS.items():
            filas.append({"caso_id": c["caso_id"], "arquitectura": arq, "subset": c["subset"],
                          "pregunta": c["pregunta"], "referencia": c["respuesta_referencia"], "respuesta": c[campo]})
    rng.shuffle(filas)  # mezcla todas las filas: una persona no ve juntas las 4 respuestas de un caso
    salida = []
    for i, f in enumerate(filas, start=1):
        codigo = f"F{i:03d}"
        clave[codigo] = {"caso_id": f["caso_id"], "arquitectura": f["arquitectura"]}
        salida.append([codigo, NOMBRE_SUBSET[f["subset"]], f["pregunta"], f["referencia"], f["respuesta"], None, None, None])
    wb = libro(["codigo", "tipo de dato", "pregunta", "respuesta de referencia (correcta)", "respuesta a evaluar",
                "correcta (1 = sí, 0 = no)", "inventa información (1 = sí, 0 = no)", "comentario"],
               salida, [8, 16, 45, 60, 70, 14, 16, 30], [5, 6, 7])
    wb.save(DESTINO / "concordancia_expertos_plantilla.xlsx")
    json.dump(clave, open(DESTINO / "clave_concordancia.json", "w", encoding="utf-8"), indent=1)
    return len(salida)


def hoja_referencia(casos: list[dict]):
    from fase3_ruta_rag import recuperar  # requiere el índice de ChromaDB

    filas = []
    for c in [c for c in casos if c["subset"] == "A"]:
        cand = recuperar(c["pregunta"], top_k=5, top_k_final=5)
        for pos, f in enumerate(cand, start=1):
            filas.append([c["caso_id"], c["pregunta"], c["respuesta_referencia"], pos, f["chunk_id"],
                          "sí" if f["text"] == c.get("fragmento_fuente") else "", f["text"][:2500], None])
    wb = libro(["caso_id", "pregunta", "respuesta de referencia", "posición recuperada", "id del fragmento",
                "fragmento c actual", "texto del fragmento", "respalda la respuesta (1 = sí, 0 = no)"],
               filas, [9, 40, 55, 11, 30, 10, 90, 16], [7])
    wb.save(DESTINO / "referencia_manuales_plantilla.xlsx")
    return len(filas)


def hoja_auditoria(rng: random.Random):
    pares = [json.loads(l) for l in open(config.DATA_PROCESSED / "dataset_qa_sintetico.jsonl", encoding="utf-8")]
    textos = {}
    for nombre in ("corpus_A.jsonl", "corpus_B.jsonl"):
        for linea in open(config.DATA_PROCESSED / nombre, encoding="utf-8"):
            c = json.loads(linea)
            textos[c["chunk_id"]] = c["text"]
    muestra = []
    for s in ("A", "B"):
        muestra += rng.sample([p for p in pares if p["subset"] == s], 30)
    rng.shuffle(muestra)
    filas = [[f"P{i:03d}", NOMBRE_SUBSET[p["subset"]], textos[p["chunk_id"]][:3000], p["pregunta"], p["respuesta"], None, None, None]
             for i, p in enumerate(muestra, start=1)]
    wb = libro(["codigo", "tipo de dato", "fragmento fuente", "pregunta", "respuesta",
                "respaldo en el fragmento (total / parcial / ninguno)", "pregunta respondible con el fragmento (1 = sí, 0 = no)",
                "comentario"], filas, [8, 16, 90, 40, 60, 18, 18, 30], [5, 6, 7])
    wb.save(DESTINO / "auditoria_dataset_plantilla.xlsx")
    json.dump({f[0]: {"chunk_id": p["chunk_id"], "pregunta": p["pregunta"]} for f, p in zip(filas, muestra)},
              open(DESTINO / "clave_auditoria.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    return len(filas)


def main():
    DESTINO.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEMILLA)
    casos = [json.loads(l) for l in open(config.DATA_PROCESSED / "resultados_respuestas.jsonl", encoding="utf-8")]
    print("concordancia_expertos_plantilla.xlsx:", hoja_concordancia(casos, rng), "filas")
    print("referencia_manuales_plantilla.xlsx:", hoja_referencia(casos), "filas")
    print("auditoria_dataset_plantilla.xlsx:", hoja_auditoria(rng), "filas")
    print("Las claves (clave_*.json) son para los autores; no las entreguen a los expertos.")


if __name__ == "__main__":
    main()
