"""
Auditoría automática del dataset sintético de la Ruta B (807 pares de pregunta y respuesta generados con
Claude a partir de los fragmentos del corpus).

Para cada par, el juez (DeepSeek, un proveedor distinto del que generó los pares) comprueba contra el
fragmento fuente si (a) la respuesta está respaldada por el fragmento y (b) la pregunta se puede responder con
él. Es una verificación automática, no una revisión experta: la revisión humana de una muestra se hace con
hojas_para_expertos.py y acuerdo_expertos.py.

Salidas (resultados/): auditoria_dataset_sintetico.csv, auditoria_dataset_sintetico.json y costo_juez_auditoria.json.
Es reanudable (auditoria_parciales.jsonl, archivo temporal).

Uso:
    python src/fase5_evaluacion/auditar_dataset_sintetico.py
    python src/fase5_evaluacion/auditar_dataset_sintetico.py --limite 10
"""
import argparse
import csv
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from metricas import USO_JUEZ, _llamar_json

PROMPT_AUDITORIA = """Eres un revisor de calidad de datos de entrenamiento. Se generó automáticamente un par de pregunta y respuesta a partir de un fragmento de documentación técnica. Comprueba el par contra el fragmento.

Fragmento fuente:
\"\"\"
{fragmento}
\"\"\"

Pregunta: {pregunta}

Respuesta: {respuesta}

Evalúa:
- "respaldo": "total" si todas las afirmaciones de la respuesta se pueden verificar en el fragmento; "parcial" si algunas sí y otras no; "ninguno" si la respuesta no se apoya en el fragmento o lo contradice.
- "pregunta_respondible": true si la pregunta se puede contestar con la información del fragmento; false si no.

Responde SOLO con JSON: {{"respaldo": "total" | "parcial" | "ninguno", "pregunta_respondible": true | false, "observacion": "una frase"}}"""


def cargar_fragmentos() -> dict:
    textos = {}
    for nombre in ("corpus_A.jsonl", "corpus_B.jsonl"):
        for linea in open(config.DATA_PROCESSED / nombre, encoding="utf-8"):
            c = json.loads(linea)
            textos[c["chunk_id"]] = c["text"]
    return textos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limite", type=int, default=None)
    args = ap.parse_args()
    if not config.DEEPSEEK_API_KEY:
        sys.exit("DEEPSEEK_API_KEY no configurada (ver .env.example).")

    from openai import OpenAI

    cliente = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)
    pares = [json.loads(l) for l in open(config.DATA_PROCESSED / "dataset_qa_sintetico.jsonl", encoding="utf-8")]
    for i, p in enumerate(pares):
        p["id"] = i
    if args.limite:
        pares = pares[: args.limite]
    textos = cargar_fragmentos()

    parciales = config.RESULTADOS_DIR / "auditoria_parciales.jsonl"
    hechos = {}
    if parciales.exists():
        for l in open(parciales, encoding="utf-8"):
            if l.strip():
                d = json.loads(l)
                hechos[d["id"]] = d
    pendientes = [p for p in pares if p["id"] not in hechos]

    def correr(p):
        d = _llamar_json(cliente, PROMPT_AUDITORIA.format(fragmento=textos[p["chunk_id"]], pregunta=p["pregunta"],
                                                          respuesta=p["respuesta"]), max_tokens=512)
        return {"id": p["id"], "chunk_id": p["chunk_id"], "subset": p["subset"], "respaldo": d.get("respaldo"),
                "pregunta_respondible": bool(d.get("pregunta_respondible")), "observacion": d.get("observacion", "")}

    errores = 0
    with ThreadPoolExecutor(max_workers=config.JUEZ_CONCURRENCIA) as pool, open(parciales, "a", encoding="utf-8") as salida:
        futuros = [pool.submit(correr, p) for p in pendientes]
        for fut in tqdm(as_completed(futuros), total=len(futuros), desc="Auditando pares"):
            try:
                d = fut.result()
            except Exception:
                errores += 1
                continue
            hechos[d["id"]] = d
            salida.write(json.dumps(d, ensure_ascii=False) + "\n")
            salida.flush()

    filas = [hechos[p["id"]] for p in pares if p["id"] in hechos]
    with open(config.RESULTADOS_DIR / "auditoria_dataset_sintetico.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "chunk_id", "subset", "respaldo", "pregunta_respondible", "observacion"])
        w.writeheader()
        w.writerows(filas)

    resumen = {"n_auditados": len(filas), "n_fallidos": errores}
    for ambito, sub in (("total", filas), ("A", [f for f in filas if f["subset"] == "A"]), ("B", [f for f in filas if f["subset"] == "B"])):
        c = Counter(f["respaldo"] for f in sub)
        n = len(sub)
        resumen[ambito] = {"n": n, "respaldo": dict(c), "pct_respaldo_total": round(100 * c["total"] / n, 1) if n else None,
                           "pct_pregunta_respondible": round(100 * sum(f["pregunta_respondible"] for f in sub) / n, 1) if n else None}
    json.dump(resumen, open(config.RESULTADOS_DIR / "auditoria_dataset_sintetico.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    u = dict(USO_JUEZ)
    u["modelo"] = config.DEEPSEEK_MODEL_JUEZ
    u["costo_usd_tarifa_pico"] = round(config.costo_usd(config.DEEPSEEK_MODEL_JUEZ, u["tokens_entrada"], u["tokens_salida"]), 4)
    json.dump(u, open(config.RESULTADOS_DIR / "costo_juez_auditoria.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(json.dumps(resumen, indent=1, ensure_ascii=False))
    print(f"Costo del juez: ${u['costo_usd_tarifa_pico']} USD")


if __name__ == "__main__":
    main()
