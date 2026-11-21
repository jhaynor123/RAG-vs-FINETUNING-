"""
Calcula los resultados que dependen de las hojas completadas por expertos (ver hojas_para_expertos.py).

  --concordancia A.xlsx [B.xlsx]   κ de Cohen entre cada experto y el juez automático, y entre los dos expertos.
                                   Binarización del juez: correcta = Answer Correctness >= 0,5;
                                   inventa información = Faithfulness < 0,5 (umbral de alucinación).
                                   Criterio de aceptación (Tabla 3 del artículo): κ >= 0,70.
  --referencia hoja.xlsx           Acierto de recuperación (top-1, top-3, top-5 y rango recíproco medio) frente
                                   a los fragmentos que la persona marcó como respaldo, y cuántas veces el
                                   fragmento c asignado automáticamente es uno de ellos.
  --auditoria hoja.xlsx            Porcentaje de pares del dataset con respaldo total y con pregunta respondible.

Salida: resultados/acuerdo_expertos.json (se acumula entre ejecuciones).

Uso:
    python src/fase5_evaluacion/acuerdo_expertos.py --concordancia exp1.xlsx exp2.xlsx
    python src/fase5_evaluacion/acuerdo_expertos.py --referencia referencia_completada.xlsx
    python src/fase5_evaluacion/acuerdo_expertos.py --auditoria auditoria_completada.xlsx
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

DESTINO = config.RESULTADOS_DIR / "acuerdo_expertos.json"
HOJAS = config.RESULTADOS_DIR / "hojas_para_expertos"


def kappa(a, b) -> float | None:
    """κ de Cohen entre dos clasificaciones de las mismas unidades."""
    a, b = list(a), list(b)
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return None if pe == 1 else (po - pe) / (1 - pe)


def leer_hoja(ruta: Path, columnas: list[str]) -> pd.DataFrame:
    ws = load_workbook(ruta).active
    filas = list(ws.values)
    df = pd.DataFrame(filas[1:], columns=[str(c).strip() for c in filas[0]])
    return df[columnas]


def binario(x):
    return None if x is None or str(x).strip() == "" else int(float(x))


def concordancia(rutas: list[Path]) -> dict:
    clave = json.load(open(HOJAS / "clave_concordancia.json", encoding="utf-8"))
    m = pd.read_csv(config.RESULTADOS_DIR / "matriz_rendimiento.csv").set_index(["caso_id", "arquitectura"])
    expertos = []
    for ruta in rutas:
        df = leer_hoja(ruta, ["codigo", "correcta (1 = sí, 0 = no)", "inventa información (1 = sí, 0 = no)"])
        df.columns = ["codigo", "correcta", "inventa"]
        df["correcta"], df["inventa"] = df.correcta.map(binario), df.inventa.map(binario)
        if df[["correcta", "inventa"]].isna().any().any():
            sys.exit(f"{ruta.name}: hay filas sin completar.")
        j = [m.loc[(clave[c]["caso_id"], clave[c]["arquitectura"])] for c in df.codigo]
        df["juez_correcta"] = [int(x.answer_correctness_media >= 0.5) for x in j]
        df["juez_inventa"] = [int(x.faithfulness_media < config.UMBRAL_ALUCINACION_FAITHFULNESS) for x in j]
        df["arquitectura"] = [clave[c]["arquitectura"] for c in df.codigo]
        expertos.append(df)
    out = {"n_respuestas": len(expertos[0]), "criterio_kappa_minimo": 0.70, "expertos": []}
    for k, df in enumerate(expertos, start=1):
        e = {"experto": k, "kappa_correcta_vs_juez": kappa(df.correcta, df.juez_correcta),
             "kappa_inventa_vs_juez": kappa(df.inventa, df.juez_inventa),
             "kappa_correcta_vs_juez_por_configuracion": {a: kappa(g.correcta, g.juez_correcta) for a, g in df.groupby("arquitectura")}}
        out["expertos"].append(e)
    if len(expertos) == 2:
        out["kappa_entre_expertos_correcta"] = kappa(expertos[0].correcta, expertos[1].correcta)
        out["kappa_entre_expertos_inventa"] = kappa(expertos[0].inventa, expertos[1].inventa)
    return out


def referencia(ruta: Path) -> dict:
    df = leer_hoja(ruta, ["caso_id", "posición recuperada", "fragmento c actual", "respalda la respuesta (1 = sí, 0 = no)"])
    df.columns = ["caso", "pos", "es_c", "respalda"]
    df["respalda"] = df.respalda.map(binario)
    if df.respalda.isna().any():
        sys.exit("Hay filas sin completar.")
    hit = {"top1": [], "top3": [], "top5": [], "rr": [], "c_respalda": []}
    for _, g in df.groupby("caso"):
        pos = sorted(g[g.respalda == 1].pos)
        hit["top1"].append(int(bool(pos) and pos[0] == 1))
        hit["top3"].append(int(bool(pos) and pos[0] <= 3))
        hit["top5"].append(int(bool(pos)))
        hit["rr"].append(1 / pos[0] if pos else 0.0)
        c = g[g.es_c == "sí"]
        if len(c):
            hit["c_respalda"].append(int(c.respalda.iloc[0] == 1))
    n = len(hit["top1"])
    return {"n_casos": n, "acierto_top1": sum(hit["top1"]) / n, "acierto_top3": sum(hit["top3"]) / n,
            "acierto_top5": sum(hit["top5"]) / n, "rango_reciproco_medio": float(np.mean(hit["rr"])),
            "casos_donde_c_respalda": f"{sum(hit['c_respalda'])}/{len(hit['c_respalda'])}"}


def auditoria(ruta: Path) -> dict:
    df = leer_hoja(ruta, ["codigo", "respaldo en el fragmento (total / parcial / ninguno)",
                          "pregunta respondible con el fragmento (1 = sí, 0 = no)"])
    df.columns = ["codigo", "respaldo", "respondible"]
    if df.isna().any().any():
        sys.exit("Hay filas sin completar.")
    df["respaldo"] = df.respaldo.str.strip().str.lower()
    n = len(df)
    c = Counter(df.respaldo)
    return {"n_pares": n, "respaldo": dict(c), "pct_respaldo_total": round(100 * c["total"] / n, 1),
            "pct_pregunta_respondible": round(100 * df.respondible.map(binario).mean(), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concordancia", nargs="+", type=Path)
    ap.add_argument("--referencia", type=Path)
    ap.add_argument("--auditoria", type=Path)
    args = ap.parse_args()
    previo = json.load(open(DESTINO, encoding="utf-8")) if DESTINO.exists() else {}
    if args.concordancia:
        previo["concordancia_juez_expertos"] = concordancia(args.concordancia)
    if args.referencia:
        previo["acierto_de_recuperacion_manuales"] = referencia(args.referencia)
    if args.auditoria:
        previo["auditoria_humana_dataset"] = auditoria(args.auditoria)
    json.dump(previo, open(DESTINO, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    print(json.dumps(previo, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
