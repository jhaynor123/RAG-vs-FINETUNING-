"""
Consolida los registros de la validación con usuarios (los que escribe app.py en
resultados/validacion_usuarios/) y calcula las cifras de la Tabla 6 y de la Fig. 6.

Salidas:
    resultados/validacion_usuarios/consolidado/resumen_real.json
    resultados/validacion_usuarios/consolidado/estadistica_validacion.json
    resultados/figuras/fig6_validacion.png

Uso:
    python src/interfaz_validacion/consolidar_validacion.py
    python src/interfaz_validacion/consolidar_validacion.py --interacciones otro.jsonl --parte-a otro_a.jsonl
"""
import argparse
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats
from statsmodels.stats.contingency_tables import cochrans_q, mcnemar

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "fase5_evaluacion"))
import config
from graficar_resultados import CONFIGS

ORIGEN = config.RESULTADOS_DIR / "validacion_usuarios"
DESTINO = ORIGEN / "consolidado"
CONSTRUCTOS = (
    ("utilidad_percibida", ["UP1", "UP2", "UP3", "UP4"]),
    ("facilidad_de_uso", ["FU1", "FU2", "FU3"]),
    ("intencion_de_uso", ["IU1", "IU2"]),
)


def leer(ruta: Path) -> list[dict]:
    if not ruta.exists():
        return []
    return [json.loads(l) for l in open(ruta, encoding="utf-8") if l.strip()]


def estadistica(inter: list[dict], parte_a: list[dict]) -> dict:
    """Pruebas de la validación (cada consulta es un bloque con las 4 respuestas):
    Friedman en las escalas de 1 a 5 y en el tiempo medido, Q de Cochran y McNemar en
    la coincidencia, bondad de ajuste en la preferencia y alfa de Cronbach de la Parte A.
    Los bloques vienen de 5 personas, así que los valores p son orientativos."""
    cfgs = ["RAG_GPT", "RAG_Claude", "RAG_Gemini", "FT_Qwen"]

    def matriz(campo):
        return np.array([[next(r[campo] for r in i["respuestas"] if r["config"] == c) for c in cfgs] for i in inter], dtype=float)

    out = {}
    for campo in ("confianza", "transparencia", "tiempo_percibido"):
        m = matriz(campo)
        chi2, p = stats.friedmanchisquare(*m.T)
        rag, ft = m[:, :3].mean(1), m[:, 3]
        out[campo] = {
            "medias": dict(zip(cfgs, m.mean(0).round(2))),
            "chi2": round(chi2, 2), "p": float(p), "W": round(chi2 / (len(m) * 3), 3),
            "wilcoxon_RAG_vs_FT": {"p": float(stats.wilcoxon(rag, ft).pvalue),
                                   "media_RAG": round(rag.mean(), 2), "media_FT": round(ft.mean(), 2)},
        }

    m = matriz("coincide")
    q = cochrans_q(m)
    out["coincide"] = {"Q": round(float(q.statistic), 2), "p": float(q.pvalue),
                       "prop": dict(zip(cfgs, m.mean(0).round(3)))}
    for j, c in enumerate(cfgs[:3]):
        solo_ft = int(((m[:, 3] == 1) & (m[:, j] == 0)).sum())
        solo_rag = int(((m[:, 3] == 0) & (m[:, j] == 1)).sum())
        out["coincide"][f"discordantes_FTvs_{c}"] = (solo_ft, solo_rag, float(mcnemar([[0, solo_ft], [solo_rag, 0]], exact=True).pvalue))

    conteo = [sum(i["config_preferida"] == c for i in inter) for c in cfgs]
    chi2, p = stats.chisquare(conteo)
    out["preferencia"] = {"conteo": dict(zip(cfgs, conteo)), "chi2_uniforme": round(chi2, 2), "p": float(p)}

    def alfa(cols):
        x = np.array([[a[c] for c in cols] for a in parte_a], dtype=float)
        k = x.shape[1]
        return k / (k - 1) * (1 - x.var(0, ddof=1).sum() / x.sum(1).var(ddof=1))

    out["alfa_cronbach_n5"] = {nombre: round(alfa(cols), 2) for nombre, (_, cols) in zip(("UP", "FU", "IU"), CONSTRUCTOS)}

    t = matriz("tiempo_seg")
    chi2, p = stats.friedmanchisquare(*t.T)
    out["tiempo_medido"] = {"medianas": dict(zip(cfgs, np.median(t, 0).round(2))), "chi2": round(chi2, 2), "p": float(p)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interacciones", type=Path, default=ORIGEN / "interacciones.jsonl")
    ap.add_argument("--parte-a", type=Path, default=ORIGEN / "tam_parte_a.jsonl")
    args = ap.parse_args()

    inter = leer(args.interacciones)
    parte_a = leer(args.parte_a)
    if not inter:
        sys.exit("No hay interacciones registradas todavía en " + str(ORIGEN))
    # Una sola Parte A por participante (la última si se repitió).
    parte_a = list({r["participante"]: r for r in parte_a}.values())

    # El día se deriva de la fecha de cada consulta: día 1 = el primer día con registros del participante.
    dias_de = defaultdict(list)
    for i in inter:
        dias_de[i["participante"]].append(datetime.fromisoformat(i["timestamp"]).date())

    resumen = {"sintetico": False, "n_participantes": len(dias_de), "n_consultas": len(inter),
               "por_configuracion": {}, "registro_diario": [], "parte_A": {}, "n_parte_A": len(parte_a)}

    for cfg in CONFIGS:
        rs = [x for i in inter for x in i["respuestas"] if x["config"] == cfg]
        if not rs:
            continue
        resumen["por_configuracion"][cfg] = {
            "n_respuestas": len(rs),
            "coincidencia_pct": round(100 * sum(x["coincide"] for x in rs) / len(rs), 1),
            "confianza": round(statistics.mean(x["confianza"] for x in rs), 2),
            "advertencia_info_ausente": round(statistics.mean(x["transparencia"] for x in rs), 2),
            "tiempo_percibido": round(statistics.mean(x["tiempo_percibido"] for x in rs), 2),
            "tiempo_medido_seg": round(statistics.median(x["tiempo_seg"] for x in rs), 2),
            "preferencia_pct_consultas": round(100 * sum(i["config_preferida"] == cfg for i in inter) / len(inter), 1),
        }

    for p, fechas in sorted(dias_de.items()):
        mias = [i for i in inter if i["participante"] == p]
        distintos = sorted(set(fechas))
        por_dia = [sum(d == dia for d in fechas) for dia in distintos]
        aciertos = sum(next(x for x in i["respuestas"] if x["config"] == i["config_preferida"])["coincide"]
                       for i in mias)
        resumen["registro_diario"].append({
            "participante": p, "estaciones": sorted({i["estacion"] for i in mias}), "dias_con_uso": len(distintos),
            "por_dia": por_dia, "total": len(mias), "aciertos": aciertos,
            "tasa_pct": round(100 * aciertos / len(mias), 1),
        })

    if parte_a:
        for constructo, codigos in CONSTRUCTOS:
            resumen["parte_A"][constructo] = round(statistics.mean(a[c] for a in parte_a for c in codigos), 2)
    faltan = sorted(set(dias_de) - {a["participante"] for a in parte_a})
    if faltan:
        resumen["aviso"] = "Sin Parte A todavía: " + ", ".join(faltan)

    DESTINO.mkdir(exist_ok=True)
    json.dump(resumen, open(DESTINO / "resumen_real.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(json.dumps(resumen, indent=1, ensure_ascii=False))

    if len(parte_a) >= 3:
        est = estadistica(inter, parte_a)
        json.dump(est, open(DESTINO / "estadistica_validacion.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)

    if all(c in resumen["por_configuracion"] for c in CONFIGS):
        from graficar_validacion import graficar

        print("guardado:", graficar(inter))
    else:
        print("Fig. 6 no generada: faltan configuraciones en los registros.")


if __name__ == "__main__":
    main()
