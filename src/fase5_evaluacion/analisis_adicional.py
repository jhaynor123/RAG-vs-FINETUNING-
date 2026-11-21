"""
Análisis adicionales de la Fase 5 (se corren después de analisis_estadistico.py):

  - Contrastes RAG vs. Fine-Tuning con corrección de Holm, y de cada generador por separado.
  - Interacción estrategia x tipo de dato (U de Mann-Whitney sobre la diferencia por caso).
  - Friedman entre generadores por tipo de dato; Q de Cochran para la tasa de alucinación.
  - Sensibilidad de la tasa de alucinación al umbral, validez de Faithfulness (correlación con
    Answer Correctness) y estabilidad del juez por métrica.
  - Intervalos de confianza del 95 % (bootstrap percentil para las medias, Wilson para la tasa de alucinación).
  - Costo por consulta por generador.
  - Si existen: condiciones de control (matriz_rendimiento_controles.csv) y fidelidad frente a los fragmentos
    recibidos (fidelidad_fragmentos_recibidos.csv).

Salidas (resultados/): reporte_adicional.json, tabla_rendimiento_ic.csv.

Uso:
    python src/fase5_evaluacion/analisis_adicional.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import cochrans_q
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

RES = config.RESULTADOS_DIR
METRICAS = {"faithfulness_media": "Faithfulness", "answer_correctness_media": "Answer Correctness",
            "answer_relevance_media": "Answer Relevance"}
SUBSETS = {"A": "Lenguaje natural", "B": "Semiestructurado"}
RAG = ["RAG_Claude", "RAG_GPT", "RAG_Gemini"]
TODAS = RAG + ["FT_Qwen"]
SEMILLA = 7
REMUESTREOS = 4000


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    centro = (p + z * z / (2 * n)) / (1 + z * z / n)
    mitad = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(centro - mitad), float(centro + mitad)


def bootstrap_ic(valores, rng) -> tuple[float, float]:
    v = np.asarray(valores, dtype=float)
    medias = rng.choice(v, size=(REMUESTREOS, len(v)), replace=True).mean(axis=1)
    return float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))


def wilcoxon(a, b) -> dict:
    """Prueba pareada de rangos con signo; r = z / raíz(N) con z tomado del valor p."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    try:
        res = stats.wilcoxon(a, b)
        z = stats.norm.isf(max(res.pvalue, 1e-300) / 2)
        return {"W": float(res.statistic), "p": float(res.pvalue), "r": float(z / np.sqrt(len(a))),
                "media_a": float(a.mean()), "media_b": float(b.mean())}
    except ValueError:
        return {"W": None, "p": 1.0, "r": 0.0, "media_a": float(a.mean()), "media_b": float(b.mean())}


def ancho(df: pd.DataFrame, columna: str, configs: list[str]) -> pd.DataFrame:
    """Una fila por caso y una columna por configuración (solo casos con todas)."""
    return df.pivot(index="caso_id", columns="arquitectura", values=columna)[configs].dropna()


def holm(ps: list[float]) -> list[float]:
    return [float(x) for x in multipletests(ps, method="holm")[1]]


def contrastes_rag_vs_ft(m: pd.DataFrame) -> dict:
    """Los seis contrastes RAG vs. Fine-Tuning de analisis_estadistico.py (con la prueba que allí se eligió
    según Shapiro-Wilk), con valor p exacto y corrección de Holm."""
    base = json.load(open(RES / "reporte_estadistico.json", encoding="utf-8"))["contraste_rag_vs_finetuning"]
    filas = [{"dominio": c["dominio"], "metrica": c["metrica"], "n": c["n_casos_pareados"], "prueba": c["prueba_usada"],
              "estadistico": c["estadistico"], "gl": c["gl"], "p": c["p_exacto"], "tamano_efecto": c["tamano_efecto"],
              "tipo_efecto": c["tipo_efecto"]} for c in base]
    for f, p in zip(filas, holm([f["p"] for f in filas])):
        f["p_holm"] = p
    por_generador = []
    for s, dom in SUBSETS.items():
        sub = m[m.subset == s]
        for col in ("faithfulness_media", "answer_correctness_media"):
            w = ancho(sub, col, TODAS)
            for g in RAG:
                por_generador.append({"dominio": dom, "metrica": METRICAS[col], "generador": g, **wilcoxon(w[g], w["FT_Qwen"])})
    return {"rag_vs_ft": filas, "cada_generador_vs_ft": por_generador,
            "max_p_por_generador": max(x["p"] for x in por_generador)}


def interaccion(m: pd.DataFrame) -> dict:
    out = {}
    for col, nombre in METRICAS.items():
        dif = {}
        for s in SUBSETS:
            w = ancho(m[m.subset == s], col, TODAS)
            dif[s] = (w[RAG].mean(axis=1) - w["FT_Qwen"]).values
        u = stats.mannwhitneyu(dif["A"], dif["B"], alternative="two-sided")
        n1, n2 = len(dif["A"]), len(dif["B"])
        puntajes = {s: {"rag": ancho(m[m.subset == s], col, TODAS)[RAG].mean(axis=1).values,
                        "ft": ancho(m[m.subset == s], col, TODAS)["FT_Qwen"].values} for s in SUBSETS}
        out[nombre] = {
            "dif_media_manuales": float(dif["A"].mean()), "dif_media_registros": float(dif["B"].mean()),
            "U": float(u.statistic), "p": float(u.pvalue), "r_rb": float(1 - 2 * u.statistic / (n1 * n2)),
            "rag_manuales_vs_registros_p": float(stats.mannwhitneyu(puntajes["A"]["rag"], puntajes["B"]["rag"]).pvalue),
            "ft_manuales_vs_registros_p": float(stats.mannwhitneyu(puntajes["A"]["ft"], puntajes["B"]["ft"]).pvalue),
        }
    return out


def friedman_y_cochran(m: pd.DataFrame) -> dict:
    out = {"friedman_generadores": {}, "cochran_alucinacion": {}}
    for ambito, sub in (("conjunto", m), *[(SUBSETS[s], m[m.subset == s]) for s in SUBSETS]):
        out["friedman_generadores"][ambito] = {}
        for col, nombre in METRICAS.items():
            w = ancho(sub, col, RAG)
            chi2, p = stats.friedmanchisquare(*[w[c].values for c in RAG])
            out["friedman_generadores"][ambito][nombre] = {"n": len(w), "chi2": float(chi2), "p": float(p)}
        w = ancho(sub.assign(alu=sub.alucinacion.astype(int)), "alu", TODAS)
        q = cochrans_q(w.values)
        out["cochran_alucinacion"][ambito] = {"n": len(w), "Q": float(q.statistic), "p": float(q.pvalue)}
    return out


def validez_y_estabilidad(m: pd.DataFrame) -> dict:
    out = {"spearman_F_vs_AC": {}, "sensibilidad_umbral": {}, "estabilidad": {}, "alucinacion_con_AC_alta": {}}
    for s, dom in SUBSETS.items():
        sub = m[m.subset == s]
        r = stats.spearmanr(sub.faithfulness_media, sub.answer_correctness_media)
        out["spearman_F_vs_AC"][dom] = {"rho": float(r.statistic), "p": float(r.pvalue), "n": len(sub)}
        out["sensibilidad_umbral"][dom] = {
            str(t): {a: round(float((g.faithfulness_media < t).mean()), 3) for a, g in sub.groupby("arquitectura")}
            for t in (0.3, 0.5, 0.7)}
        out["alucinacion_con_AC_alta"][dom] = {
            a: int(((g.faithfulness_media < 0.5) & (g.answer_correctness_media >= 0.5)).sum()) for a, g in sub.groupby("arquitectura")}
    for col in ("faithfulness_std", "answer_correctness_std", "answer_relevance_std"):
        out["estabilidad"][col] = {"media": float(m[col].mean()), "pct_sobre_0_05": float(100 * (m[col] > 0.05).mean())}
    out["estabilidad"]["pct_respuestas_con_alguna_metrica_sobre_0_05"] = float(
        100 * (m[["faithfulness_std", "answer_correctness_std", "answer_relevance_std"]] > 0.05).any(axis=1).mean())
    return out


def tabla_con_ic(m: pd.DataFrame, configs: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(SEMILLA)
    filas = []
    for a in configs:
        for s, dom in SUBSETS.items():
            sub = m[(m.arquitectura == a) & (m.subset == s)]
            if sub.empty:
                continue
            fila = {"configuracion": a, "dominio": dom, "n": len(sub)}
            for col, nombre in METRICAS.items():
                lo, hi = bootstrap_ic(sub[col], rng)
                fila[nombre], fila[nombre + "_ic_inf"], fila[nombre + "_ic_sup"] = float(sub[col].mean()), lo, hi
            k = int(sub.alucinacion.sum())
            lo, hi = wilson(k, len(sub))
            fila["tasa_alucinacion"], fila["tasa_alucinacion_ic_inf"], fila["tasa_alucinacion_ic_sup"] = k / len(sub), lo, hi
            filas.append(fila)
    return pd.DataFrame(filas)


def costos_por_consulta() -> dict:
    gen = json.load(open(RES / "costo_generacion_rag.json", encoding="utf-8"))
    return {p: round(gen[p]["costo_usd"] / gen[p]["llamadas"], 4) for p in ("claude", "gpt", "gemini")}


def controles(m: pd.DataFrame) -> dict | None:
    ruta = RES / "matriz_rendimiento_controles.csv"
    if not ruta.exists():
        return None
    c = pd.read_csv(ruta)
    todo = pd.concat([m, c], ignore_index=True)
    nuevas = ["Qwen_RAG_congelado", "Qwen_base_sin_RAG"]
    comparaciones = [("Qwen_RAG_congelado", "FT_Qwen", "estrategia con el mismo modelo: RAG frente a ajuste fino"),
                     ("Qwen_RAG_congelado", "Qwen_base_sin_RAG", "efecto de la recuperación sobre el modelo sin ajustar"),
                     ("FT_Qwen", "Qwen_base_sin_RAG", "efecto del ajuste fino sobre el modelo sin ajustar")]
    filas = []
    for s, dom in SUBSETS.items():
        sub = todo[todo.subset == s]
        for col, nombre in METRICAS.items():
            w = ancho(sub, col, TODAS + nuevas)
            for x, y, sentido in comparaciones:
                filas.append({"dominio": dom, "metrica": nombre, "comparacion": f"{x} vs {y}", "sentido": sentido,
                              "n": len(w), **wilcoxon(w[x], w[y])})
            filas.append({"dominio": dom, "metrica": nombre, "comparacion": "Qwen_RAG_congelado vs promedio RAG comercial",
                          "sentido": "capacidad del modelo: Qwen frente a los tres generadores comerciales", "n": len(w),
                          **wilcoxon(w["Qwen_RAG_congelado"], w[RAG].mean(axis=1))})
    for f, p in zip(filas, holm([f["p"] for f in filas])):
        f["p_holm"] = p
    alu = {}
    for s, dom in SUBSETS.items():
        sub = todo[todo.subset == s]
        alu[dom] = {a: round(float(g.alucinacion.mean()), 3) for a, g in sub.groupby("arquitectura")}
    w = ancho(todo.assign(alu=todo.alucinacion.astype(int)), "alu", TODAS + nuevas)
    q = cochrans_q(w.values)
    tabla = tabla_con_ic(todo, TODAS + nuevas)
    tabla.to_csv(RES / "tabla_rendimiento_controles_ic.csv", index=False, encoding="utf-8")
    return {"contrastes": filas, "tasa_alucinacion": alu, "cochran_6_configuraciones": {"n": len(w), "Q": float(q.statistic), "p": float(q.pvalue)}}


def fidelidad_recibida(m: pd.DataFrame) -> dict | None:
    ruta = RES / "fidelidad_fragmentos_recibidos.csv"
    if not ruta.exists():
        return None
    f = pd.read_csv(ruta)
    f["alucinacion_recibidos"] = f.faithfulness_recibidos_media < config.UMBRAL_ALUCINACION_FAITHFULNESS
    out = {"tasa_alucinacion_recibidos": {}, "faithfulness_recibidos_media": {}, "abstencion": {}, "tasa_alucinacion_contra_c": {}}
    ac = pd.concat([m, pd.read_csv(RES / "matriz_rendimiento_controles.csv")]) if (RES / "matriz_rendimiento_controles.csv").exists() else m
    f = f.merge(ac[["caso_id", "arquitectura", "faithfulness_media", "answer_correctness_media"]], on=["caso_id", "arquitectura"], how="left")
    for s, dom in SUBSETS.items():
        sub = f[f.subset == s]
        out["tasa_alucinacion_recibidos"][dom] = {a: round(float(g.alucinacion_recibidos.mean()), 3) for a, g in sub.groupby("arquitectura")}
        out["tasa_alucinacion_contra_c"][dom] = {a: round(float((g.faithfulness_media < 0.5).mean()), 3) for a, g in sub.groupby("arquitectura")}
        out["faithfulness_recibidos_media"][dom] = {a: round(float(g.faithfulness_recibidos_media.mean()), 3) for a, g in sub.groupby("arquitectura")}
        out["abstencion"][dom] = {a: {k: int((g.abstencion == k).sum()) for k in ("total", "parcial", "ninguna")} for a, g in sub.groupby("arquitectura")}
        r = stats.spearmanr(sub.faithfulness_recibidos_media, sub.answer_correctness_media, nan_policy="omit")
        out.setdefault("spearman_F_recibidos_vs_AC", {})[dom] = {"rho": float(r.statistic), "p": float(r.pvalue), "n": int(len(sub))}
    # respuestas correctas según el juez (AC >= 0,5) que cuentan como alucinación solo contra el fragmento c
    out["AC_alta_y_alucinacion_contra_c_pero_no_contra_recibidos"] = {
        a: int(((g.answer_correctness_media >= 0.5) & (g.faithfulness_media < 0.5) & (g.faithfulness_recibidos_media >= 0.5)).sum())
        for a, g in f[f.subset == "A"].groupby("arquitectura")}
    return out


def cruce_usuarios_juez(m: pd.DataFrame) -> dict | None:
    """Compara la coincidencia que marcaron los participantes con el puntaje del juez para el mismo caso y
    configuración. Los participantes valoraron respuestas generadas en vivo, no las que calificó el juez, así que
    el cruce es a nivel de caso y configuración."""
    ruta = config.RESULTADOS_DIR / "validacion_usuarios" / "interacciones_sin_texto.jsonl"
    if not ruta.exists():
        return None
    filas = []
    for l in open(ruta, encoding="utf-8"):
        i = json.loads(l)
        for r in i["respuestas"]:
            filas.append({"caso_id": i["caso_id"], "arquitectura": r["config"], "coincide": int(r["coincide"])})
    u = pd.DataFrame(filas).merge(m[["caso_id", "arquitectura", "answer_correctness_media", "faithfulness_media"]],
                                  on=["caso_id", "arquitectura"])
    out = {"n_pares_caso_configuracion": len(u), "por_configuracion": {}}
    for a, g in u.groupby("arquitectura"):
        si, no = g[g.coincide == 1], g[g.coincide == 0]
        out["por_configuracion"][a] = {
            "n_coincide_si": len(si), "n_coincide_no": len(no),
            "AC_juez_media_si": float(si.answer_correctness_media.mean()),
            "AC_juez_media_no": float(no.answer_correctness_media.mean()) if len(no) else None}
    rag = u[u.arquitectura != "FT_Qwen"]
    out["rag_respuestas_no_coincide"] = int((rag.coincide == 0).sum())
    out["rag_no_coincide_con_AC_juez_igual_a_cero"] = int(((rag.coincide == 0) & (rag.answer_correctness_media == 0)).sum())
    out["rag_AC_juez_media"] = float(rag.answer_correctness_media.mean())
    out["rag_coincidencia_usuarios"] = float(rag.coincide.mean())
    return out


def _cargar_corpus() -> tuple[list[dict], list[dict]]:
    leer = lambda n: [json.loads(l) for l in open(config.DATA_PROCESSED / n, encoding="utf-8")]
    return leer("corpus_A.jsonl"), leer("corpus_B.jsonl")


def similitud_entrenamiento_prueba() -> dict | None:
    """Coseno máximo entre cada pregunta de prueba y todas las preguntas de entrenamiento (embeddings de la Ruta A).
    Requiere data/processed (documentación de la planta), que no se distribuye."""
    try:
        casos = [json.loads(l) for l in open(config.DATA_PROCESSED / "resultados_respuestas.jsonl", encoding="utf-8")]
        entrenamiento = [json.loads(l)["messages"][1]["content"] for n in ("ft_train.jsonl", "ft_val.jsonl")
                         for l in open(config.DATA_PROCESSED / n, encoding="utf-8")]
    except FileNotFoundError:
        return None
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from embeddings_utils import embed_textos

    sim = np.array(embed_textos([c["pregunta"] for c in casos])) @ np.array(embed_textos(entrenamiento)).T
    mx, sub = sim.max(axis=1), np.array([c["subset"] for c in casos])
    return {d: {"n": int((sub == s).sum()), "media_maxima": float(mx[sub == s].mean()), "maxima": float(mx[sub == s].max()),
                **{f"n_sobre_{u}": int((mx[sub == s] >= u).sum()) for u in (0.8, 0.9, 0.95)}}
            for s, d in (("A", "Lenguaje natural"), ("B", "Semiestructurado"))}


def solapamiento_respuestas() -> dict | None:
    """Cuántas respuestas de referencia de los casos de prueba aparecen en el dataset de entrenamiento (con otra
    redacción de la pregunta). El filtro de fuga del dataset solo elimina preguntas idénticas, no respuestas.
    Requiere data/processed (documentación de la planta), que no se distribuye."""
    import re
    try:
        casos = [json.loads(l) for l in open(config.DATA_PROCESSED / "resultados_respuestas.jsonl", encoding="utf-8")]
        entrenamiento = [json.loads(l)["messages"][2]["content"] for n in ("ft_train.jsonl", "ft_val.jsonl")
                         for l in open(config.DATA_PROCESSED / n, encoding="utf-8")]
    except FileNotFoundError:
        return None
    norm = lambda t: re.sub(r"\W+", " ", t.lower()).strip()
    entrenadas = [norm(t) for t in entrenamiento]
    conjunto = set(entrenadas)

    def f1(a: str, b: str) -> float:
        x, y = set(a.split()), set(b.split())
        i = len(x & y)
        return 0.0 if not i else 2 * i / (len(x) + len(y))

    out = {}
    for s, dom in SUBSETS.items():
        refs = [norm(c["respuesta_referencia"]) for c in casos if c["subset"] == s]
        mejor = [max(f1(r, e) for e in entrenadas) for r in refs]
        out[dom] = {"n_casos": len(refs), "casos_con_respuesta_identica_en_entrenamiento": sum(r in conjunto for r in refs),
                    "f1_palabras_maximo_medio": float(np.mean(mejor)), "casos_con_f1_maximo_ge_0_9": int(sum(x >= 0.9 for x in mejor))}
    return out


def cobertura_lexica_manuales() -> dict | None:
    """Diagnóstico exploratorio (no es el acierto de recuperación): para los 100 casos de manuales, el mejor fragmento
    posible por cobertura léxica de la respuesta de referencia (ponderada por IDF) frente a los tres recuperados, según si
    la respuesta declaró abstención total. Cuenta además los casos con fragmentos de registros entre los tres recuperados."""
    ruta = RES / "fidelidad_fragmentos_recibidos.csv"
    try:
        casos = [json.loads(l) for l in open(config.DATA_PROCESSED / "resultados_respuestas.jsonl", encoding="utf-8")]
        corpus_a, corpus_b = _cargar_corpus()
    except FileNotFoundError:
        return None
    if not ruta.exists():
        return None
    import math
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from busqueda_lexica import tokenizar

    fid = pd.read_csv(ruta)
    abst = fid[fid.arquitectura == "RAG_Claude"].set_index("caso_id").abstencion
    todos = corpus_a + corpus_b
    ids = [c["chunk_id"] for c in todos]
    toks = [set(tokenizar(c["text"])) for c in todos]
    n_a, n = len(corpus_a), len(todos)
    df: dict = {}
    for t in toks:
        for w in t:
            df[w] = df.get(w, 0) + 1
    idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}
    filas, con_registros = [], 0
    for c in casos:
        if c["subset"] != "A":
            continue
        ref = {w for w in tokenizar(c["respuesta_referencia"]) if w in idf and idf[w] > 1.0}
        top3 = [ids.index(i) for i in c["fragmentos_recuperados_claude"]]
        con_registros += any(j >= n_a for j in top3)
        den = sum(idf[w] for w in ref)
        cob = [sum(idf[w] for w in ref if w in t) / den for t in toks[:n_a]]
        filas.append({"abst": abst.get(c["caso_id"]), "cob_mejor": max(cob), "mejor_en_top3": int(np.argmax(cob)) in top3,
                      "cob_top3": sum(idf[w] for w in ref if any(w in toks[j] for j in top3)) / den})
    d = pd.DataFrame(filas)
    por = {k: {"n": int(len(g)), "cobertura_mejor_mediana": float(g.cob_mejor.median()), "cobertura_top3_mediana": float(g.cob_top3.median()),
               "pct_mejor_fragmento_en_top3": float(100 * g.mejor_en_top3.mean())} for k, g in d.groupby("abst")}
    return {"casos_con_fragmento_de_registros_en_top3": int(con_registros), "por_abstencion_de_claude": por}


def energia_gpu() -> dict:
    """Tope de energía de la GPU local: potencia de placa (304 W) por el tiempo medido de cada etapa."""
    potencia_kw = 0.304
    etapas = {"entrenamiento": 878.07, "inferencia_ajustado_200_respuestas": 734.0}
    ruta = RES / "tiempo_controles.json"
    if ruta.exists():
        etapas["controles_400_respuestas"] = json.load(open(ruta, encoding="utf-8"))["segundos"]
    return {"potencia_placa_kw": potencia_kw, "segundos": etapas,
            "kwh_tope": {k: round(potencia_kw * v / 3600, 3) for k, v in etapas.items()},
            "kwh_tope_total": round(potencia_kw * sum(etapas.values()) / 3600, 3)}


def main():
    m = pd.read_csv(RES / "matriz_rendimiento.csv")
    reporte = {
        "contrastes": contrastes_rag_vs_ft(m),
        "interaccion_estrategia_x_tipo_de_dato": interaccion(m),
        "friedman_y_cochran": friedman_y_cochran(m),
        "validez_y_estabilidad": validez_y_estabilidad(m),
        "costo_por_consulta_usd": costos_por_consulta(),
        "controles": controles(m),
        "fidelidad_fragmentos_recibidos": fidelidad_recibida(m),
        "cruce_usuarios_juez": cruce_usuarios_juez(m),
        "similitud_entrenamiento_prueba": similitud_entrenamiento_prueba(),
        "solapamiento_respuestas": solapamiento_respuestas(),
        "cobertura_lexica_manuales": cobertura_lexica_manuales(),
        "energia_gpu": energia_gpu(),
    }
    tabla_con_ic(m, TODAS).to_csv(RES / "tabla_rendimiento_ic.csv", index=False, encoding="utf-8")
    with open(RES / "reporte_adicional.json", "w", encoding="utf-8") as f:
        json.dump(reporte, f, indent=1, ensure_ascii=False, default=float)
    print("Guardado: resultados/reporte_adicional.json y tabla_rendimiento_ic.csv")


if __name__ == "__main__":
    main()
