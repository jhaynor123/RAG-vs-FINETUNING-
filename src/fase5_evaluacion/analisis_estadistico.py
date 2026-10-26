"""
Fase 5 — Análisis estadístico.

El diseño es asimétrico: la Ruta A (RAG) usa tres generadores comerciales por
API y la Ruta B (Fine-Tuning) un solo modelo de pesos abiertos ajustado
localmente. Para cada tipo de dato (lenguaje natural / semiestructurado) y cada
métrica se compara RAG contra Fine-Tuning; el lado RAG promedia sus tres
generadores por caso.

Lee resultados/matriz_rendimiento.csv (de juez_llm.py, con la columna
"arquitectura" en {RAG_Claude, RAG_GPT, RAG_Gemini, FT_Qwen}) y calcula:

  1. Contraste RAG vs. Fine-Tuning: normalidad de las diferencias pareadas
     (Shapiro-Wilk), después T-Student pareada o Wilcoxon, con su tamaño del
     efecto (d de Cohen o r).
  2. Friedman entre los tres generadores de la Ruta A, con Wilcoxon por pares
     corregido por Bonferroni, y entre las cuatro configuraciones en la tasa de
     alucinación.
  3. Tabla de rendimiento, tasa de alucinación y dispersión del juez.

Uso:
    python src/fase5_evaluacion/analisis_estadistico.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

METRICAS = ["faithfulness_media", "answer_correctness_media", "answer_relevance_media"]
NOMBRES_LEGIBLES = {
    "faithfulness_media": "Faithfulness",
    "answer_correctness_media": "Answer Correctness",
    "answer_relevance_media": "Answer Relevance",
}
SUBSET_A_DOMINIO = {"A": "Lenguaje natural", "B": "Semiestructurado"}

# Configuraciones que se promedian para formar el puntaje de cada ruta.
GENERADORES_POR_RUTA = {
    "RAG": ["RAG_Claude", "RAG_GPT", "RAG_Gemini"],
    "FineTuning": ["FT_Qwen"],
}


def cargar_matriz() -> pd.DataFrame:
    path = config.RESULTADOS_DIR / "matriz_rendimiento.csv"
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Corre primero juez_llm.py")
    return pd.read_csv(path)


def promediar_por_ruta(df: pd.DataFrame, metrica: str) -> pd.DataFrame:
    """Promedia por caso los generadores de cada ruta (con uno solo, es ese valor)."""
    filas = []
    for (caso_id, subset), grupo in df.groupby(["caso_id", "subset"]):
        fila = {"caso_id": caso_id, "subset": subset}
        for ruta, arquitecturas in GENERADORES_POR_RUTA.items():
            valores = grupo.loc[grupo["arquitectura"].isin(arquitecturas), metrica]
            fila[ruta] = valores.mean() if len(valores) else np.nan
        filas.append(fila)
    return pd.DataFrame(filas)


def comparar_rutas(df: pd.DataFrame) -> list[dict]:
    filas_reporte = []
    for subset, dominio in SUBSET_A_DOMINIO.items():
        sub_df = df[df["subset"] == subset]

        for metrica in METRICAS:
            promedios = promediar_por_ruta(sub_df, metrica).dropna(subset=["RAG", "FineTuning"])
            if len(promedios) < 3:
                print(f"  [aviso] muy pocos casos con ambas rutas en {dominio}/{NOMBRES_LEGIBLES[metrica]} "
                      f"({len(promedios)}), se omite el contraste")
                continue

            a = promedios["RAG"].values
            b = promedios["FineTuning"].values
            diffs = a - b

            _, p_normalidad = stats.shapiro(diffs) if len(diffs) >= 3 else (None, 0.0)
            es_normal = p_normalidad > config.ALPHA_SIGNIFICANCIA

            if es_normal:
                stat, p_valor = stats.ttest_rel(a, b)
                sd_diff = diffs.std(ddof=1) or 1e-9
                effect_size = diffs.mean() / sd_diff  # d de Cohen (muestras pareadas)
                nombre_prueba, nombre_efecto = "T-Student pareada", "d de Cohen"
            else:
                try:
                    stat, p_valor = stats.wilcoxon(a, b)
                except ValueError:
                    stat, p_valor = np.nan, 1.0
                n = len(diffs)
                effect_size = (
                    abs(stat - n * (n + 1) / 4) / np.sqrt(n * (n + 1) * (2 * n + 1) / 24) / np.sqrt(n)
                    if n > 0 else np.nan
                )
                nombre_prueba, nombre_efecto = "Wilcoxon (rangos con signo)", "r"

            filas_reporte.append({
                "dominio": dominio,
                "metrica": NOMBRES_LEGIBLES[metrica],
                "n_casos_pareados": len(promedios),
                "media_RAG": round(float(a.mean()), 4),
                "media_FineTuning": round(float(b.mean()), 4),
                "ganador": "RAG" if a.mean() > b.mean() else ("FineTuning" if b.mean() > a.mean() else "empate"),
                "p_normalidad_shapiro": round(float(p_normalidad), 4),
                "prueba_usada": nombre_prueba,
                "estadistico": round(float(stat), 4) if stat == stat else None,
                "gl": len(a) - 1 if es_normal else None,
                "p_valor": round(float(p_valor), 6),
                "significativo_0.05": bool(p_valor < config.ALPHA_SIGNIFICANCIA),
                "tamano_efecto": round(float(effect_size), 4) if effect_size == effect_size else None,
                "tipo_efecto": nombre_efecto,
            })
    return filas_reporte


def tabla_rendimiento(df: pd.DataFrame) -> list[dict]:
    """Media de cada métrica y tasa de alucinación por configuración y tipo de dato (Fig. 4)."""
    filas = []
    for arq in ("RAG_GPT", "RAG_Claude", "RAG_Gemini", "FT_Qwen"):
        for subset, dominio in SUBSET_A_DOMINIO.items():
            sub = df[(df["arquitectura"] == arq) & (df["subset"] == subset)]
            if sub.empty:
                continue
            filas.append({
                "configuracion": arq, "dominio": dominio, "n": len(sub),
                "faithfulness": round(float(sub["faithfulness_media"].mean()), 4),
                "answer_correctness": round(float(sub["answer_correctness_media"].mean()), 4),
                "answer_relevance": round(float(sub["answer_relevance_media"].mean()), 4),
                "tasa_alucinacion_pct": round(float(100 * sub["alucinacion"].mean()), 2),
            })
    return filas


def _wide(df: pd.DataFrame, columna: str, arquitecturas: list[str]) -> pd.DataFrame:
    """Un renglón por caso y una columna por configuración (solo casos con todas)."""
    return df.pivot(index="caso_id", columns="arquitectura", values=columna)[arquitecturas].dropna()


def friedman_ruta_a(df: pd.DataFrame) -> list[dict]:
    """Friedman entre los 3 generadores de la Ruta A, por métrica (200 casos),
    con comparaciones por pares de Wilcoxon corregidas por Bonferroni."""
    rag = GENERADORES_POR_RUTA["RAG"]
    filas = []
    for metrica in METRICAS:
        w = _wide(df, metrica, rag)
        chi2, p = stats.friedmanchisquare(*[w[c].values for c in rag])
        k, n = len(rag), len(w)
        pares = []
        for i in range(k):
            for j in range(i + 1, k):
                try:
                    _, pp = stats.wilcoxon(w[rag[i]].values, w[rag[j]].values)
                except ValueError:
                    pp = 1.0
                pares.append({"par": f"{rag[i]} vs {rag[j]}", "p_bonferroni": round(min(1.0, float(pp) * 3), 6)})
        filas.append({
            "metrica": NOMBRES_LEGIBLES[metrica], "n": n,
            "medias": {c: round(float(w[c].mean()), 4) for c in rag},
            "chi2": round(float(chi2), 4), "gl": k - 1, "p_valor": round(float(p), 6),
            "W_kendall": round(float(chi2 / (n * (k - 1))), 4),
            "significativo_0.05": bool(p < config.ALPHA_SIGNIFICANCIA),
            "comparaciones_por_pares": pares,
        })
    return filas


def friedman_alucinacion(df: pd.DataFrame) -> dict:
    """Friedman sobre el indicador binario de alucinación (Faithfulness < umbral)
    entre las 4 configuraciones, sobre los 200 casos."""
    cols = GENERADORES_POR_RUTA["RAG"] + GENERADORES_POR_RUTA["FineTuning"]
    w = _wide(df.assign(alucinacion=df["alucinacion"].astype(float)), "alucinacion", cols)
    chi2, p = stats.friedmanchisquare(*[w[c].values for c in cols])
    k, n = len(cols), len(w)
    return {
        "n": n, "tasas_pct": {c: round(float(100 * w[c].mean()), 2) for c in cols},
        "chi2": round(float(chi2), 4), "gl": k - 1, "p_valor": round(float(p), 6),
        "W_kendall": round(float(chi2 / (n * (k - 1))), 4),
        "significativo_0.05": bool(p < config.ALPHA_SIGNIFICANCIA),
    }


def desglose_por_generador(df: pd.DataFrame) -> list[dict]:
    """Media descriptiva (sin pruebas) de cada generador por ruta, dominio y métrica."""
    filas = []
    for subset, dominio in SUBSET_A_DOMINIO.items():
        for ruta, arquitecturas in GENERADORES_POR_RUTA.items():
            for arq in arquitecturas:
                sub = df[(df["subset"] == subset) & (df["arquitectura"] == arq)]
                if sub.empty:
                    continue
                fila = {"dominio": dominio, "ruta": ruta, "generador": arq.split("_", 1)[1], "n_casos": len(sub)}
                for metrica in METRICAS:
                    fila[NOMBRES_LEGIBLES[metrica]] = round(float(sub[metrica].mean()), 4)
                filas.append(fila)
    return filas


def tasa_alucinacion(df: pd.DataFrame) -> list[dict]:
    filas = []
    for subset, dominio in SUBSET_A_DOMINIO.items():
        for ruta, arquitecturas in GENERADORES_POR_RUTA.items():
            sub = df[(df["subset"] == subset) & (df["arquitectura"].isin(arquitecturas))]
            if sub.empty:
                continue
            tasa = 100 * sub["alucinacion"].mean()
            filas.append({
                "dominio": dominio,
                "ruta": ruta,
                "n_respuestas": len(sub),
                "tasa_alucinacion_pct": round(float(tasa), 2),
            })
    return filas


def dispersion_juez(df: pd.DataFrame) -> dict:
    stds = df[["faithfulness_std", "answer_correctness_std", "answer_relevance_std"]].values.flatten()
    media_std = float(np.mean(stds)) if len(stds) else None
    return {
        "desviacion_estandar_media_entre_k_repeticiones": round(media_std, 4) if media_std is not None else None,
        "dentro_del_umbral_0.05": bool(media_std is not None and media_std <= config.MAX_STD_ACEPTABLE),
    }


def main():
    df = cargar_matriz()
    print(f"Matriz de rendimiento cargada: {len(df)} filas\n")

    print("=== RAG vs. Fine-Tuning por dominio y métrica (RAG = promedio de sus 3 generadores) ===")
    reporte = comparar_rutas(df)
    for r in reporte:
        print(
            f"  [{r['dominio']}] {r['metrica']}: RAG={r['media_RAG']} vs FT={r['media_FineTuning']} "
            f"-> gana {r['ganador']} "
            f"| {r['prueba_usada']}, p={r['p_valor']} "
            f"({'SIGNIFICATIVO' if r['significativo_0.05'] else 'no significativo'}), "
            f"{r['tipo_efecto']}={r['tamano_efecto']}"
        )

    print("\n=== Desglose descriptivo por generador (sin pruebas de hipótesis) ===")
    desglose = desglose_por_generador(df)
    for r in desglose:
        metricas_str = ", ".join(f"{k}={v}" for k, v in r.items() if k in NOMBRES_LEGIBLES.values())
        print(f"  [{r['dominio']}] {r['ruta']} / {r['generador']}: {metricas_str}")

    print("\n=== Tasa de alucinación por ruta (Faithfulness < %.2f) ===" % config.UMBRAL_ALUCINACION_FAITHFULNESS)
    alucinacion = tasa_alucinacion(df)
    for r in alucinacion:
        print(f"  [{r['dominio']}] {r['ruta']}: {r['tasa_alucinacion_pct']}% (n={r['n_respuestas']})")

    print("\n=== Dispersión del juez entre repeticiones ===")
    disp = dispersion_juez(df)
    print(f"  {disp}")

    print("\n=== Rendimiento por configuración y tipo de dato ===")
    matriz = tabla_rendimiento(df)
    for r in matriz:
        print(f"  {r['configuracion']:11} {r['dominio']:17} F={r['faithfulness']} AC={r['answer_correctness']} "
              f"AR={r['answer_relevance']} alucinación={r['tasa_alucinacion_pct']}%")
    pd.DataFrame(matriz).to_csv(config.RESULTADOS_DIR / "tabla_rendimiento.csv", index=False, encoding="utf-8")

    print("\n=== Friedman entre los 3 generadores de la Ruta A (por métrica) ===")
    fried = friedman_ruta_a(df)
    for r in fried:
        print(f"  {r['metrica']}: chi2({r['gl']})={r['chi2']} p={r['p_valor']} W={r['W_kendall']} medias={r['medias']}")
    print("\n=== Friedman de la tasa de alucinación entre las 4 configuraciones ===")
    fried_alu = friedman_alucinacion(df)
    print(f"  {fried_alu}")

    salida = {
        "tabla_rendimiento": matriz,
        "contraste_rag_vs_finetuning": reporte,
        "friedman_ruta_A": fried,
        "friedman_alucinacion": fried_alu,
        "desglose_por_generador": desglose,
        "tasa_alucinacion": alucinacion,
        "dispersion_juez": disp,
    }
    destino = config.RESULTADOS_DIR / "reporte_estadistico.json"
    with open(destino, "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)
    print(f"\nReporte completo guardado en: {destino}")


if __name__ == "__main__":
    main()
