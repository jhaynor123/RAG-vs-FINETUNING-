"""
Figuras del artículo con los resultados de la Fase 5.

  Fig. 4  fig4_rendimiento.png : un gráfico por configuración, con una línea por tipo de dato
          sobre las cuatro métricas (0 a 1).
  Fig. 5  fig5_contrastes.png  : arriba, medias con IC 95 % por configuración y métrica;
          abajo, distribución de la diferencia pareada RAG - Fine-Tuning.
  Fig. 7  fig7_costos.png      : costo real por componente (USD).

Lee resultados/tabla_rendimiento.csv, matriz_rendimiento.csv, reporte_estadistico.json y los
archivos de costos de la Fase 5. Cada serie tiene su propio color y forma de marcador, para
que no dependa solo del color.

Uso:
    python src/fase5_evaluacion/graficar_resultados.py
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

TINTA, TINTA_2, REJILLA = "#0b0b0b", "#52514e", "#e5e4e0"
COLOR_DOMINIO = {"Lenguaje natural": "#2a78d6", "Semiestructurado": "#eb6834"}
MARCA_DOMINIO = {"Lenguaje natural": "o", "Semiestructurado": "s"}
CONFIGS = ["RAG_GPT", "RAG_Claude", "RAG_Gemini", "FT_Qwen"]
NOMBRE = {"RAG_GPT": "RAG · ChatGPT", "RAG_Claude": "RAG · Claude", "RAG_Gemini": "RAG · Gemini",
          "FT_Qwen": "Fine-Tuning · Qwen2.5-7B"}
CORTO = {"RAG_GPT": "ChatGPT", "RAG_Claude": "Claude", "RAG_Gemini": "Gemini", "FT_Qwen": "Fine-Tuning"}
COLOR_CONFIG = {"RAG_GPT": "#2a78d6", "RAG_Claude": "#eb6834", "RAG_Gemini": "#1baf7a", "FT_Qwen": "#52514e"}
MARCA_CONFIG = {"RAG_GPT": "o", "RAG_Claude": "s", "RAG_Gemini": "^", "FT_Qwen": "D"}
DOMINIOS = ["Lenguaje natural", "Semiestructurado"]
SUBSET = {"Lenguaje natural": "A", "Semiestructurado": "B"}
METRICAS = [("faithfulness_media", "Faithfulness"), ("answer_correctness_media", "Answer Correctness"),
            ("answer_relevance_media", "Answer Relevance")]
DEST = config.RESULTADOS_DIR / "figuras"


def estilo(eje, ylim=(-0.04, 1.05)):
    eje.set_ylim(*ylim)
    eje.grid(axis="y", color=REJILLA, linewidth=0.8)
    eje.set_axisbelow(True)
    for lado in ("top", "right"):
        eje.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        eje.spines[lado].set_color(TINTA_2)
    eje.tick_params(colors=TINTA_2, labelsize=9)


def ic95(valores):
    v = np.asarray(valores, dtype=float)
    return stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))


def fig4():
    df = pd.read_csv(config.RESULTADOS_DIR / "tabla_rendimiento.csv")
    etiquetas = ["Faithfulness", "Answer\nCorrectness", "Answer\nRelevance", "Tasa de\nalucinación"]
    fig, ejes = plt.subplots(2, 2, figsize=(11, 7.6), sharey=True)
    for eje, cfg in zip(ejes.ravel(), CONFIGS):
        for dom in DOMINIOS:
            f = df[(df["configuracion"] == cfg) & (df["dominio"] == dom)].iloc[0]
            y = [f["faithfulness"], f["answer_correctness"], f["answer_relevance"], f["tasa_alucinacion_pct"] / 100]
            eje.plot(range(4), y, color=COLOR_DOMINIO[dom], marker=MARCA_DOMINIO[dom], markersize=8, linewidth=2.2,
                     markeredgecolor="white", markeredgewidth=1.2, label=dom)
        eje.set_title(NOMBRE[cfg], fontsize=12, fontweight="bold", color=TINTA, pad=8)
        eje.set_xticks(range(4))
        eje.set_xticklabels(etiquetas, fontsize=9)
        eje.set_xlim(-0.3, 3.3)
        estilo(eje)
    for eje in ejes[:, 0]:
        eje.set_ylabel("Valor (0 a 1)", fontsize=10, color=TINTA_2)
    h, l = ejes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.0))
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    salida = DEST / "fig4_rendimiento.png"
    plt.savefig(salida, dpi=200, facecolor="white")
    plt.close(fig)
    return salida


def fig5():
    m = pd.read_csv(config.RESULTADOS_DIR / "matriz_rendimiento.csv")
    rep = json.load(open(config.RESULTADOS_DIR / "reporte_estadistico.json", encoding="utf-8"))
    contr = {(c["dominio"], c["metrica"]): c for c in rep["contraste_rag_vs_finetuning"]}
    fig, ejes = plt.subplots(2, 3, figsize=(13, 8), gridspec_kw={"height_ratios": [1, 1.05]})
    for k, (col, nombre) in enumerate(METRICAS):
        arriba = ejes[0, k]
        for d_i, dom in enumerate(DOMINIOS):
            sub = m[m["subset"] == SUBSET[dom]]
            for c_i, cfg in enumerate(CONFIGS):
                v = sub[sub["arquitectura"] == cfg][col].values
                arriba.errorbar(c_i + (d_i - 0.5) * 0.22, v.mean(), yerr=ic95(v), fmt=MARCA_DOMINIO[dom],
                                color=COLOR_DOMINIO[dom], markersize=8, markeredgecolor="white", markeredgewidth=1.2,
                                capsize=3, linewidth=1.6, label=dom if c_i == 0 else None)
        arriba.set_title(nombre, fontsize=11.5, fontweight="bold", color=TINTA)
        arriba.set_xticks(range(4))
        arriba.set_xticklabels([CORTO[c] for c in CONFIGS], fontsize=9)
        arriba.set_xlim(-0.5, 3.5)
        estilo(arriba)
        if k == 0:
            arriba.set_ylabel("Media con IC 95 %", fontsize=10, color=TINTA_2)

        abajo = ejes[1, k]
        datos, colores = [], []
        for dom in DOMINIOS:
            sub = m[m["subset"] == SUBSET[dom]]
            rag = sub[sub["arquitectura"].isin(CONFIGS[:3])].groupby("caso_id")[col].mean()
            ft = sub[sub["arquitectura"] == "FT_Qwen"].set_index("caso_id")[col]
            datos.append((rag - ft.reindex(rag.index)).dropna().values)
            colores.append(COLOR_DOMINIO[dom])
        bp = abajo.boxplot(datos, positions=[0, 1], widths=0.5, patch_artist=True, showmeans=True,
                           medianprops=dict(color=TINTA, linewidth=1.6),
                           meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor=TINTA, markersize=6),
                           flierprops=dict(marker="o", markersize=3, markerfacecolor=TINTA_2, markeredgecolor="none", alpha=0.5),
                           whiskerprops=dict(color=TINTA_2), capprops=dict(color=TINTA_2))
        for caja, color in zip(bp["boxes"], colores):
            caja.set_facecolor(color)
            caja.set_alpha(0.55)
            caja.set_edgecolor(color)
        abajo.axhline(0, color=TINTA, linestyle="--", linewidth=1)
        abajo.set_xticks([0, 1])
        abajo.set_xticklabels(DOMINIOS, fontsize=9)
        estilo(abajo, ylim=(-1.1, 1.35))
        for x, dom in enumerate(DOMINIOS):
            c = contr[(dom, nombre)]
            p = "< 0,001" if c["p_valor"] < 0.001 else f"{c['p_valor']:.3f}".replace(".", ",")
            ef = f"{c['tipo_efecto'].replace(' de Cohen', '')} = {c['tamano_efecto']:.2f}".replace(".", ",")
            prueba = "Wilcoxon" if c["prueba_usada"].startswith("Wilcoxon") else "T-Student"
            abajo.text(x, 1.12, f"{prueba}\np {p}\n{ef}", ha="center", va="bottom", fontsize=8, color=TINTA_2, linespacing=1.25)
        if k == 0:
            abajo.set_ylabel("Diferencia por caso\n(RAG − Fine-Tuning)", fontsize=10, color=TINTA_2)
    h, l = ejes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.0))
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    salida = DEST / "fig5_contrastes.png"
    plt.savefig(salida, dpi=200, facecolor="white")
    plt.close(fig)
    return salida


def fig7():
    gen = json.load(open(config.RESULTADOS_DIR / "costo_generacion_rag.json", encoding="utf-8"))
    qa = json.load(open(config.RESULTADOS_DIR / "costo_dataset_qa.json", encoding="utf-8"))
    juez = json.load(open(config.RESULTADOS_DIR / "costo_juez.json", encoding="utf-8"))
    def mtok(x):
        return f"{(x['tokens_entrada'] + x['tokens_salida']) / 1e6:.2f}".replace(".", ",")

    def etiqueta(nombre, x):
        llamadas = f"{x['llamadas']:,}".replace(",", " ")
        return f"{nombre}\n{llamadas} llamadas · {mtok(x)} M tokens"

    items = [(etiqueta("Juez DeepSeek-V4-Pro (2 400 juicios)", juez), juez["costo_usd_tarifa_pico"]),
             (etiqueta("Dataset de entrenamiento (Claude Sonnet 5.5)", qa), qa["costo_usd"]),
             (etiqueta("Generador RAG · Claude", gen["claude"]), gen["claude"]["costo_usd"]),
             (etiqueta("Generador RAG · Gemini", gen["gemini"]), gen["gemini"]["costo_usd"]),
             (etiqueta("Generador RAG · ChatGPT", gen["gpt"]), gen["gpt"]["costo_usd"]),
             ("Generador Fine-Tuning · Qwen2.5-7B\n200 respuestas en GPU local, sin API", 0.0)]
    total = sum(v for _, v in items)
    fig, eje = plt.subplots(figsize=(9.5, 5.4))
    y = np.arange(len(items))[::-1]
    eje.barh(y, [v for _, v in items], height=0.55, color="#2a78d6")
    for yy, (_, v) in zip(y, items):
        eje.text(v + 0.12, yy, f"${v:.2f}".replace(".", ","), va="center", fontsize=10, color=TINTA)
    eje.set_yticks(y)
    eje.set_yticklabels([n for n, _ in items], fontsize=9, color=TINTA)
    eje.set_xlim(0, max(v for _, v in items) * 1.18)
    eje.set_xlabel("Costo real (USD)", fontsize=10, color=TINTA_2)
    eje.grid(axis="x", color=REJILLA, linewidth=0.8)
    eje.set_axisbelow(True)
    for lado in ("top", "right"):
        eje.spines[lado].set_visible(False)
    eje.tick_params(colors=TINTA_2)
    eje.text(0.98, 0.04, f"Total de la campaña (API): ${total:.2f}".replace(".", ","), transform=eje.transAxes,
             ha="right", fontsize=10.5, fontweight="bold", color=TINTA)
    plt.tight_layout()
    salida = DEST / "fig7_costos.png"
    plt.savefig(salida, dpi=200, facecolor="white")
    plt.close(fig)
    return salida


def main():
    DEST.mkdir(exist_ok=True)
    for f in (fig4, fig5, fig7):
        print("guardado:", f())


if __name__ == "__main__":
    main()
