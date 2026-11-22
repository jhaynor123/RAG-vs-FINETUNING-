"""
Diagramas del artículo (Fig. 1: diseño experimental en cinco fases; Fig. 2: arquitectura en cinco capas) y
el diagrama del juez (Fig. S1), en español e inglés y sin título incrustado (el título va en la leyenda).

Salidas (resultados/figuras/): fig1_diseno_{es,en}.png, fig2_arquitectura_{es,en}.png, figS1_juez_{es,en}.png

Uso:
    python src/fase5_evaluacion/graficar_diagramas.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

DEST = config.RESULTADOS_DIR / "figuras"
NAVY, GRIS, CLARO, TEAL, CORAL, TINTA = "#1f2a44", "#8a8f98", "#f3f4f7", "#1b8a94", "#d2573f", "#2b2b2b"

TEXTOS = {
    "es": {
        "fases": [
            ("FASE 1", "Adquisición y\npreprocesamiento", "Ingesta y limpieza\n(Unstructured.io)", "Filtro de idioma y\nfragmentos de 512 tokens",
             "Corpus bifurcado\n(subconjuntos A y B)", "Garantiza la representatividad de ambos tipos de dato"),
            ("FASE 2", "Orquestación y\nrepresentación", "Orquestador\nLlamaIndex", "Embeddings densos y\ndataset sintético QA",
             "Dos representaciones\ndel mismo corpus", "Asegura la simetría: ninguna ruta ve datos distintos"),
            ("FASE 3", "Construcción de\nlas rutas", "Ruta A: ChromaDB + BM25\n+ 3 LLM comerciales", "Ruta B: Qwen2.5-7B\n+ QLoRA de 4 bits",
             "Rutas construidas y\ncontroles con Qwen", "Aísla las estrategias: contexto externo frente a pesos ajustados"),
            ("FASE 4", "Validación técnica", "Prueba técnica\nde ambas rutas", "Revisión por personal\ntécnico de la planta",
             "Sistema listo para\nla evaluación", "Confirma que las rutas funcionan antes de evaluar"),
            ("FASE 5", "Evaluación", "Juez ciego DeepSeek\n(3 métricas, k = 3)", "Validación con\nusuarios (TAM)",
             "Matriz de rendimiento\ny aceptación", "Mide el desempeño y la utilidad percibida"),
        ],
        "salida": "SALIDA",
        "capas": ["Capa 1", "Capa 2", "Capa 3", "Capa 5"],
        "capa_nombres": ["Repositorios\ndocumentales", "Preprocesamiento", "Orquestación", "Evaluación externa"],
        "capa_detalle": ["manuales y registros", "Unstructured.io · 512 tokens", "LlamaIndex", "Juez DeepSeek"],
        "ruta_a": "Capa 4 · Ruta A", "ruta_a_sub": "Índice híbrido\nChromaDB + BM25", "ruta_a_resp": "Recuperador + 3 LLM\nChatGPT · Claude · Gemini\n(sin entrenamiento)",
        "ruta_b": "Capa 4 · Ruta B", "ruta_b_sub": "Dataset JSONL\n(pares QA sintéticos)", "ruta_b_resp": "Adaptador LoRA + Qwen\n(sin recuperación)",
        "como": "Cómo responde",
        "juez": ["Respuesta sin etiqueta\ny en orden aleatorio", "Juez DeepSeek\n(temperatura 0, k = 3)", "Faithfulness ·\nAnswer Correctness ·\nAnswer Relevance",
                 "Matriz de rendimiento\npor caso y configuración"],
        "controles": ["Estabilidad (k = 3)", "Sesgo de posición", "Auto-preferencia", "Concordancia con expertos"],
        "juez_titulo": "Controles de confiabilidad",
    },
    "en": {
        "fases": [
            ("PHASE 1", "Data acquisition and\npreprocessing", "Ingestion and cleaning\n(Unstructured.io)", "Language filter and\n512-token chunks",
             "Bifurcated corpus\n(subsets A and B)", "Ensures both data types are represented"),
            ("PHASE 2", "Orchestration and\nrepresentation", "LlamaIndex\norchestrator", "Dense embeddings and\nsynthetic QA dataset",
             "Two representations\nof the same corpus", "Ensures symmetry: neither route sees different data"),
            ("PHASE 3", "Route\nconstruction", "Route A: ChromaDB + BM25\n+ 3 commercial LLMs", "Route B: Qwen2.5-7B\n+ 4-bit QLoRA",
             "Routes built and\ncontrols with Qwen", "Isolates the strategies: external context vs. tuned weights"),
            ("PHASE 4", "Technical validation", "Technical test\nof both routes", "Review by plant\ntechnical staff",
             "System ready for\nevaluation", "Confirms the routes work before evaluating"),
            ("PHASE 5", "Evaluation", "Blind DeepSeek judge\n(3 metrics, k = 3)", "User validation\n(TAM)",
             "Performance matrix\nand acceptance", "Measures performance and perceived usefulness"),
        ],
        "salida": "OUTPUT",
        "capas": ["Layer 1", "Layer 2", "Layer 3", "Layer 5"],
        "capa_nombres": ["Document\nrepositories", "Preprocessing", "Orchestration", "External evaluation"],
        "capa_detalle": ["manuals and records", "Unstructured.io · 512 tokens", "LlamaIndex", "DeepSeek judge"],
        "ruta_a": "Layer 4 · Route A", "ruta_a_sub": "Hybrid index\nChromaDB + BM25", "ruta_a_resp": "Retriever + 3 LLMs\nChatGPT · Claude · Gemini\n(no training)",
        "ruta_b": "Layer 4 · Route B", "ruta_b_sub": "JSONL dataset\n(synthetic QA pairs)", "ruta_b_resp": "LoRA adapter + Qwen\n(no retrieval)",
        "como": "How it answers",
        "juez": ["Unlabelled response\nin random order", "DeepSeek judge\n(temperature 0, k = 3)", "Faithfulness ·\nAnswer Correctness ·\nAnswer Relevance",
                 "Performance matrix\nper case and configuration"],
        "controles": ["Stability (k = 3)", "Position bias", "Self-preference", "Agreement with experts"],
        "juez_titulo": "Reliability controls",
    },
}


def caja(ax, x, y, w, h, texto, fc=CLARO, ec=GRIS, tc=TINTA, lw=1.4, size=10, bold=False, ls="-", align="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.08", fc=fc, ec=ec, lw=lw, ls=ls))
    ax.text(x + w / 2 if align == "center" else x + 0.18, y + h / 2, texto, ha=align, va="center", fontsize=size, color=tc,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def flecha(ax, p, q, color=NAVY, lw=1.8, ls="-", style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=14, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


def fig1(idioma: str):
    t = TEXTOS[idioma]
    ancho = 14.0
    fig, ax = plt.subplots(figsize=(ancho, 8.4))
    ax.set_xlim(0, ancho)
    ax.set_ylim(0, 8.4)
    ax.axis("off")
    alto, paso = 1.2, 1.62
    xf, wf, x1, x2, ws, xo, wo = 0.15, 3.2, 3.7, 7.0, 3.0, 10.3, 3.2
    for i, (fase, nombre, p1, p2, salida, proposito) in enumerate(t["fases"]):
        y = 8.4 - 0.2 - alto - i * paso
        ax.add_patch(FancyBboxPatch((xf, y), wf, alto, boxstyle="round,pad=0,rounding_size=0.06", fc=NAVY, ec=NAVY))
        ax.text(xf + 0.2, y + alto - 0.28, fase, color="#9aa6c0", fontsize=9, fontweight="bold", va="center")
        ax.text(xf + 0.2, y + 0.4, nombre, color="white", fontsize=10.5, fontweight="bold", va="center", linespacing=1.2)
        caja(ax, x1, y, ws, alto, p1, size=9.2)
        caja(ax, x2, y, ws, alto, p2, size=9.2)
        caja(ax, xo, y, wo, alto, salida, fc="white", ec=NAVY, lw=2, size=9, bold=True)
        ax.text(xo + 0.15, y + alto - 0.2, t["salida"], fontsize=7.5, color=GRIS, fontweight="bold", va="center")
        for xa in (xf + wf, x1 + ws, x2 + ws):
            flecha(ax, (xa + 0.03, y + alto / 2), (xa + 0.27, y + alto / 2), lw=1.4)
        ax.text(x1, y - 0.18, proposito, fontsize=8.8, style="italic", color="#5b6270", va="center")
        if i < 4:
            flecha(ax, (xf + wf / 2, y - 0.03), (xf + wf / 2, y - paso + alto + 0.03), lw=1.6)
    fig.savefig(DEST / f"fig1_diseno_{idioma}.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig2(idioma: str):
    t = TEXTOS[idioma]
    fig, ax = plt.subplots(figsize=(15.5, 6.6))
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 6.6)
    ax.axis("off")
    xs = [0.2, 2.65, 5.1]
    y0, w, h = 2.55, 2.0, 1.5
    for k, x in enumerate(xs):
        ax.text(x + w / 2, y0 + h + 0.62, t["capas"][k], ha="center", fontsize=11, fontweight="bold", color=NAVY)
        ax.text(x + w / 2, y0 + h + 0.22, t["capa_nombres"][k], ha="center", va="center", fontsize=8.8, color="#4a4f5a", linespacing=1.15)
        caja(ax, x, y0, w, h, t["capa_detalle"][k].replace(" · ", "\n").replace(" y ", "\ny ").replace(" and ", "\nand "), ls="--", size=8.8)
        if k < 2:
            flecha(ax, (x + w + 0.04, y0 + h / 2), (xs[k + 1] - 0.04, y0 + h / 2), color=TEAL, lw=2)
    xi, wi, xr, wr = 8.0, 2.1, 10.7, 3.0
    ax.text(xi + wi / 2, 6.25, t["ruta_a"], ha="center", fontsize=11, fontweight="bold", color=NAVY)
    caja(ax, xi, 4.45, wi, 1.55, t["ruta_a_sub"], fc="white", ec=TEAL, ls="--", size=8.8)
    caja(ax, xr, 4.45, wr, 1.55, t["ruta_a_resp"], fc="#e8f6f8", ec=TEAL, lw=2, size=8.8, bold=True)
    ax.text(xr + wr / 2, 6.25, t["como"], ha="center", fontsize=10, fontweight="bold", color=TEAL)
    flecha(ax, (xi + wi + 0.04, 5.22), (xr - 0.04, 5.22), color=TEAL, lw=2)
    ax.text(xi + wi / 2, 0.45, t["ruta_b"], ha="center", fontsize=11, fontweight="bold", color=NAVY)
    caja(ax, xi, 0.7, wi, 1.55, t["ruta_b_sub"], fc="white", ec=CORAL, ls="--", size=8.8)
    caja(ax, xr, 0.7, wr, 1.55, t["ruta_b_resp"], fc="#fdeee9", ec=CORAL, lw=2, size=8.8, bold=True)
    ax.text(xr + wr / 2, 0.35, t["como"], ha="center", fontsize=10, fontweight="bold", color=CORAL)
    flecha(ax, (xi + wi + 0.04, 1.47), (xr - 0.04, 1.47), color=CORAL, lw=2)
    flecha(ax, (xs[2] + w + 0.05, y0 + h * 0.8), (xi - 0.05, 5.0), color=TEAL, lw=2, rad=-0.25)
    flecha(ax, (xs[2] + w + 0.05, y0 + h * 0.2), (xi - 0.05, 1.7), color=CORAL, lw=2, rad=0.25)
    x5 = 14.0
    caja(ax, x5 - 0.1, y0, 1.6, h, t["capa_detalle"][3].replace(" ", "\n", 1), ls="--", size=8.8)
    ax.text(x5 + 0.7, y0 + h + 1.15, t["capas"][3], ha="center", fontsize=11, fontweight="bold", color=NAVY)
    ax.text(x5 + 0.7, y0 + h + 0.78, t["capa_nombres"][3], ha="center", va="center", fontsize=8.8, color="#4a4f5a")
    flecha(ax, (xr + wr + 0.05, 5.0), (x5 + 0.4, y0 + h + 0.03), color=TEAL, lw=2, rad=-0.2)
    flecha(ax, (xr + wr + 0.05, 1.7), (x5 + 0.4, y0 - 0.03), color=CORAL, lw=2, rad=0.2)
    fig.savefig(DEST / f"fig2_arquitectura_{idioma}.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figS1(idioma: str):
    t = TEXTOS[idioma]
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    x = 0.2
    for k, txt in enumerate(t["juez"]):
        caja(ax, x, 2.35, 2.5, 1.4, txt, fc="white" if k in (0, 3) else CLARO, ec=NAVY if k in (0, 3) else GRIS, size=9.5, bold=k in (0, 3))
        if k < 3:
            flecha(ax, (x + 2.53, 3.05), (x + 2.97, 3.05), lw=1.6)
        x += 2.85
    ax.text(0.2, 1.85, t["juez_titulo"], fontsize=10.5, fontweight="bold", color=NAVY)
    x = 0.2
    for txt in t["controles"]:
        caja(ax, x, 0.35, 2.5, 1.0, txt, fc="#eef6f6", ec=TEAL, size=9.5)
        x += 2.85
    flecha(ax, (5.9, 2.3), (5.9, 1.4), color=TEAL, lw=1.6, ls="--")
    fig.savefig(DEST / f"figS1_juez_{idioma}.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    DEST.mkdir(exist_ok=True)
    for idioma in ("es", "en"):
        for f in (fig1, fig2, figS1):
            f(idioma)
        print("guardado: fig1, fig2 y figS1 en", idioma)


if __name__ == "__main__":
    main()
