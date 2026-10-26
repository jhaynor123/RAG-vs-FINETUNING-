"""
Fig. 6 del artículo: valoración de las configuraciones por los participantes
(Parte B del instrumento), a partir de resultados/validacion_usuarios/.

  (a) Coincidencia con el manual según el participante: proporción con
      intervalo de Wilson del 95 %.
  (b) Escala Likert (1 a 5): media con IC del 95 % por ítem y configuración.
  (c) Preferencia de uso: reparto del 100 % de las consultas entre configuraciones.

Se invoca desde consolidar_validacion.py.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "fase5_evaluacion"))
import config
from graficar_resultados import (COLOR_CONFIG, CONFIGS, CORTO, MARCA_CONFIG, REJILLA, TINTA, TINTA_2, ic95)



def wilson(k, n, z=1.96):
    p = k / n
    centro = (p + z * z / (2 * n)) / (1 + z * z / n)
    mitad = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centro - mitad, centro + mitad


def limpiar(eje):
    for lado in ("top", "right"):
        eje.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        eje.spines[lado].set_color(TINTA_2)
    eje.tick_params(colors=TINTA_2, labelsize=9)


def graficar(inter: list[dict], salida=None):
    """`inter`: interacciones de la validación (una por consulta, con sus 4 respuestas)."""
    resp = {c: [r for i in inter for r in i["respuestas"] if r["config"] == c] for c in CONFIGS}
    n_cons = len(inter)

    fig = plt.figure(figsize=(13.5, 5.8))
    g = fig.add_gridspec(1, 3, width_ratios=[1, 1.5, 1], wspace=0.28)
    a, b, c = (fig.add_subplot(g[0, k]) for k in range(3))

    # (a) coincidencia: proporción con intervalo de Wilson
    for y, cfg in zip(range(len(CONFIGS))[::-1], CONFIGS):
        k = sum(r["coincide"] for r in resp[cfg]); n = len(resp[cfg])
        lo, hi = wilson(k, n)
        a.errorbar(k / n, y, xerr=[[k / n - lo], [hi - k / n]], fmt=MARCA_CONFIG[cfg], color=COLOR_CONFIG[cfg],
                   markersize=9, markeredgecolor="white", markeredgewidth=1.2, capsize=3, linewidth=1.8)
        a.text(hi + 0.025, y, f"{k / n:.2f}".replace(".", ","), fontsize=9.5, color=TINTA, ha="left", va="center")
    a.set_yticks(range(len(CONFIGS))[::-1])
    a.set_yticklabels([CORTO[x] for x in CONFIGS], fontsize=10, color=TINTA)
    a.set_xlim(0, 1.12)
    a.set_ylim(-0.6, len(CONFIGS) - 0.4)
    a.set_xlabel("Proporción de respuestas que el participante\nmarcó como coincidentes (IC de Wilson 95 %)",
                 fontsize=9.5, color=TINTA_2)
    a.set_title("(a) Coincidencia con el manual", fontsize=11.5, fontweight="bold", color=TINTA)
    a.grid(axis="x", color=REJILLA, linewidth=0.8)
    a.set_axisbelow(True)
    limpiar(a)

    # (b) Likert: media con IC 95 %
    items = [("confianza", "Confianza para\nejecutar la acción"), ("transparencia", "Advertencia ante\ninfo. ausente"),
             ("tiempo_percibido", "Tiempo de espera\npercibido")]
    for k_item, (campo, _) in enumerate(items):
        for j, cfg in enumerate(CONFIGS):
            v = np.array([r[campo] for r in resp[cfg]], dtype=float)
            x = k_item + (j - 1.5) * 0.17
            b.errorbar(x, v.mean(), yerr=ic95(v), fmt=MARCA_CONFIG[cfg], color=COLOR_CONFIG[cfg], markersize=8,
                       markeredgecolor="white", markeredgewidth=1.2, capsize=3, linewidth=1.6,
                       label=CORTO[cfg] if k_item == 0 else None)
    b.set_xticks(range(len(items)))
    b.set_xticklabels([t for _, t in items], fontsize=9.5)
    b.set_ylim(0.8, 5.2)
    b.set_ylabel("Media (1 a 5) con IC 95 %", fontsize=9.5, color=TINTA_2)
    b.set_title("(b) Escala Likert", fontsize=11.5, fontweight="bold", color=TINTA)
    b.grid(axis="y", color=REJILLA, linewidth=0.8)
    b.set_axisbelow(True)
    limpiar(b)
    h, l = b.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.03))

    # (c) preferencia: barra apilada al 100 %
    cuenta = {cfg: sum(i["config_preferida"] == cfg for i in inter) for cfg in CONFIGS}
    base = 0.0
    for cfg in CONFIGS:
        frac = cuenta[cfg] / n_cons
        if frac == 0:
            continue
        c.bar(0, frac, bottom=base, width=0.5, color=COLOR_CONFIG[cfg], edgecolor="white", linewidth=2)
        etiqueta = f"{CORTO[cfg]}  {frac * 100:.0f} %".replace(".", ",")
        c.text(0.3, base + frac / 2, etiqueta, ha="left", va="center", fontsize=9.5, color=TINTA)
        base += frac
    c.set_xlim(-0.4, 1.25)
    c.set_ylim(0, 1)
    c.set_xticks([])
    c.set_yticks([0, 0.25, 0.5, 0.75, 1])
    c.set_yticklabels(["0 %", "25 %", "50 %", "75 %", "100 %"])
    c.set_ylabel("Consultas en que fue la preferida", fontsize=9.5, color=TINTA_2)
    c.set_title("(c) Preferencia de uso", fontsize=11.5, fontweight="bold", color=TINTA)
    limpiar(c)

    salida = salida or config.RESULTADOS_DIR / "figuras" / "fig6_validacion.png"
    salida.parent.mkdir(exist_ok=True)
    plt.savefig(salida, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return salida
