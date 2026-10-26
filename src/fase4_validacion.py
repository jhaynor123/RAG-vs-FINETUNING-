"""
Fase 4 — Validación técnica del producto, antes de la evaluación a escala.

Corre un lote pequeño de consultas por cada ruta y comprueba que no haya
errores de ejecución y que la Ruta A recupere fragmentos. Guarda un CSV con
las respuestas para que personal técnico de la planta las revise; la columna
"aprobado_experto" queda vacía a propósito, para que la complete esa persona.

Uso:
    python src/fase4_validacion.py                  # ambas rutas
    python src/fase4_validacion.py --rutas rag      # solo la Ruta A
    python src/fase4_validacion.py --rutas ft       # solo la Ruta B (requiere GPU)
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from fase3_ruta_rag import responder_rag

CONSULTAS_PILOTO = [
    "¿Qué precauciones debo tomar antes de abrir la caja de control de la JUKI LU-2216N-7?",
    "¿Qué aceite lleva la JUKI LU-2216N-7 y hasta dónde se llena el depósito?",
    "Interpreta el ticket ERR-LUB-01.",
    "Interpreta el ticket ERR-TEN-01.",
    "¿Cómo se ajusta la tensión del hilo de la aguja en la SINGER 2662?",
]


def _fila(ruta: str, consulta: str, respuesta, n_fragmentos, ok: bool) -> dict:
    return {
        "ruta": ruta,
        "consulta": consulta,
        "respuesta": respuesta,
        "n_fragmentos_recuperados": n_fragmentos,
        "ejecucion_sin_error": ok,
        "aprobado_experto": "",  # lo completa el revisor humano
    }


def pilotar_ruta_rag(consultas: list[str]) -> list[dict]:
    filas = []
    for q in consultas:
        try:
            r = responder_rag(q)
            if r["respuesta"] is None:  # sin clave de API: no se llamó al generador
                filas.append(_fila("A_RAG", q, f"[sin respuesta] {r['aviso']}", len(r["fragmentos"]), False))
            else:
                filas.append(_fila("A_RAG", q, r["respuesta"], len(r["fragmentos"]), True))
        except Exception as e:
            filas.append(_fila("A_RAG", q, f"[ERROR] {e}", 0, False))
    return filas


def pilotar_ruta_ft(consultas: list[str]) -> list[dict]:
    try:
        from fase3_ruta_finetuning.inference_ft import responder_finetuning
    except Exception as e:
        print(f"[fase4] No se pudo cargar la Ruta B ({e}); requiere GPU y Unsloth instalado.")
        return []

    filas = []
    for q in consultas:
        try:
            filas.append(_fila("B_FineTuning", q, responder_finetuning(q), "N/A", True))
        except Exception as e:
            filas.append(_fila("B_FineTuning", q, f"[ERROR] {e}", "N/A", False))
    return filas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rutas", default="rag,ft")
    args = ap.parse_args()
    rutas = set(args.rutas.split(","))

    filas = []
    if "rag" in rutas:
        print(f"Ruta A (RAG): {len(CONSULTAS_PILOTO)} consultas ...")
        filas += pilotar_ruta_rag(CONSULTAS_PILOTO)
    if "ft" in rutas:
        print(f"Ruta B (Fine-Tuning): {len(CONSULTAS_PILOTO)} consultas ...")
        filas += pilotar_ruta_ft(CONSULTAS_PILOTO)

    if not filas:
        print("No se generaron resultados.")
        return

    destino = config.RESULTADOS_DIR / "fase4_piloto_validacion.csv"
    with open(destino, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        writer.writeheader()
        writer.writerows(filas)

    n_ok = sum(f["ejecucion_sin_error"] for f in filas)
    print(f"\n{n_ok}/{len(filas)} respuestas generadas sin error. CSV para revisión: {destino}")


if __name__ == "__main__":
    main()
