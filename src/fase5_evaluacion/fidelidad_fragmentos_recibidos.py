"""
Fidelidad de las respuestas RAG frente a los fragmentos que el generador REALMENTE recibió.

En la evaluación principal, Faithfulness se mide contra un único fragmento por caso (campo
fragmento_fuente), que se asignó con el mismo recuperador de la Ruta A. Eso hace circular la medida para
RAG y no distingue una alucinación de una respuesta correcta apoyada en otro fragmento recuperado.

Este script repite solo Faithfulness (una repetición por defecto, mismo juez) con el contexto exacto que recibió cada
configuración de la Ruta A (los tres fragmentos recuperados, con el formato del prompt), y clasifica
las respuestas en que el modelo declara que el contexto no tiene la información (abstención total,
parcial o ninguna), con el mismo juez.

Entradas: data/processed/resultados_respuestas.jsonl (fragmentos recuperados por caso) y corpus_*.jsonl.
Salidas (resultados/): fidelidad_fragmentos_recibidos.csv y costo_juez_fidelidad.json.
Es reanudable (fidelidad_parciales.jsonl, archivo temporal).

Uso:
    python src/fase5_evaluacion/fidelidad_fragmentos_recibidos.py
    python src/fase5_evaluacion/fidelidad_fragmentos_recibidos.py --limite 5
"""
import argparse
import csv
import json
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from metricas import USO_JUEZ, abstencion, faithfulness
from prompts import formatear_contexto

CAMPOS_RAG = {
    "RAG_Claude": "respuesta_rag_claude",
    "RAG_GPT": "respuesta_rag_gpt",
    "RAG_Gemini": "respuesta_rag_gemini",
    "Qwen_RAG_congelado": "respuesta_qwen_rag",
}

def cargar_fragmentos() -> dict:
    fragmentos = {}
    for nombre in ("corpus_A.jsonl", "corpus_B.jsonl"):
        for linea in open(config.DATA_PROCESSED / nombre, encoding="utf-8"):
            c = json.loads(linea)
            fragmentos[c["chunk_id"]] = {"source_file": c["source_file"], "text": c["text"]}
    return fragmentos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limite", type=int, default=None)
    ap.add_argument("--repeticiones", type=int, default=1,
                    help="Repeticiones de Faithfulness por respuesta (1 por defecto: su desviación entre repeticiones fue 0,010)")
    args = ap.parse_args()
    if not config.DEEPSEEK_API_KEY:
        sys.exit("DEEPSEEK_API_KEY no configurada (ver .env.example).")

    from openai import OpenAI

    cliente = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)
    filas = [json.loads(l) for l in open(config.DATA_PROCESSED / "resultados_respuestas.jsonl", encoding="utf-8")]
    if args.limite:
        filas = filas[: args.limite]
    fragmentos = cargar_fragmentos()

    parciales = config.RESULTADOS_DIR / "fidelidad_parciales.jsonl"
    hechos = {}
    if parciales.exists():
        for l in open(parciales, encoding="utf-8"):
            if l.strip():
                d = json.loads(l)
                hechos[(d["caso_id"], d["arquitectura"])] = d
        print(f"[reanudando] {len(hechos)} respuestas ya evaluadas")

    tareas = [(f, a, f[campo]) for f in filas for a, campo in CAMPOS_RAG.items()
              if f.get(campo) and (f["caso_id"], a) not in hechos]

    def correr(tarea):
        fila, arq, respuesta = tarea
        contexto = formatear_contexto([fragmentos[i] for i in fila["fragmentos_recuperados_claude"]])
        puntajes = [faithfulness(fila["pregunta"], respuesta, contexto, cliente)["score"]
                    for _ in range(args.repeticiones)]
        return {"caso_id": fila["caso_id"], "arquitectura": arq, "subset": fila["subset"],
                "faithfulness_recibidos_media": statistics.mean(puntajes),
                "faithfulness_recibidos_std": statistics.pstdev(puntajes),
                "abstencion": abstencion(fila["pregunta"], respuesta, cliente)}

    errores = []
    with ThreadPoolExecutor(max_workers=config.JUEZ_CONCURRENCIA) as pool, open(parciales, "a", encoding="utf-8") as salida:
        futuros = {pool.submit(correr, t): t for t in tareas}
        for fut in tqdm(as_completed(futuros), total=len(futuros), desc="Fidelidad (fragmentos recibidos)"):
            try:
                d = fut.result()
            except Exception as e:  # un fallo no detiene la corrida
                errores.append(f"{futuros[fut][0]['caso_id']}/{futuros[fut][1]}: {type(e).__name__}: {e}")
                continue
            hechos[(d["caso_id"], d["arquitectura"])] = d
            salida.write(json.dumps(d, ensure_ascii=False) + "\n")
            salida.flush()

    orden = [(f["caso_id"], a) for f in filas for a in CAMPOS_RAG if (f["caso_id"], a) in hechos]
    with open(config.RESULTADOS_DIR / "fidelidad_fragmentos_recibidos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["caso_id", "subset", "arquitectura", "faithfulness_recibidos_media",
                                          "faithfulness_recibidos_std", "abstencion"])
        w.writeheader()
        for clave in orden:
            w.writerow(hechos[clave])

    u = dict(USO_JUEZ)
    u["modelo"] = config.DEEPSEEK_MODEL_JUEZ
    u["costo_usd_tarifa_pico"] = round(config.costo_usd(config.DEEPSEEK_MODEL_JUEZ, u["tokens_entrada"], u["tokens_salida"]), 4)
    json.dump(u, open(config.RESULTADOS_DIR / "costo_juez_fidelidad.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"Guardado: fidelidad_fragmentos_recibidos.csv ({len(orden)} respuestas). Costo del juez: ${u['costo_usd_tarifa_pico']} USD")
    if errores:
        print(f"[aviso] {len(errores)} respuestas fallaron:", *errores[:5], sep="\n  ")


if __name__ == "__main__":
    main()
