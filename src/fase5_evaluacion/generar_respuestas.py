"""
Corre los 200 casos de prueba (casos_prueba/casos_de_prueba.jsonl) por las
cuatro configuraciones: los tres generadores de la Ruta A (Claude, GPT,
Gemini) y el modelo de la Ruta B (Qwen2.5-7B), con los mismos parámetros de
generación (config.GENERACION_TEMPERATURE, config.MAX_TOKENS_RESPUESTA).

Los campos que escribe (respuesta_rag_claude, respuesta_rag_gpt,
respuesta_rag_gemini, respuesta_ft) deben coincidir con CAMPOS_RESPUESTA de
juez_llm.py.

La Ruta B necesita GPU. El script es incremental sobre
data/processed/resultados_respuestas.jsonl: la Ruta A se puede correr en una
máquina (--solo-rag) y la Ruta B en otra (--solo-ft).

Uso:
    python src/fase5_evaluacion/generar_respuestas.py
    python src/fase5_evaluacion/generar_respuestas.py --proveedores claude,gpt
    python src/fase5_evaluacion/generar_respuestas.py --solo-rag
    python src/fase5_evaluacion/generar_respuestas.py --solo-ft
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# proveedor -> campo en resultados_respuestas.jsonl (ver CAMPOS_RESPUESTA en juez_llm.py)
CAMPO_POR_PROVEEDOR = {
    "claude": "respuesta_rag_claude",
    "gpt": "respuesta_rag_gpt",
    "gemini": "respuesta_rag_gemini",
}
MODELO_POR_PROVEEDOR = {
    "claude": config.CLAUDE_MODEL_GENERADOR,
    "gpt": config.GPT_MODEL_GENERADOR,
    "gemini": config.GEMINI_MODEL_GENERADOR,
}
CONCURRENCIA_RAG = 8


def cargar_casos() -> list[dict]:
    path = config.CASOS_PRUEBA_DIR / "casos_de_prueba.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def generar_rag(casos: list[dict], resultados: dict, proveedores: list[str]):
    """Ruta A: casos x proveedores en paralelo. Un fallo se registra en el
    campo error_<proveedor> sin detener el resto. Guarda el costo real."""
    from fase3_ruta_rag import USO, responder_rag  # import tardío: ChromaDB no se instala en el entorno de WSL

    def una(tarea):
        caso, proveedor = tarea
        try:
            r = responder_rag(caso["pregunta"], proveedor=proveedor)
            return caso["caso_id"], proveedor, r.get("respuesta"), [f["chunk_id"] for f in r["fragmentos"]], None
        except Exception as e:
            return caso["caso_id"], proveedor, None, [], f"{type(e).__name__}: {e}"

    tareas = [(c, p) for c in casos for p in proveedores]
    with ThreadPoolExecutor(max_workers=CONCURRENCIA_RAG) as pool:
        for cid, proveedor, respuesta, fragmentos, error in tqdm(pool.map(una, tareas), total=len(tareas), desc="Ruta A (RAG)"):
            fila = resultados[cid]
            fila[CAMPO_POR_PROVEEDOR[proveedor]] = respuesta
            fila[f"fragmentos_recuperados_{proveedor}"] = fragmentos
            if error:
                fila[f"error_{proveedor}"] = error

    costos = {}
    for proveedor in proveedores:
        u = dict(USO[proveedor])
        modelo = MODELO_POR_PROVEEDOR[proveedor]
        u["modelo"] = modelo
        u["costo_usd"] = round(config.costo_usd(modelo, u["tokens_entrada"], u["tokens_salida"]), 4)
        costos[proveedor] = u
    costos["total_usd"] = round(sum(v["costo_usd"] for v in costos.values()), 4)
    with open(config.RESULTADOS_DIR / "costo_generacion_rag.json", "w", encoding="utf-8") as f:
        json.dump(costos, f, indent=2, ensure_ascii=False)
    print("  Costo real de la generación RAG:", json.dumps({k: (v["costo_usd"] if isinstance(v, dict) else v) for k, v in costos.items()}))


def generar_ft(casos: list[dict], resultados: dict):
    """Ruta B: requiere GPU (WSL); secuencial, el modelo vive en la GPU."""
    from fase3_ruta_finetuning.inference_ft import responder_finetuning

    for caso in tqdm(casos, desc="Ruta B (Fine-Tuning)"):
        fila = resultados[caso["caso_id"]]
        try:
            fila["respuesta_ft"] = responder_finetuning(caso["pregunta"], subset=caso["subset"])
        except Exception as e:
            fila["respuesta_ft"] = None
            fila["error_ft"] = f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo-rag", action="store_true")
    ap.add_argument("--solo-ft", action="store_true")
    ap.add_argument(
        "--proveedores",
        default="claude,gpt,gemini",
        help="Lista separada por comas de generadores de la Ruta A a correr (default: los 3)",
    )
    ap.add_argument("--limite", type=int, default=None, help="Solo los primeros N casos (prueba rápida)")
    args = ap.parse_args()
    proveedores = [p.strip() for p in args.proveedores.split(",") if p.strip()]
    for p in proveedores:
        if p not in CAMPO_POR_PROVEEDOR:
            raise SystemExit(f"Proveedor desconocido: {p!r}. Opciones: {list(CAMPO_POR_PROVEEDOR)}")

    casos = cargar_casos()
    if not casos:
        print("No hay casos en casos_prueba/casos_de_prueba.jsonl. Corre primero importar_casos_reales.py")
        return
    if args.limite:
        casos = casos[: args.limite]

    destino = config.DATA_PROCESSED / "resultados_respuestas.jsonl"
    resultados = {}
    if destino.exists():
        # incremental: la Ruta A corre en Windows y la Ruta B en WSL (GPU), sobre el mismo archivo
        for l in open(destino, encoding="utf-8"):
            d = json.loads(l)
            resultados[d["caso_id"]] = d
    for caso in casos:
        resultados.setdefault(caso["caso_id"], {**caso})

    if not args.solo_ft:
        generar_rag(casos, resultados, proveedores)
    if not args.solo_rag:
        generar_ft(casos, resultados)

    with open(destino, "w", encoding="utf-8") as f:
        for fila in resultados.values():
            f.write(json.dumps(fila, ensure_ascii=False) + "\n")

    print(f"\nGuardado: {destino}")
    for proveedor, campo in CAMPO_POR_PROVEEDOR.items():
        print(f"  Respuestas RAG · {proveedor}: {sum(1 for f in resultados.values() if f.get(campo))}/{len(resultados)}")
    print(f"  Respuestas Fine-Tuning (Qwen): {sum(1 for f in resultados.values() if f.get('respuesta_ft'))}/{len(resultados)}")


if __name__ == "__main__":
    main()
