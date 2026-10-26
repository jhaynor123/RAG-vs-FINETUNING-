"""
Fase 5 — Juez ciego con DeepSeek.

Para cada caso de prueba y cada configuración (RAG_Claude, RAG_GPT,
RAG_Gemini, FT_Qwen) calcula las tres métricas de metricas.py k veces
(config.K_REPETICIONES_JUEZ), para medir la estabilidad del propio juez. Las
respuestas que lee salen de generar_respuestas.py (CAMPOS_RESPUESTA).

DeepSeek no comparte proveedor con ninguno de los tres generadores de la
Ruta A, por lo que no hace falta excluirlo de ninguna comparación (control de
auto-preferencia, Tabla 3).

Las tres métricas puntúan cada respuesta por separado, así que no hay una
"posición" que pueda sesgarlas. Aun así, verificar_sesgo_de_posicion() cubre
ese control de la Tabla 3: presenta las cuatro respuestas juntas, sin etiqueta
y en dos órdenes distintos, y mide si la preferencia del juez cambia solo por
el orden.

La evaluación corre en paralelo y es reanudable: cada respuesta evaluada se
guarda al instante en resultados/juicios_parciales.jsonl (archivo temporal
que se puede borrar al terminar).

Salidas en resultados/:
  juicios_detalle.jsonl    cada juicio individual, con su justificación
  matriz_rendimiento.csv   promedio por caso y configuración (entrada de analisis_estadistico.py)
  costo_juez.json          llamadas, tokens y costo real
  sesgo_de_posicion.json   solo con --sesgo-posicion

Uso:
    python src/fase5_evaluacion/juez_llm.py
    python src/fase5_evaluacion/juez_llm.py --limite 10 --sesgo-posicion
"""
import argparse
import csv
import json
import random
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config
from metricas import USO_JUEZ, evaluar_respuesta, registrar_uso_juez

random.seed(7)

# Etiqueta de configuración (la usa también analisis_estadistico.py) -> campo de resultados_respuestas.jsonl.
CAMPOS_RESPUESTA = {
    "RAG_Claude": "respuesta_rag_claude",
    "RAG_GPT": "respuesta_rag_gpt",
    "RAG_Gemini": "respuesta_rag_gemini",
    "FT_Qwen": "respuesta_ft",
}


def cargar_resultados_respuestas() -> list[dict]:
    path = config.DATA_PROCESSED / "resultados_respuestas.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def _evaluar_una(fila: dict, arquitectura: str, respuesta: str, cliente) -> tuple[list, dict]:
    """k repeticiones de las 3 métricas para UNA respuesta. Devuelve (juicios, fila de la matriz)."""
    juicios = []
    scores = {"faithfulness": [], "answer_correctness": [], "answer_relevance": []}
    for k in range(1, config.K_REPETICIONES_JUEZ + 1):
        m = evaluar_respuesta(
            pregunta=fila["pregunta"],
            respuesta=respuesta,
            referencia=fila["respuesta_referencia"],
            contexto=fila["fragmento_fuente"],
            cliente=cliente,
        )
        juicios.append(
            {
                "caso_id": fila["caso_id"],
                "subset": fila["subset"],
                "arquitectura": arquitectura,
                "repeticion": k,
                "faithfulness": m["faithfulness"]["score"],
                "answer_correctness": m["answer_correctness"]["score"],
                "answer_relevance": m["answer_relevance"]["score"],
                "detalle": m,
            }
        )
        for nombre in scores:
            scores[nombre].append(m[nombre]["score"])

    matriz = {
        "caso_id": fila["caso_id"],
        "subset": fila["subset"],
        "arquitectura": arquitectura,
        "faithfulness_media": statistics.mean(scores["faithfulness"]),
        "faithfulness_std": statistics.pstdev(scores["faithfulness"]),
        "answer_correctness_media": statistics.mean(scores["answer_correctness"]),
        "answer_correctness_std": statistics.pstdev(scores["answer_correctness"]),
        "answer_relevance_media": statistics.mean(scores["answer_relevance"]),
        "answer_relevance_std": statistics.pstdev(scores["answer_relevance"]),
        "alucinacion": statistics.mean(scores["faithfulness"]) < config.UMBRAL_ALUCINACION_FAITHFULNESS,
    }
    return juicios, matriz


def evaluar_todo(filas: list[dict], cliente, limite: int = None) -> tuple[list, list, list]:
    """Evalúa todas las respuestas en paralelo (config.JUEZ_CONCURRENCIA hilos).

    Un fallo en una respuesta no detiene la corrida: queda en `errores` y esa
    respuesta se omite de la matriz. Si la corrida se corta, al relanzarla se
    saltan las respuestas ya guardadas en juicios_parciales.jsonl.
    """
    if limite:
        filas = filas[:limite]

    parciales = config.RESULTADOS_DIR / "juicios_parciales.jsonl"
    hechos = {}
    if parciales.exists():
        for l in open(parciales, encoding="utf-8"):
            if l.strip():
                d = json.loads(l)
                hechos[(d["caso_id"], d["arquitectura"])] = (d["juicios"], d["matriz"])
        print(f"[reanudando] {len(hechos)} respuestas ya evaluadas en {parciales.name}")

    todas = [
        (fila, arquitectura, fila[campo])
        for fila in filas
        for arquitectura, campo in CAMPOS_RESPUESTA.items()
        if fila.get(campo)
    ]
    pendientes = [t for t in todas if (t[0]["caso_id"], t[1]) not in hechos]

    def correr(tarea):
        fila, arquitectura, respuesta = tarea
        try:
            return tarea, _evaluar_una(fila, arquitectura, respuesta, cliente), None
        except Exception as e:
            return tarea, None, {"caso_id": fila["caso_id"], "arquitectura": arquitectura, "error": f"{type(e).__name__}: {e}"}

    errores = []
    with ThreadPoolExecutor(max_workers=config.JUEZ_CONCURRENCIA) as pool, open(parciales, "a", encoding="utf-8") as salida:
        futuros = [pool.submit(correr, t) for t in pendientes]
        for futuro in tqdm(as_completed(futuros), total=len(futuros), desc="Evaluando respuestas"):
            (fila, arquitectura, _), resultado, error = futuro.result()
            if error:
                errores.append(error)
                continue
            juicios, matriz = resultado
            hechos[(fila["caso_id"], arquitectura)] = (juicios, matriz)
            salida.write(json.dumps({"caso_id": fila["caso_id"], "arquitectura": arquitectura,
                                     "juicios": juicios, "matriz": matriz}, ensure_ascii=False) + "\n")
            salida.flush()

    # orden estable: el de las respuestas originales
    juicios_total, matriz_total = [], []
    for fila, arquitectura, _ in todas:
        if (fila["caso_id"], arquitectura) in hechos:
            j, m = hechos[(fila["caso_id"], arquitectura)]
            juicios_total.extend(j)
            matriz_total.append(m)

    if errores:
        print(f"\n[aviso] {len(errores)}/{len(pendientes)} respuestas no se pudieron evaluar (ver juicios_errores.jsonl)")
    return juicios_total, matriz_total, errores


PROMPT_PREFERENCIA_4VIAS = """Estás comparando 4 respuestas a la misma consulta técnica, presentadas SIN etiqueta de procedencia y en orden aleatorio. No sabes qué sistema generó cada una.

Consulta: {pregunta}

Respuesta 1:
\"\"\"
{r1}
\"\"\"

Respuesta 2:
\"\"\"
{r2}
\"\"\"

Respuesta 3:
\"\"\"
{r3}
\"\"\"

Respuesta 4:
\"\"\"
{r4}
\"\"\"

¿Cuál responde mejor la consulta? Responde SOLO con JSON: {{"preferida": "1", "2", "3" o "4", "justificacion": "..."}}"""


def verificar_sesgo_de_posicion(filas: list[dict], cliente, n_muestra: int = 20) -> dict:
    """Control de sesgo de posición (Tabla 3): presenta las cuatro respuestas
    sin etiqueta en dos órdenes distintos y mide con qué frecuencia el juez
    cambia de preferida solo por el orden."""
    etiquetas = list(CAMPOS_RESPUESTA.keys())
    candidatos = [f for f in filas if all(f.get(CAMPOS_RESPUESTA[e]) for e in etiquetas)]
    muestra = random.sample(candidatos, min(n_muestra, len(candidatos)))

    coincidencias_por_posicion = 0
    for fila in muestra:
        respuestas = {e: fila[CAMPOS_RESPUESTA[e]] for e in etiquetas}

        orden_1 = etiquetas[:]
        random.shuffle(orden_1)
        orden_2 = etiquetas[:]
        random.shuffle(orden_2)
        while orden_2 == orden_1:  # el segundo orden debe ser distinto del primero
            random.shuffle(orden_2)

        pref_1 = _preferida_a_etiqueta(_llamar_preferencia_4vias(cliente, fila["pregunta"], [respuestas[e] for e in orden_1]), orden_1)
        pref_2 = _preferida_a_etiqueta(_llamar_preferencia_4vias(cliente, fila["pregunta"], [respuestas[e] for e in orden_2]), orden_2)

        if pref_1 == pref_2:
            coincidencias_por_posicion += 1

    pct_estable = 100 * coincidencias_por_posicion / len(muestra) if muestra else 0
    desfase_pct = 100 - pct_estable
    return {
        "n_muestra": len(muestra),
        "pct_preferencia_estable_ante_cambio_de_orden": pct_estable,
        "desfase_por_posicion_pct": desfase_pct,
        "dentro_del_umbral": desfase_pct <= config.MAX_DESFASE_POSICION_PCT,
    }


def _preferida_a_etiqueta(juicio: dict, orden: list[str]) -> str | None:
    """Traduce la posición ("1".."4") que eligió el juez de vuelta a la
    etiqueta de arquitectura real, usando el orden en que se presentaron
    esa vez."""
    pos = juicio.get("preferida")
    if pos not in ("1", "2", "3", "4"):
        return None
    return orden[int(pos) - 1]


def _llamar_preferencia_4vias(cliente, pregunta: str, respuestas_en_orden: list[str]) -> dict:
    resp = cliente.chat.completions.create(
        model=config.DEEPSEEK_MODEL_JUEZ,
        max_tokens=config.max_tokens_juez(1536),
        **config.kwargs_juez(),
        messages=[
            {
                "role": "user",
                "content": PROMPT_PREFERENCIA_4VIAS.format(
                    pregunta=pregunta,
                    r1=respuestas_en_orden[0],
                    r2=respuestas_en_orden[1],
                    r3=respuestas_en_orden[2],
                    r4=respuestas_en_orden[3],
                ),
            }
        ],
    )
    registrar_uso_juez(resp)
    if resp.choices[0].finish_reason == "length":
        raise ValueError("Respuesta del juez (preferencia 4 vías) truncada por max_tokens -- sube el límite.")
    bruto = resp.choices[0].message.content.strip().strip("`")
    inicio, fin = bruto.find("{"), bruto.rfind("}") + 1
    return json.loads(bruto[inicio:fin])


def guardar_resultados(juicios: list[dict], matriz: list[dict], errores: list[dict] = ()):
    with open(config.RESULTADOS_DIR / "juicios_detalle.jsonl", "w", encoding="utf-8") as f:
        for j in juicios:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")

    if errores:
        with open(config.RESULTADOS_DIR / "juicios_errores.jsonl", "w", encoding="utf-8") as f:
            for e in errores:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    if matriz:
        with open(config.RESULTADOS_DIR / "matriz_rendimiento.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(matriz[0].keys()))
            writer.writeheader()
            writer.writerows(matriz)

    print(f"\nGuardado: {config.RESULTADOS_DIR / 'juicios_detalle.jsonl'} ({len(juicios)} juicios)")
    print(f"Guardado: {config.RESULTADOS_DIR / 'matriz_rendimiento.csv'} ({len(matriz)} filas)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limite", type=int, default=None, help="Limita el número de casos (pruebas rápidas)")
    ap.add_argument("--sesgo-posicion", action="store_true", help="Corre también el chequeo de sesgo de posición")
    args = ap.parse_args()

    if not config.DEEPSEEK_API_KEY:
        print("DEEPSEEK_API_KEY no configurada (ver .env.example).")
        return

    from openai import OpenAI

    cliente = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)

    filas = cargar_resultados_respuestas()
    if not filas:
        print("No hay resultados en data/processed/resultados_respuestas.jsonl. Corre primero generar_respuestas.py")
        return

    print("=== Fase 5: Juez ciego con DeepSeek ===\n")
    juicios, matriz, errores = evaluar_todo(filas, cliente, limite=args.limite)
    guardar_resultados(juicios, matriz, errores)
    u = dict(USO_JUEZ)
    u["modelo"] = config.DEEPSEEK_MODEL_JUEZ
    u["costo_usd_tarifa_pico"] = round(config.costo_usd(config.DEEPSEEK_MODEL_JUEZ, u["tokens_entrada"], u["tokens_salida"]), 4)
    with open(config.RESULTADOS_DIR / "costo_juez.json", "w", encoding="utf-8") as f:
        json.dump(u, f, indent=2, ensure_ascii=False)
    print(f"Costo real del juez (tarifa pico): ${u['costo_usd_tarifa_pico']} USD ({u['llamadas']} llamadas)")

    if args.sesgo_posicion:
        print("\nChequeo de sesgo de posición ...")
        resultado_sesgo = verificar_sesgo_de_posicion(filas, cliente)
        print(json.dumps(resultado_sesgo, indent=2, ensure_ascii=False))
        with open(config.RESULTADOS_DIR / "sesgo_de_posicion.json", "w", encoding="utf-8") as f:
            json.dump(resultado_sesgo, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
