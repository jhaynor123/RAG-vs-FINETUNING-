"""
Importa los 200 casos de prueba (100 del subconjunto A, derivados de los
manuales, y 100 del subconjunto B, derivados de los registros de incidencia),
que vienen en formato chat JSONL ({"messages": [system, user, assistant]}), y
los convierte al formato que usa el resto del pipeline:
casos_prueba/casos_de_prueba.jsonl.

Estos 200 pares son solo de evaluación (Fase 5). No deben entrar al dataset
de entrenamiento de la Ruta B: si el modelo viera la respuesta exacta, la
comparación mediría memorización y no generalización. El dataset de
entrenamiento se genera aparte (fase2_orquestacion.py) a partir de los
fragmentos del corpus, y descarta toda pregunta igual a un caso de prueba.

Uso:
    python casos_prueba/importar_casos_reales.py
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Máquinas de los manuales (subconjunto A); se usan para etiquetar el "tema" de cada caso.
MAQUINAS_CONOCIDAS = [
    "JUKI LU-2216N-7",
    "BROTHER BAS-326H-484",
    "SINGER 2662",
    "PEGASUS WT200",
    "BROTHER HM-8180",
]


def _extraer_tema(pregunta: str, subset: str) -> str:
    if subset == "A":
        for maquina in MAQUINAS_CONOCIDAS:
            if maquina in pregunta:
                return maquina
        return "desconocido"
    else:
        m = re.search(r"\bERR-[A-Z]+-\d+\b", pregunta)
        return m.group(0) if m else "desconocido"


def cargar_jsonl_mensajes(path: Path, subset: str) -> list[dict]:
    casos = []
    with open(path, encoding="utf-8") as f:
        for i, linea in enumerate(f):
            linea = linea.strip()
            if not linea:
                continue
            d = json.loads(linea)
            mensajes = {m["role"]: m["content"] for m in d["messages"]}
            pregunta = mensajes.get("user", "")
            respuesta = mensajes.get("assistant", "")
            casos.append(
                {
                    "caso_id": f"{subset}_{i:04d}",
                    "subset": subset,
                    "pregunta": pregunta,
                    "respuesta_referencia": respuesta,
                    "tema": _extraer_tema(pregunta, subset),
                    "system_prompt_original": mensajes.get("system", ""),
                    "fragmento_fuente": "",  # se completa con enlazar_fragmento_fuente.py
                    "origen": "real_usuario",
                }
            )
    return casos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--estructurada",
        default=str(config.ROOT_DIR / "data" / "raw" / "qa_reales" / "DATA_ESTRUCTURADA.json"),
        help="JSONL con los 100 pares del subconjunto A (manuales)",
    )
    ap.add_argument(
        "--semiestructurada",
        default=str(config.ROOT_DIR / "data" / "raw" / "qa_reales" / "DATA_SEMIESTRUCTURADA.json"),
        help="JSONL con los 100 pares del subconjunto B (registros)",
    )
    args = ap.parse_args()

    casos_a = cargar_jsonl_mensajes(Path(args.estructurada), "A")
    casos_b = cargar_jsonl_mensajes(Path(args.semiestructurada), "B")

    print(f"Subconjunto A (manuales): {len(casos_a)} casos")
    temas_a = sorted(set(c["tema"] for c in casos_a))
    print(f"  máquinas detectadas: {temas_a}")

    print(f"Subconjunto B (registros): {len(casos_b)} casos")
    temas_b = sorted(set(c["tema"] for c in casos_b))
    print(f"  {len(temas_b)} códigos de ticket distintos detectados")

    todos = casos_a + casos_b
    destino = config.CASOS_PRUEBA_DIR / "casos_de_prueba.jsonl"
    with open(destino, "w", encoding="utf-8") as f:
        for c in todos:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"\nGuardado: {destino} ({len(todos)} casos totales)")
    print("\n'fragmento_fuente' queda vacío: se completa con enlazar_fragmento_fuente.py tras indexar el corpus.")


if __name__ == "__main__":
    main()
