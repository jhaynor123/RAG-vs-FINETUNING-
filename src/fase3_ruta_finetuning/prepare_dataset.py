"""
Prepara el dataset sintético QA (generado en fase2_orquestacion.py) en el
formato de conversación que espera el entrenamiento QLoRA de la Ruta B.

Entrada:  data/processed/dataset_qa_sintetico.jsonl
Salida:   data/processed/ft_train.jsonl, data/processed/ft_val.jsonl
          (formato chat: {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]})

Cada ejemplo lleva el rol de sistema de su subconjunto (config.SYSTEM_PROMPTS), como
declara el artículo. División 90/10 train/val, estratificada por subset (A/B).

Uso:
    python src/fase3_ruta_finetuning/prepare_dataset.py
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

random.seed(42)

def main():
    src = config.DATA_PROCESSED / "dataset_qa_sintetico.jsonl"
    if not src.exists():
        print(f"No existe {src}. Corre primero fase2_orquestacion.py (genera el dataset QA).")
        return

    pares = [json.loads(l) for l in open(src, encoding="utf-8")]
    if not pares:
        print("El dataset QA está vacío.")
        return

    por_subset: dict[str, list] = {}
    for p in pares:
        por_subset.setdefault(p["subset"], []).append(p)

    train, val = [], []
    for subset, items in por_subset.items():
        random.shuffle(items)
        corte = max(1, int(len(items) * 0.9)) if len(items) > 1 else len(items)
        train.extend(items[:corte])
        val.extend(items[corte:])

    def a_formato_chat(p):
        return {
            "messages": [
                {"role": "system", "content": config.SYSTEM_PROMPTS[p["subset"]]},
                {"role": "user", "content": p["pregunta"]},
                {"role": "assistant", "content": p["respuesta"]},
            ]
        }

    for nombre, split in (("ft_train.jsonl", train), ("ft_val.jsonl", val)):
        destino = config.DATA_PROCESSED / nombre
        with open(destino, "w", encoding="utf-8") as f:
            for p in split:
                f.write(json.dumps(a_formato_chat(p), ensure_ascii=False) + "\n")
        print(f"  {destino}: {len(split)} ejemplos")


if __name__ == "__main__":
    main()
