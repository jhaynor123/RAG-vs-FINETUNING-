"""
Completa el campo "fragmento_fuente" de casos_de_prueba.jsonl (que deja vacío
importar_casos_reales.py): busca en el índice de la Fase 2 el fragmento más
parecido a cada pregunta, con la misma recuperación híbrida (vectorial + BM25)
de la Ruta A pero restringida al subconjunto del caso. "score_enlace" es el
puntaje híbrido normalizado a [0, 1]. Faithfulness (Fase 5) verifica cada
respuesta contra ese fragmento.

Debe correrse después de fase1_ingesta.py y fase2_orquestacion.py --solo-indexar.

Uso:
    python casos_prueba/enlazar_fragmento_fuente.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import config
from fase3_ruta_rag import recuperar


def cargar_casos() -> list[dict]:
    path = config.CASOS_PRUEBA_DIR / "casos_de_prueba.jsonl"
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def main():
    casos = cargar_casos()
    if not casos:
        print("No hay casos en casos_de_prueba.jsonl. Corre primero importar_casos_reales.py")
        return

    try:
        recuperar("prueba", top_k=1, top_k_final=1)
    except RuntimeError as e:
        print(e)
        return

    enlazados, sin_match = 0, 0
    for caso in casos:
        mejor = recuperar(caso["pregunta"], top_k=1, top_k_final=1, subset=caso["subset"])
        if mejor:
            caso["fragmento_fuente"] = mejor[0]["text"]
            caso["source_file"] = mejor[0]["metadata"]["source_file"]
            caso["score_enlace"] = round(mejor[0]["score"], 4)
            enlazados += 1
        else:
            sin_match += 1

    destino = config.CASOS_PRUEBA_DIR / "casos_de_prueba.jsonl"
    with open(destino, "w", encoding="utf-8") as f:
        for c in casos:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"\nEnlazados: {enlazados}/{len(casos)} (sin match: {sin_match})")
    print(f"Guardado: {destino}")
    print("Un 'score_enlace' bajo indica que la pregunta tiene poco respaldo en el corpus indexado.")


if __name__ == "__main__":
    main()
