"""
Condiciones de control de la comparación (Tabla 1 del artículo), con el modelo abierto SIN ajustar:

  Qwen_RAG_congelado  Ruta A sobre Qwen2.5-7B-Instruct con los pesos congelados: el mismo prompt y los
                      mismos fragmentos recuperados que recibieron los tres generadores comerciales.
  Qwen_base_sin_RAG   Qwen2.5-7B-Instruct sin recuperación ni ajuste, con el mismo rol de sistema por
                      subconjunto que el modelo ajustado: es la línea base de la Ruta B.

Separan el efecto de la estrategia del efecto de la capacidad del modelo. Usan el mismo modelo base
cuantizado en 4 bits sobre el que se entrenó el adaptador LoRA, y los parámetros de generación de la
Ruta B (config.GENERACION_TEMPERATURE, config.MAX_TOKENS_RESPUESTA).

Requiere GPU (se corre en el entorno de la Ruta B). No usa ChromaDB: los fragmentos son los ya
recuperados para cada caso, guardados en data/processed/resultados_respuestas.jsonl.
Es incremental: escribe respuesta_qwen_rag y respuesta_qwen_base en ese mismo archivo.

Uso:
    python src/fase5_evaluacion/generar_controles.py
    python src/fase5_evaluacion/generar_controles.py --limite 5
"""
import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from prompts import PROMPT_GENERADOR, formatear_contexto

MAX_SEQ_LENGTH = 4096  # el prompt con tres fragmentos de 512 tokens supera los 2048 del entrenamiento
SEMILLA = 42


def cargar_fragmentos() -> dict:
    fragmentos = {}
    for nombre in ("corpus_A.jsonl", "corpus_B.jsonl"):
        for linea in open(config.DATA_PROCESSED / nombre, encoding="utf-8"):
            c = json.loads(linea)
            fragmentos[c["chunk_id"]] = {"source_file": c["source_file"], "text": c["text"]}
    return fragmentos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limite", type=int, default=None, help="Solo los primeros N casos (prueba rápida)")
    args = ap.parse_args()

    import torch
    from unsloth import FastLanguageModel

    torch.manual_seed(SEMILLA)
    print(f"Cargando el modelo base sin adaptador: {config.FT_MODELO_BASE}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.FT_MODELO_BASE, max_seq_length=MAX_SEQ_LENGTH, dtype=None, load_in_4bit=True
    )
    FastLanguageModel.for_inference(model)

    def generar(mensajes: list[dict]) -> str:
        entradas = tokenizer.apply_chat_template(
            mensajes, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        salida = model.generate(
            input_ids=entradas, attention_mask=torch.ones_like(entradas),
            max_new_tokens=config.MAX_TOKENS_RESPUESTA, temperature=config.GENERACION_TEMPERATURE, do_sample=True,
        )
        return tokenizer.decode(salida[0][entradas.shape[1]:], skip_special_tokens=True).strip()

    destino = config.DATA_PROCESSED / "resultados_respuestas.jsonl"
    filas = [json.loads(l) for l in open(destino, encoding="utf-8")]
    fragmentos = cargar_fragmentos()
    pendientes = filas[: args.limite] if args.limite else filas

    for k, fila in enumerate(tqdm(pendientes, desc="Controles (Qwen sin ajustar)")):
        if not fila.get("respuesta_qwen_rag"):
            contexto = formatear_contexto([fragmentos[i] for i in fila["fragmentos_recuperados_claude"]])
            prompt = PROMPT_GENERADOR.format(contexto=contexto, consulta=fila["pregunta"])
            fila["respuesta_qwen_rag"] = generar([{"role": "user", "content": prompt}])
        if not fila.get("respuesta_qwen_base"):
            fila["respuesta_qwen_base"] = generar([
                {"role": "system", "content": config.SYSTEM_PROMPTS[fila["subset"]]},
                {"role": "user", "content": fila["pregunta"]},
            ])
        if (k + 1) % 10 == 0:  # guardado incremental
            with open(destino, "w", encoding="utf-8") as f:
                for x in filas:
                    f.write(json.dumps(x, ensure_ascii=False) + "\n")

    with open(destino, "w", encoding="utf-8") as f:
        for x in filas:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"Guardado: {destino}")
    for campo in ("respuesta_qwen_rag", "respuesta_qwen_base"):
        print(f"  {campo}: {sum(1 for x in filas if x.get(campo))}/{len(filas)}")


if __name__ == "__main__":
    main()
