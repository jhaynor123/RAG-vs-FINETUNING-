"""
Inferencia con el modelo de la Ruta B (Fine-Tuning) ya entrenado.

A diferencia de la Ruta A, aquí no se recupera contexto ni se llama a una
API: el modelo responde solo con lo que aprendió en sus pesos durante el
entrenamiento QLoRA. Esa diferencia es lo que mide la Fase 5.

Requiere GPU (igual que train_qlora.py). Si config.FT_SERVIDOR_URL está
definida, la consulta se envía al servidor local de servidor_ft.py.

Uso:
    python src/fase3_ruta_finetuning/inference_ft.py "tu pregunta aquí"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

_modelo_cargado: dict = {}  # adapter_dir -> (model, tokenizer); cargar tarda ~40 s, no repetirlo por consulta


def _cargar_modelo(adapter_dir: str):
    if adapter_dir not in _modelo_cargado:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=adapter_dir,  # carga el modelo base + adaptador LoRA guardado
            max_seq_length=config.FT_MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        _modelo_cargado[adapter_dir] = (model, tokenizer)
    return _modelo_cargado[adapter_dir]


def responder_finetuning(consulta: str, adapter_dir: str = None, subset: str = "A") -> str:
    """`subset` ("A" manuales / "B" registros) elige el rol de sistema con el que se entrenó ese tipo de dato."""
    if config.FT_SERVIDOR_URL:
        return _responder_por_servidor(consulta, subset)
    return _responder_local(consulta, adapter_dir, subset)


def _responder_por_servidor(consulta: str, subset: str) -> str:
    import json
    import urllib.request

    peticion = urllib.request.Request(
        config.FT_SERVIDOR_URL.rstrip("/") + "/responder",
        data=json.dumps({"consulta": consulta, "subset": subset}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(peticion, timeout=300) as r:
        return json.loads(r.read().decode("utf-8"))["respuesta"]


def _responder_local(consulta: str, adapter_dir: str = None, subset: str = "A") -> str:
    adapter_dir = adapter_dir or str(config.ROOT_DIR / "modelos" / "qwen25_7b_ft_planta_textil")
    import torch

    model, tokenizer = _cargar_modelo(adapter_dir)

    mensajes = [
        {"role": "system", "content": config.SYSTEM_PROMPTS[subset]},
        {"role": "user", "content": consulta},
    ]
    entradas = tokenizer.apply_chat_template(
        mensajes, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    salida = model.generate(
        input_ids=entradas, attention_mask=torch.ones_like(entradas),
        max_new_tokens=config.MAX_TOKENS_RESPUESTA,
        temperature=config.GENERACION_TEMPERATURE, do_sample=True
    )
    texto = tokenizer.decode(salida[0][entradas.shape[1] :], skip_special_tokens=True)
    return texto.strip()


if __name__ == "__main__":
    consulta = " ".join(sys.argv[1:]) or "¿Qué código de error indica sobrecalentamiento del motor principal?"
    print(f"Consulta: {consulta}\n")
    print("Respuesta (Ruta B / Fine-Tuning):")
    print(responder_finetuning(consulta))
