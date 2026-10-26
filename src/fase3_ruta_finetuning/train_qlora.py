"""
Fase 3 — Ruta B (Fine-Tuning): entrenamiento QLoRA de 4 bits con Unsloth sobre
el modelo base de pesos abiertos definido en config.py (Qwen2.5-7B-Instruct).

Requiere GPU. En este estudio se entrenó en una AMD RX 9070 XT (16 GB) con
ROCm sobre WSL2; los pasos de instalación están en requirements-finetune.txt.

Uso:
    # corrida corta para descartar NaN o inestabilidad (en AMD, por el bug de bitsandbytes)
    python src/fase3_ruta_finetuning/train_qlora.py --max_steps 20

    # entrenamiento completo
    python src/fase3_ruta_finetuning/train_qlora.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_steps", type=int, default=-1, help="Limita los pasos (para pruebas rápidas)")
    ap.add_argument("--output_dir", type=str, default=str(config.ROOT_DIR / "modelos" / "qwen25_7b_ft_planta_textil"))
    args = ap.parse_args()

    # Los imports pesados van dentro de main() para poder importar el módulo sin GPU.
    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig

    print(f"Cargando modelo base: {config.FT_MODELO_BASE}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.FT_MODELO_BASE,
        max_seq_length=config.FT_MAX_SEQ_LENGTH,
        dtype=None,  # Unsloth detecta automáticamente bf16/fp16 según el hardware
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.FT_LORA_R,
        lora_alpha=config.FT_LORA_ALPHA,
        lora_dropout=config.FT_LORA_DROPOUT,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    def formatear(ejemplo):
        texto = tokenizer.apply_chat_template(
            ejemplo["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": texto}

    train_path = str(config.DATA_PROCESSED / "ft_train.jsonl")
    val_path = str(config.DATA_PROCESSED / "ft_val.jsonl")
    dataset = load_dataset("json", data_files={"train": train_path, "validation": val_path})
    dataset = dataset.map(formatear)

    args_entrenamiento = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=config.FT_BATCH_SIZE,
        gradient_accumulation_steps=config.FT_GRAD_ACCUM,
        num_train_epochs=config.FT_NUM_EPOCHS,
        learning_rate=config.FT_LEARNING_RATE,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        seed=42,
        dataset_text_field="text",
        max_seq_length=config.FT_MAX_SEQ_LENGTH,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"] if len(dataset["validation"]) > 0 else None,
        args=args_entrenamiento,
    )

    print("Iniciando entrenamiento QLoRA ...")
    stats = trainer.train()
    print(stats)

    print(f"Guardando adaptador LoRA en {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Listo. Prueba el modelo con inference_ft.py.")


if __name__ == "__main__":
    main()
