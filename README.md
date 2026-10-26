# RAG frente a Fine-Tuning sobre documentación de una planta textil

Código del estudio que compara dos formas de adaptar un modelo de lenguaje a la
documentación técnica de una planta textil:

- **Ruta A, RAG:** recuperación híbrida (vectorial + BM25) y generación con tres
  modelos comerciales (Claude Sonnet 5.5, GPT-6.1-sol y Gemini 3.8 Flash).
- **Ruta B, Fine-Tuning:** Qwen2.5-7B-Instruct ajustado con QLoRA de 4 bits (Unsloth).

Las cuatro configuraciones responden 200 consultas de prueba (100 sobre manuales en
lenguaje natural y 100 sobre registros de incidencia semiestructurados), que califica
un juez automático ciego (DeepSeek-V4-Pro, tres métricas, tres repeticiones). Después,
cinco participantes valoran las respuestas con una interfaz y un instrumento basado en
el modelo de aceptación tecnológica (TAM).

## Estructura

```
config.py                         parámetros del experimento (modelos, rutas, protocolo)
src/
  fase1_ingesta.py                lectura, limpieza y segmentación de los documentos
  limpieza_idioma.py              filtro de idioma y reparación de glifos rotos de PDF
  fase2_orquestacion.py           índice vectorial (ChromaDB) y dataset sintético de pares QA
  busqueda_lexica.py              BM25 con tokenizador que conserva códigos como "ERR-LUB-01"
  embeddings_utils.py             modelo de embeddings compartido
  fase3_ruta_rag.py               Ruta A: recuperación híbrida y generadores
  fase3_ruta_finetuning/          Ruta B: preparación de datos, entrenamiento e inferencia
  fase4_validacion.py             prueba técnica de ambas rutas con unas pocas consultas
  fase5_evaluacion/               respuestas de las 4 configuraciones, juez, estadística y figuras
  interfaz_validacion/            interfaz con usuarios, consolidación y estadística de la validación
casos_prueba/                     los 200 casos de prueba y el enlace a su fragmento fuente
resultados/                       resultados, costos, figuras y registros de la validación
tests/                            pruebas de las piezas que no dependen de APIs ni de GPU
```

## Qué corre dónde

| Etapa | Requiere |
|---|---|
| Fases 1 y 2 (ingesta, índice), Fase 3 Ruta A, análisis y figuras | CPU; las fases con APIs necesitan internet y claves |
| Dataset sintético, generadores RAG, juez | Claves de API (Anthropic, OpenAI, Google, DeepSeek) |
| Entrenamiento e inferencia de la Ruta B | GPU. En el estudio, AMD RX 9070 XT con ROCm sobre WSL2 |

## Instalación

```bash
python -m venv venv
venv\Scripts\activate            # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
copy .env.example .env           # completar las claves de API
```

La Ruta B se instala aparte; los pasos (ROCm, PyTorch, bitsandbytes) están en
`requirements-finetune.txt`.

## Orden de ejecución

```bash
python src/fase1_ingesta.py                              # data/raw -> data/processed/corpus_*.jsonl
python src/fase2_orquestacion.py --solo-indexar          # índice de la Ruta A
python casos_prueba/importar_casos_reales.py             # 200 casos -> casos_de_prueba.jsonl
python casos_prueba/enlazar_fragmento_fuente.py          # fragmento fuente de cada caso (Faithfulness)
python src/fase2_orquestacion.py --solo-qa               # dataset sintético de entrenamiento

python src/fase3_ruta_finetuning/prepare_dataset.py      # train/val 90/10
python src/fase3_ruta_finetuning/train_qlora.py          # en el entorno con GPU

python src/fase4_validacion.py                           # prueba técnica

python src/fase5_evaluacion/generar_respuestas.py --solo-rag   # Ruta A (APIs)
python src/fase5_evaluacion/generar_respuestas.py --solo-ft    # Ruta B (GPU)
python src/fase5_evaluacion/juez_llm.py --sesgo-posicion       # juez ciego
python src/fase5_evaluacion/analisis_estadistico.py            # contrastes y tablas
python src/fase5_evaluacion/graficar_resultados.py             # Figs. 4, 5 y 7
```

La estadística y las figuras de la Fase 5 se pueden reproducir sin APIs ni GPU a partir
de `resultados/matriz_rendimiento.csv`.

### Validación con usuarios

```bash
# en el entorno con GPU: servidor del modelo Fine-Tuning
python src/fase3_ruta_finetuning/servidor_ft.py

# interfaz (la ven los participantes)
streamlit run src/interfaz_validacion/app.py

# al terminar: tablas, estadística y Fig. 6
python src/interfaz_validacion/consolidar_validacion.py
```

La interfaz muestra las cuatro respuestas sin su procedencia y en orden aleatorio;
`resultados/validacion_usuarios/` guarda cada consulta con la configuración real de
cada respuesta. Si el servidor del modelo no responde, la consulta no se guarda.

## Datos

Los manuales y los registros de incidencia son propiedad de la planta y no se
distribuyen (`data/raw/` no está en el repositorio). Tampoco se incluyen el índice de
ChromaDB, el adaptador LoRA entrenado ni los textos de las respuestas, que contienen
fragmentos de esa documentación. Se pueden solicitar al autor de correspondencia con
autorización de la empresa.

## Limitaciones conocidas

Están descritas en el artículo (Sección de limitaciones). En lo que toca al código:

- El control de sesgo de posición del juez no cumplió el criterio (15 % de cambios
  frente a un máximo de 2 %).
- No se ejecutaron la concordancia del juez con expertos humanos ni la condición de
  control (RAG sobre el modelo abierto con pesos congelados).
- En la validación con usuarios, la interfaz usó para las estaciones de registros (P2 y
  P4) el rol de sistema de los registros semiestructurados, aunque las consultas fueron
  de manuales; `SUBSET_POR_ESTACION` en `app.py` define esa asignación.

## Pruebas

```bash
pip install pytest
pytest tests
```
