"""
Pruebas de las piezas que no dependen de APIs, de GPU ni de los datos de la
planta: tokenizador y BM25, reparación de glifos, costos, utilidades de
estadística y consistencia de los resultados guardados.

    pytest tests
"""
import json
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "src" / "fase5_evaluacion"))

import config
from busqueda_lexica import BM25, tokenizar
from limpieza_idioma import filtrar_bloques_español, limpiar_especifico, reparar_cid_rotos


def test_tokenizador_conserva_codigos_y_sus_partes():
    tokens = tokenizar("Interpreta el ticket ERR-LUB-01")
    assert "err-lub-01" in tokens
    assert {"err", "lub", "01"} <= set(tokens)


def test_bm25_prioriza_el_codigo_exacto():
    docs = ["ticket ERR-LUB-01 bandeja de aceite", "ticket ERR-STC-04 puntada saltada", "manual de la máquina"]
    puntajes = BM25(docs).puntajes("ERR-LUB-01")
    assert puntajes.index(max(puntajes)) == 0
    assert puntajes[2] == 0


def test_reparar_cid_rotos():
    assert reparar_cid_rotos("m(cid:105)quina") == "máquina"
    assert reparar_cid_rotos("(cid:43)(cid:44)") == "HI"   # offset +29 sobre ASCII
    assert reparar_cid_rotos("a(cid:99999)b") == "ab"       # sin reconstrucción posible: se elimina


def test_filtro_de_idioma_descarta_otro_idioma():
    bloques = [
        "Antes de abrir la tapa, desconecte la máquina de la red eléctrica y espere cinco minutos.",
        "Before opening the cover, disconnect the machine from the power supply and wait five minutes.",
    ]
    conservados, stats = filtrar_bloques_español(bloques)
    assert conservados == bloques[:1]
    assert stats["bloques_descartados_por_idioma"] == 1


def test_limpieza_generica_quita_numeracion():
    assert limpiar_especifico("Texto útil\n12\nPage 3\nOtro texto", "manual.pdf") == "Texto útil\nOtro texto"


def test_costo_usd():
    # 1 M de tokens de entrada y 1 M de salida a (2,00; 10,00)
    assert config.costo_usd("claude-sonnet-5-5", 1_000_000, 1_000_000) == pytest.approx(12.0)


def test_kwargs_del_juez():
    kw = config.kwargs_juez()
    assert kw["extra_body"] == {"thinking": {"type": config.DEEPSEEK_THINKING}}
    assert config.max_tokens_juez(100) >= 100


def test_wilson_contiene_la_proporcion():
    from interfaz_validacion.graficar_validacion import wilson

    lo, hi = wilson(58, 75)
    assert lo < 58 / 75 < hi


# --- consistencia de los resultados guardados -------------------------------

@pytest.mark.skipif(not (config.RESULTADOS_DIR / "matriz_rendimiento.csv").exists(), reason="sin resultados")
def test_matriz_rendimiento_tiene_800_filas_y_rangos_validos():
    import pandas as pd

    m = pd.read_csv(config.RESULTADOS_DIR / "matriz_rendimiento.csv")
    assert len(m) == 800  # 200 casos x 4 configuraciones
    assert set(m["arquitectura"]) == {"RAG_Claude", "RAG_GPT", "RAG_Gemini", "FT_Qwen"}
    for col in ("faithfulness_media", "answer_correctness_media"):
        assert m[col].between(0, 1).all()
    # Answer Relevance es una similitud coseno: puede ser levemente negativa (3 de 800 casos, mínimo -0,022)
    assert m["answer_relevance_media"].between(-1, 1).all()


@pytest.mark.skipif(not (config.RESULTADOS_DIR / "costo_juez.json").exists(), reason="sin resultados")
def test_costo_total_de_la_campana():
    leer = lambda n: json.load(open(config.RESULTADOS_DIR / n, encoding="utf-8"))
    total = (leer("costo_generacion_rag.json")["total_usd"] + leer("costo_dataset_qa.json")["costo_usd"]
             + leer("costo_juez.json")["costo_usd_tarifa_pico"])
    assert total == pytest.approx(15.26, abs=0.01)
