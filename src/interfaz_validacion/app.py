"""
Interfaz de validación con usuarios (segundo nivel de evaluación del artículo).

El participante escribe una consulta; el sistema la envía a las cuatro
configuraciones (tres generadores RAG y el modelo Fine-Tuning) y muestra las
cuatro respuestas sin indicar su procedencia y en orden aleatorio, con el
tiempo de cada una. El participante valora cada respuesta (Parte B del
instrumento) y elige la que usaría; la configuración real de cada respuesta
solo se guarda en el registro. Al cierre llena una vez la Parte A, sobre el
sistema en general.

Todo queda en resultados/validacion_usuarios/ como JSONL (interacciones.jsonl
y tam_parte_a.jsonl). consolidar_validacion.py lo resume.

El modelo Fine-Tuning se consulta al servidor local de servidor_ft.py (en WSL,
con la GPU); sin él, esa configuración no responde y la consulta no se guarda.

Uso:
    streamlit run src/interfaz_validacion/app.py
"""
import json
import random
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # raíz del proyecto -> config
sys.path.insert(0, str(Path(__file__).parent.parent))  # src/ -> fase3_ruta_rag, fase3_ruta_finetuning
os.environ.setdefault("FT_SERVIDOR_URL", "http://localhost:8765")  # servidor_ft.py en WSL (GPU)
import config
from fase3_ruta_rag import responder_rag

LOG_DIR = config.RESULTADOS_DIR / "validacion_usuarios"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_INTERACCIONES = LOG_DIR / "interacciones.jsonl"
LOG_TAM_PARTE_A = LOG_DIR / "tam_parte_a.jsonl"

ESTACIONES = [
    "Preparación y ajuste de máquina",
    "Operación en línea y formación de puntada",
    "Mantenimiento y lubricación en línea",
    "Seguridad y control operativo",
    "Supervisión transversal",
]

# Tipo de dato asignado a cada estación (Tabla 4): define el rol de sistema con el
# que responde el modelo Fine-Tuning. Supervisión usa el de lenguaje natural.
SUBSET_POR_ESTACION = dict(zip(ESTACIONES, ["A", "B", "A", "B", "A"]))

# Etiqueta interna (nunca se muestra al participante) -> función que genera
# la respuesta. FT_Qwen usa la Ruta B (sin recuperación, sin API).
CONFIGURACIONES = ["RAG_Claude", "RAG_GPT", "RAG_Gemini", "FT_Qwen"]

ITEMS_PARTE_A = [
    ("UP1", "El sistema me permite resolver más rápido las consultas técnicas de mi puesto."),
    ("UP2", "Usar el sistema mejora mi desempeño en las tareas de operación y mantenimiento."),
    ("UP3", "El sistema me ahorra tiempo de búsqueda en los manuales y en los registros de incidencias."),
    ("UP4", "En general, considero que el sistema es útil para mi trabajo diario."),
    ("FU1", "Aprender a usar el sistema me resultó sencillo."),
    ("FU2", "Formular mis consultas al sistema no me exigió un esfuerzo adicional."),
    ("FU3", "Las respuestas del sistema son claras y fáciles de interpretar."),
    ("IU1", "Usaría el sistema de forma habitual en mi puesto de trabajo."),
    ("IU2", "Recomendaría el sistema a un compañero de mi área."),
]


def _append_jsonl(path: Path, registro: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(registro, ensure_ascii=False) + "\n")


def _leer_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def _generar_respuesta(config_id: str, pregunta: str, subset: str) -> dict:
    """Llama al generador correspondiente y mide el tiempo. Nunca lanza --
    si falla o no hay clave configurada, lo dice en el propio resultado en
    vez de romper la sesión de un participante en pleno turno."""
    t0 = time.time()
    try:
        if config_id == "FT_Qwen":
            from fase3_ruta_finetuning.inference_ft import responder_finetuning
            texto = responder_finetuning(pregunta, subset=subset)
        else:
            proveedor = config_id.split("_", 1)[1].lower()
            r = responder_rag(pregunta, proveedor=proveedor)
            texto = r.get("respuesta")
            if texto is None:
                return {"config": config_id, "texto": None, "tiempo_seg": None,
                         "disponible": False, "motivo": r.get("aviso", "sin respuesta")}
        return {"config": config_id, "texto": texto, "tiempo_seg": round(time.time() - t0, 2), "disponible": True}
    except Exception as e:
        return {"config": config_id, "texto": None, "tiempo_seg": None,
                 "disponible": False, "motivo": f"{type(e).__name__}: {e}"}


def _servidor_ft_activo() -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen(config.FT_SERVIDOR_URL.rstrip("/") + "/salud", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def pantalla_identificacion():
    st.title("Sistema de soporte técnico — planta textil")
    st.caption("Antes de empezar, identifícate. Esto se pide una sola vez por sesión.")
    with st.form("identificacion"):
        participante = st.text_input("Tu identificador (ej. P1, P2...)")
        estacion = st.selectbox("Tu estación de trabajo", ESTACIONES)
        enviado = st.form_submit_button("Empezar")
    if enviado and participante.strip():
        st.session_state["participante"] = participante.strip()
        st.session_state["estacion"] = estacion
        st.rerun()


ESCALA = [1, 2, 3, 4, 5]
ITEMS_PARTE_B = [
    ("confianza", "Confiaría en esta respuesta para ejecutar una acción sobre la máquina."),
    ("transparencia", "Cuando el sistema no tiene la información, lo indica en lugar de responder de todos modos."),
    ("tiempo_percibido", "El tiempo de espera fue adecuado para el ritmo de mi trabajo."),
]


@st.cache_resource(show_spinner=False)
def _calentar():
    """Carga el modelo de embeddings y el índice UNA vez, para que el tiempo que ve (y se registra
    de) la primera respuesta no incluya esa carga."""
    from fase3_ruta_rag import recuperar

    recuperar("calentamiento")
    return True


def tab_consultas():
    st.subheader("Haz tu consulta")
    st.caption(
        "Escribe tu pregunta como se la harías a un compañero técnico. "
        "El sistema va a mostrarte varias respuestas posibles, en orden aleatorio "
        "y sin decir de dónde viene cada una."
    )
    n_guardadas = st.session_state.get("n_guardadas", 0)
    pregunta = st.text_area("Tu consulta", key=f"pregunta_actual_{n_guardadas}", height=80)

    if st.button("Preguntar", type="primary", disabled=not pregunta.strip()):
        orden = CONFIGURACIONES[:]
        random.shuffle(orden)
        subset = SUBSET_POR_ESTACION[st.session_state["estacion"]]
        with st.spinner("Consultando... (puede tardar unos segundos)"):
            with ThreadPoolExecutor(max_workers=len(orden)) as pool:
                resultados = list(pool.map(lambda c: _generar_respuesta(c, pregunta, subset), orden))
        st.session_state["n_consulta"] = st.session_state.get("n_consulta", 0) + 1
        st.session_state["resultados_pendientes"] = resultados
        st.session_state["pregunta_pendiente"] = pregunta

    resultados = st.session_state.get("resultados_pendientes")
    if not resultados:
        return

    st.divider()
    disponibles = [r for r in resultados if r["disponible"]]
    if len(disponibles) < len(resultados):
        # Con menos de 4 configuraciones la comparación ciega queda incompleta: no se registra nada.
        st.error(
            f"{len(resultados) - len(disponibles)} de {len(resultados)} configuraciones no respondió, así que esta "
            "consulta NO se puede registrar. Avisa a quien coordina la sesión (falta el servidor del modelo o hay un "
            "problema con una API) y vuelve a intentar."
        )
        for r in resultados:
            if not r["disponible"]:
                st.caption(f"{r['config']}: {r.get('motivo', 'sin respuesta')}")
        return

    n = st.session_state["n_consulta"]  # las claves de los controles cambian en cada consulta: nada se arrastra
    st.caption("Escala de 1 a 5: 1 = totalmente en desacuerdo, 5 = totalmente de acuerdo.")
    valoraciones = {}
    etiquetas_visibles = [f"Respuesta {chr(65 + i)}" for i in range(len(disponibles))]
    for etiqueta, r in zip(etiquetas_visibles, disponibles):
        with st.container(border=True):
            st.markdown(f"**{etiqueta}**  ·  _{r['tiempo_seg']} s_")
            st.write(r["texto"])
            coincide = st.radio("¿Coincide con lo que indica el manual o el registro de la planta?", ["Sí", "No"],
                                index=None, key=f"coincide_{n}_{etiqueta}", horizontal=True)
            val = {"coincide": None if coincide is None else coincide == "Sí"}
            for campo, texto in ITEMS_PARTE_B:
                val[campo] = st.radio(texto, ESCALA, index=None, key=f"{campo}_{n}_{etiqueta}", horizontal=True)
            valoraciones[etiqueta] = val

    preferida = st.radio("Entre las respuestas que viste para esta consulta, ¿cuál usarías en tu puesto?",
                         etiquetas_visibles, index=None, horizontal=True, key=f"preferida_{n}")

    completo = preferida is not None and all(v is not None for val in valoraciones.values() for v in val.values())
    if not completo:
        st.caption("Para guardar, responde todas las preguntas de cada respuesta y elige cuál preferirías.")

    if st.button("Guardar y hacer otra consulta", disabled=not completo):
        registro = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "participante": st.session_state["participante"],
            "estacion": st.session_state["estacion"],
            "pregunta": st.session_state["pregunta_pendiente"],
            "respuestas": [
                {
                    "config": r["config"],  # la etiqueta REAL -- nunca se mostró al participante
                    "texto": r["texto"],
                    "tiempo_seg": r["tiempo_seg"],
                    **valoraciones[etiqueta],
                }
                for etiqueta, r in zip(etiquetas_visibles, disponibles)
            ],
            "config_preferida": disponibles[etiquetas_visibles.index(preferida)]["config"],
        }
        _append_jsonl(LOG_INTERACCIONES, registro)
        st.session_state["resultados_pendientes"] = None
        st.session_state["pregunta_pendiente"] = None
        st.session_state["n_guardadas"] = n_guardadas + 1
        st.session_state["aviso_guardado"] = True
        st.rerun()


def tab_cuestionario_final():
    st.subheader("Cuestionario de cierre")
    st.caption("Complétalo una sola vez, al terminar de usar el sistema.")

    participante = st.session_state["participante"]
    ya_respondio_a = any(r["participante"] == participante for r in _leer_jsonl(LOG_TAM_PARTE_A))

    st.markdown("**Parte A — El sistema en general** (se responde una sola vez)")
    if ya_respondio_a:
        st.info("Ya registramos tu respuesta a la Parte A.")
    else:
        st.caption("Escala de 1 a 5: 1 = totalmente en desacuerdo, 5 = totalmente de acuerdo.")
        with st.form("parte_a"):
            respuestas_a = {}
            for codigo, texto in ITEMS_PARTE_A:
                respuestas_a[codigo] = st.radio(f"{codigo}. {texto}", ESCALA, index=None, horizontal=True, key=f"A_{codigo}")
            enviar_a = st.form_submit_button("Enviar Parte A")
        if enviar_a and any(v is None for v in respuestas_a.values()):
            st.error("Faltan ítems por responder. Responde los 9 para poder enviar la Parte A.")
        elif enviar_a:
            _append_jsonl(LOG_TAM_PARTE_A, {
                "participante": participante,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                **respuestas_a,
            })
            st.success("Parte A guardada.")
            st.rerun()

    st.divider()
    st.markdown(
        "**Parte B — resultado por configuración**\n\n"
        "Esta parte se calcula sola a partir de cómo valoraste cada respuesta durante el uso "
        "(no hace falta que la llenes aparte) -- se agrega abajo solo para que puedas revisarla."
    )
    interacciones = [i for i in _leer_jsonl(LOG_INTERACCIONES) if i["participante"] == participante]
    if not interacciones:
        st.write("Todavía no hay consultas registradas de este participante.")
        return

    por_config: dict[str, list[dict]] = {}
    for i in interacciones:
        for r in i["respuestas"]:
            por_config.setdefault(r["config"], []).append(r)
        # la preferencia de uso se cuenta aparte, por consulta
    n_consultas = len(interacciones)
    for cfg in CONFIGURACIONES:
        respuestas = por_config.get(cfg, [])
        if not respuestas:
            continue
        pct_coincide = 100 * sum(r["coincide"] for r in respuestas) / len(respuestas)
        confianza_media = sum(r["confianza"] for r in respuestas) / len(respuestas)
        transparencia_media = sum(r["transparencia"] for r in respuestas) / len(respuestas)
        tiempo_medio = sum(r["tiempo_percibido"] for r in respuestas) / len(respuestas)
        n_preferida = sum(1 for i in interacciones if i["config_preferida"] == cfg)
        st.write(
            f"**{cfg}** — coincide: {pct_coincide:.0f}% · confianza: {confianza_media:.1f}/5 · "
            f"transparencia: {transparencia_media:.1f}/5 · tiempo percibido: {tiempo_medio:.1f}/5 · "
            f"preferida en {n_preferida}/{n_consultas} consultas"
        )


def main():
    st.set_page_config(page_title="Validación del sistema — planta textil", layout="centered")

    if "participante" not in st.session_state:
        pantalla_identificacion()
        return

    st.sidebar.write(f"**Participante:** {st.session_state['participante']}")
    st.sidebar.write(f"**Estación:** {st.session_state['estacion']}")
    if st.sidebar.button("Cambiar de participante"):
        for k in ("participante", "estacion", "resultados_pendientes", "pregunta_pendiente"):
            st.session_state.pop(k, None)
        st.rerun()

    st.sidebar.divider()
    st.sidebar.caption("Estado del sistema (solo para quien coordina)")
    st.sidebar.write("Modelo Fine-Tuning (GPU): " + ("🟢 listo" if _servidor_ft_activo() else "🔴 apagado"))
    claves = {"Claude": config.ANTHROPIC_API_KEY, "ChatGPT": config.OPENAI_API_KEY, "Gemini": config.GOOGLE_API_KEY}
    st.sidebar.write("Claves de API: " + ", ".join(f"{n} {'🟢' if k else '🔴'}" for n, k in claves.items()))

    with st.spinner("Preparando el sistema (solo la primera vez, ~30 s)..."):
        _calentar()
    if st.session_state.pop("aviso_guardado", False):
        st.success("Guardado. Puedes hacer la siguiente consulta.")

    tab1, tab2 = st.tabs(["Consultas", "Cuestionario de cierre"])
    with tab1:
        tab_consultas()
    with tab2:
        tab_cuestionario_final()


if __name__ == "__main__":
    main()
