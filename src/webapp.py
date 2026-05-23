from pathlib import Path

import joblib
import streamlit as st

from responses import RESPUESTAS


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "intent_classifier.joblib"
UMBRAL_CONFIANZA = 0.25


@st.cache_resource
def cargar_modelo():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "No se encontró el modelo entrenado. Primero ejecuta: python src/train.py"
        )
    return joblib.load(MODEL_PATH)


def predecir_intencion(modelo, texto: str):
    probabilidades = modelo.predict_proba([texto])[0]
    clases = modelo.classes_

    mejor_indice = probabilidades.argmax()
    intent = clases[mejor_indice]
    confianza = float(probabilidades[mejor_indice])

    ranking = sorted(
        zip(clases, probabilidades),
        key=lambda item: item[1],
        reverse=True,
    )

    return intent, confianza, ranking


def obtener_respuesta(intent: str, confianza: float):
    if confianza < UMBRAL_CONFIANZA:
        return RESPUESTAS["fallback"]
    return RESPUESTAS.get(intent, RESPUESTAS["fallback"])


def main():
    st.set_page_config(page_title="Chatbot NLP de librería", page_icon="📚")
    st.title("📚 Chatbot NLP de librería")
    st.write(
        "Esta mini app demuestra la misma lógica del chatbot de consola: clasificación de intención + respuesta predefinida."
    )

    modelo = cargar_modelo()
    texto = st.text_input("Escribe un mensaje", placeholder="Ejemplo: quiero saber dónde va mi pedido")

    if st.button("Analizar") and texto.strip():
        intent, confianza, ranking = predecir_intencion(modelo, texto)
        respuesta = obtener_respuesta(intent, confianza)

        st.subheader("Resultado")
        st.write(f"**Intención detectada:** `{intent}`")
        st.write(f"**Confianza estimada:** {confianza:.2f}")
        st.write(f"**Respuesta del bot:** {respuesta}")

        st.subheader("Top 3 intenciones")
        for etiqueta, probabilidad in ranking[:3]:
            st.write(f"- {etiqueta}: {probabilidad:.2f}")


if __name__ == "__main__":
    main()
