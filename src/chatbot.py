from pathlib import Path

import joblib

from responses import RESPUESTAS


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "intent_classifier.joblib"
UMBRAL_CONFIANZA = 0.25


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
    modelo = cargar_modelo()

    print("Chatbot de librería listo.")
    print("Escribe 'salir' para terminar.\n")

    while True:
        texto = input("Tú: ").strip()

        if not texto:
            print("Bot: Escribe un mensaje para poder ayudarte.\n")
            continue

        if texto.lower() in {"salir", "exit", "quit"}:
            print("Bot: Hasta luego.")
            break

        intent, confianza, ranking = predecir_intencion(modelo, texto)
        respuesta = obtener_respuesta(intent, confianza)

        print(f"Intención detectada: {intent}")
        print(f"Confianza estimada: {confianza:.2f}")
        print("Top 3 intenciones:")
        for etiqueta, probabilidad in ranking[:3]:
            print(f"- {etiqueta}: {probabilidad:.2f}")
        print(f"Bot: {respuesta}\n")


if __name__ == "__main__":
    main()
