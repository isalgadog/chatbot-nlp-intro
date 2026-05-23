from pathlib import Path
import csv

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "intents.csv"
MODEL_PATH = BASE_DIR / "models" / "intent_classifier.joblib"


def cargar_datos(data_path: Path):
    textos = []
    etiquetas = []

    with data_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            texto = row["texto"].strip()
            intent = row["intent"].strip()

            if texto and intent:
                textos.append(texto)
                etiquetas.append(intent)

    return textos, etiquetas


def crear_pipeline():
    return Pipeline(
        steps=[
            (
                "vectorizer",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                ),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=1000),
            ),
        ]
    )


def main():
    textos, etiquetas = cargar_datos(DATA_PATH)

    if not textos:
        raise ValueError("No se encontraron datos de entrenamiento.")

    pipeline = crear_pipeline()
    pipeline.fit(textos, etiquetas)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    print("Modelo entrenado correctamente.")
    print(f"Total de ejemplos: {len(textos)}")
    print(f"Modelo guardado en: {MODEL_PATH}")


if __name__ == "__main__":
    main()
