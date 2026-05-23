from pathlib import Path
import csv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "intents.csv"


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
            ("vectorizer", TfidfVectorizer(lowercase=True, ngram_range=(1, 2))),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )


def main():
    textos, etiquetas = cargar_datos(DATA_PATH)

    x_train, x_test, y_train, y_test = train_test_split(
        textos,
        etiquetas,
        test_size=0.25,
        random_state=42,
        stratify=etiquetas,
    )

    modelo = crear_pipeline()
    modelo.fit(x_train, y_train)
    predicciones = modelo.predict(x_test)

    accuracy = accuracy_score(y_test, predicciones)

    print("Evaluación del modelo")
    print("=" * 40)
    print(f"Ejemplos totales: {len(textos)}")
    print(f"Entrenamiento: {len(x_train)}")
    print(f"Prueba: {len(x_test)}")
    print(f"Accuracy: {accuracy:.2f}\n")
    print("Reporte de clasificación:")
    print(classification_report(y_test, predicciones, zero_division=0))


if __name__ == "__main__":
    main()
