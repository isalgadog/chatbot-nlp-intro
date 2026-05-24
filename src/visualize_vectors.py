from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "intents.csv"
OUTPUT_PATH = BASE_DIR / "models" / "intent_map_2d.png"
INTENCIONES_MOSTRADAS = [
    "sucursales_horarios",
    "mayoreo",
    "info_titulos",
]


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


def filtrar_intenciones(textos, etiquetas, intenciones_permitidas):
    textos_filtrados = []
    etiquetas_filtradas = []

    for texto, etiqueta in zip(textos, etiquetas):
        if etiqueta in intenciones_permitidas:
            textos_filtrados.append(texto)
            etiquetas_filtradas.append(etiqueta)

    return textos_filtrados, etiquetas_filtradas


def agregar_elipse(ax, puntos_intent, color, etiqueta):
    if len(puntos_intent) < 2:
        return

    centro = puntos_intent.mean(axis=0)
    cov = np.cov(puntos_intent, rowvar=False)
    valores, vectores = np.linalg.eigh(cov)
    orden = valores.argsort()[::-1]
    valores = np.maximum(valores[orden], 1e-6)
    vectores = vectores[:, orden]

    puntos_centrados = puntos_intent - centro
    puntos_rotados = puntos_centrados @ vectores
    escalas = np.sqrt(valores)
    puntos_normalizados = puntos_rotados / escalas
    radio = max(np.linalg.norm(puntos_normalizados, axis=1).max(), 1.0)

    angulo = np.degrees(np.arctan2(vectores[1, 0], vectores[0, 0]))
    ancho, alto = 2 * radio * escalas

    elipse = Ellipse(
        xy=centro,
        width=ancho,
        height=alto,
        angle=angulo,
        facecolor=color,
        edgecolor=color,
        alpha=0.12,
        lw=1.8,
        zorder=1,
    )
    ax.add_patch(elipse)
    ax.text(
        centro[0],
        centro[1],
        etiqueta,
        color=color,
        fontsize=9,
        ha="center",
        va="center",
        weight="bold",
        zorder=4,
    )


def main():
    textos, etiquetas = cargar_datos(DATA_PATH)
    textos, etiquetas = filtrar_intenciones(textos, etiquetas, INTENCIONES_MOSTRADAS)

    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
    matriz = vectorizer.fit_transform(textos)
    matriz_densa = matriz.toarray()

    reducer = PCA(n_components=2, random_state=42)
    puntos = reducer.fit_transform(matriz_densa)

    colores = {
        "nuevo_pedido": "blue",
        "cancelar_pedido": "red",
        "rastrear_pedido": "green",
        "info_titulos": "blue",
        "sucursales_horarios": "red",
        "mayoreo": "green",
    }

    fig, ax = plt.subplots(figsize=(11, 8))

    for intent in sorted(set(etiquetas)):
        puntos_intent = np.array(
            [puntos[i] for i, etiqueta in enumerate(etiquetas) if etiqueta == intent]
        )
        xs = puntos_intent[:, 0]
        ys = puntos_intent[:, 1]
        color = colores.get(intent)

        agregar_elipse(ax, puntos_intent, color, intent)
        ax.scatter(xs, ys, label=intent, alpha=0.85, s=80, color=color, zorder=3)

    for i, texto in enumerate(textos):
        ax.annotate(
            str(i + 1),
            (puntos[i, 0], puntos[i, 1]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
            zorder=5,
        )

    ax.set_title("Mapa 2D de ejemplos vectorizados por intención (3 intenciones)")
    ax.set_xlabel("Componente 1")
    ax.set_ylabel("Componente 2")
    ax.legend(title="Intención")
    ax.grid(alpha=0.2)
    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=200)
    print(f"Visualización guardada en: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
