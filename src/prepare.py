"""Etapa 1 del pipeline: genera los archivos de datos del proyecto a partir de Iris (Fisher, 1936).

Exporta las 150 flores con los nombres de columna del contrato (los mismos que usará la API), les asigna un
id estable (el número de la muestra en el conjunto original) y las divide 80/20 de forma estratificada:
40 flores de cada especie para entrenar y 10 de cada una para la prueba final.

Uso:
  python src/prepare.py
  python src/prepare.py --semilla 42 --test-size 0.2
"""

import argparse
import csv
import os
from collections import Counter

import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import comun


def parsear_argumentos():
    parser = argparse.ArgumentParser(description="Genera data/train.csv y data/test.csv a partir de Iris.")
    parser.add_argument("--semilla", type=int, default=42, help="semilla de la partición (la misma de la clase)")
    parser.add_argument("--test-size", type=float, default=0.2, help="proporción de flores para la prueba")
    return parser.parse_args()


def generar_particion(contrato, semilla=42, test_size=0.2):
    """Devuelve (columnas, filas_entrenamiento, filas_prueba) con los nombres y el orden del contrato."""
    campos = contrato["features"]["fields"]
    orden = contrato["features"]["order"]
    col_id = contrato["record_id"]["name"]
    col_objetivo = contrato["target"]["name"]

    iris = load_iris()
    # El contrato guarda el nombre original de cada columna: así el cambio a snake_case queda documentado
    posicion = {nombre: iris.feature_names.index(campos[nombre]["source_name"]) for nombre in orden}
    filas = []
    for i, (medidas, especie) in enumerate(zip(iris.data, iris.target)):
        fila = {col_id: i + 1}
        for nombre in orden:
            fila[nombre] = repr(float(medidas[posicion[nombre]]))  # "5.1": la misma precisión de la fuente
        fila[col_objetivo] = int(especie)
        filas.append(fila)

    # Estratificar mantiene la proporción de especies en ambos archivos (40/40/40 y 10/10/10)
    entrenamiento, prueba = train_test_split(filas, test_size=test_size, random_state=semilla, stratify=iris.target)
    return [col_id] + orden + [col_objetivo], entrenamiento, prueba


def escribir_csv(ruta, filas, columnas):
    # newline="" evita líneas en blanco extra en Windows; el orden por id hace el archivo fácil de revisar
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=columnas)
        escritor.writeheader()
        escritor.writerows(sorted(filas, key=lambda fila: int(fila[columnas[0]])))


def main():
    args = parsear_argumentos()
    os.chdir(comun.RAIZ)
    log = comun.configurar_logs("preparacion")
    contrato = comun.leer_json(comun.CONTRATO)
    columnas, entrenamiento, prueba = generar_particion(contrato, args.semilla, args.test_size)

    comun.DIR_DATOS.mkdir(exist_ok=True)
    escribir_csv(comun.TRAIN_CSV, entrenamiento, columnas)
    escribir_csv(comun.TEST_CSV, prueba, columnas)

    etiquetas = contrato["target"]["labels"]
    col_objetivo = contrato["target"]["name"]
    conteo = {
        nombre: {etiquetas[str(k)]: v for k, v in sorted(Counter(f[col_objetivo] for f in grupo).items())}
        for nombre, grupo in (("train", entrenamiento), ("test", prueba))
    }
    campos = contrato["features"]["fields"]
    comun.guardar_json({
        "fecha_utc": comun.ahora_utc(),
        "fuente": f"sklearn.datasets.load_iris (scikit-learn {sklearn.__version__})",
        "dataset": "Iris (Fisher, 1936)",
        "filas_totales": len(entrenamiento) + len(prueba),
        "test_size": args.test_size,
        "estratificado": True,
        "semilla": args.semilla,
        "filas_train": len(entrenamiento),
        "filas_test": len(prueba),
        "conteo_por_especie": conteo,
        "sha256": {"train": comun.sha256_archivo(comun.TRAIN_CSV), "test": comun.sha256_archivo(comun.TEST_CSV)},
        "columnas_originales": {nombre: campos[nombre]["source_name"] for nombre in contrato["features"]["order"]},
    }, comun.PREPARACION)

    log.info("Fuente: Iris de scikit-learn %s (150 flores, 3 especies)", sklearn.__version__)
    log.info("data/train.csv: %d flores %s", len(entrenamiento), conteo["train"])
    log.info("data/test.csv:  %d flores %s", len(prueba), conteo["test"])
    log.info("Partición estratificada 80/20 con semilla %d; metadatos en data/preparacion.json", args.semilla)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
