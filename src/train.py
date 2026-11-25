"""Etapa 3 del pipeline: entrena el Random Forest con los datos validados y lo registra en MLflow.

Evoluciona el train.py de la actividad 1.1, que a su vez parte del de la sesión 2 de Seminario de Grado II.
Cambios frente a la versión de la 1.1:
  - lee data/train.csv y data/test.csv, que genera prepare.py, en lugar de partir load_iris() aquí;
  - se niega a entrenar si esos archivos no tienen una validación aprobada vigente (misma huella SHA-256);
  - toma del contrato el orden y los nombres de las variables, los mismos que recibirá la API;
  - registra como artefactos el contrato, el reporte de validación y los metadatos de la preparación;
  - deja el RUN_ID en reportes/ultimo_run.json para las pruebas automáticas.
La configuración por defecto (100 árboles, profundidad 3) es la que se seleccionó en la actividad 1.1.

Ejemplos:
  python src/train.py
  python src/train.py --run-name Prueba_depth5 --max-depth 5 --hipotesis "Resultado esperado y por qué"
"""

import argparse
import os
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone

# MLflow 3.18 imprime un aviso informativo en cada ejecución; no afecta los resultados
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

import comun

AUTOR = "Jonathan Chamat"     # queda en la etiqueta "autor" de cada run
CV_FOLDS, CV_REPEATS = 5, 10  # 5 particiones repetidas 10 veces: 50 evaluaciones por run, como en la 1.1


class DatosNoValidados(Exception):
    """Los archivos de datos no tienen una validación aprobada que corresponda a su contenido actual."""


def leer_profundidad(valor):
    """Convierte el texto de la terminal en profundidad: "None" significa árboles sin límite."""
    return None if valor == "None" else int(valor)


def parsear_argumentos():
    parser = argparse.ArgumentParser(description="Entrena el Random Forest con los datos validados y lo registra en MLflow.")
    parser.add_argument("--run-name", default=f"Pipeline_{datetime.now():%Y%m%d_%H%M%S}", help="nombre del run en MLflow")
    parser.add_argument("--n-estimators", type=int, default=100, help="número de árboles del bosque")
    parser.add_argument("--max-depth", type=leer_profundidad, default=3,
                        help="profundidad máxima de cada árbol (None = sin límite)")
    parser.add_argument("--model-seed", type=int, default=42,
                        help="semilla del bosque (muestras bootstrap y selección de variables)")
    parser.add_argument("--hipotesis", default="", help="resultado esperado; se guarda como descripción del run")
    return parser.parse_args()


def verificar_validacion():
    """Solo se entrena con archivos que pasaron la validación, byte a byte (poka-yoke del pipeline).

    Si alguien edita train.csv o el contrato después de validar, las huellas ya no coinciden y no se entrena.
    """
    if not comun.REPORTE_VALIDACION.exists():
        raise DatosNoValidados("no existe reportes/validacion_datos.json")
    reporte = comun.leer_json(comun.REPORTE_VALIDACION)
    if reporte.get("resultado") != "APROBADO":
        raise DatosNoValidados("la última validación rechazó los datos")
    actuales = {"train": comun.TRAIN_CSV, "test": comun.TEST_CSV}
    for rol, ruta in actuales.items():
        if not ruta.exists() or comun.sha256_archivo(ruta) != reporte["archivos"][rol]["sha256"]:
            raise DatosNoValidados(f"{ruta.name} cambió después de validarse")
    if comun.sha256_archivo(comun.CONTRATO) != reporte["contrato"]["sha256"]:
        raise DatosNoValidados("el contrato cambió después de la validación")
    return reporte


def cargar(ruta, contrato):
    """Lee un CSV ya validado y devuelve X con las columnas en el orden del contrato, y la especie."""
    orden = contrato["features"]["order"]
    objetivo = contrato["target"]["name"]
    tipos = {col: "float64" for col in orden} | {objetivo: "int64", contrato["record_id"]["name"]: "int64"}
    datos = pd.read_csv(ruta, dtype=tipos)
    return datos[orden], datos[objetivo]


def commit_git():
    """Versión del código con la que se entrenó; '+cambios' si había archivos sin confirmar."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=comun.RAIZ,
                                capture_output=True, text=True, check=True).stdout.strip()
        cambios = subprocess.run(["git", "status", "--porcelain"], cwd=comun.RAIZ,
                                 capture_output=True, text=True, check=True).stdout.strip()
        return commit + ("+cambios" if cambios else "")
    except (OSError, subprocess.CalledProcessError):
        return "sin_git"


def main():
    args = parsear_argumentos()
    os.chdir(comun.RAIZ)  # rutas relativas de MLflow (mlflow.db, mlruns/) siempre en la raíz del proyecto
    log = comun.configurar_logs("entrenamiento")
    contrato = comun.leer_json(comun.CONTRATO)

    try:
        reporte = verificar_validacion()
    except DatosNoValidados as exc:
        log.error("No se entrena: %s. Primero se ejecuta la validación (src/validate_data.py).", exc)
        return 1
    preparacion = comun.leer_json(comun.PREPARACION)
    X_train, y_train = cargar(comun.TRAIN_CSV, contrato)
    X_test, y_test = cargar(comun.TEST_CSV, contrato)
    log.info("Datos validados: %d flores de entrenamiento y %d de prueba (contrato v%s)",
             len(X_train), len(X_test), reporte["contrato"]["version"])

    # La variable de entorno permite usar otro servidor de MLflow sin editar el código (actividad 3.1)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    experimento = os.getenv("MLFLOW_EXPERIMENT", contrato["model"]["experiment"])
    mlflow.set_experiment(experimento)

    def crear_modelo(oob=False):
        """Un solo lugar para la configuración: el modelo final y la validación cruzada usan la misma."""
        return RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth,
                                      random_state=args.model_seed, oob_score=oob)

    with mlflow.start_run(run_name=args.run_name, description=args.hipotesis or None) as run:
        mlflow.log_params({
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "model_seed": args.model_seed,
            "split_seed": preparacion["semilla"],
            "test_size": preparacion["test_size"],
            "split_estratificado": preparacion["estratificado"],
            "cv_folds": CV_FOLDS,
            "cv_repeats": CV_REPEATS,
            "filas_train": len(X_train),
            "filas_test": len(X_test),
        })
        mlflow.set_tags({
            "autor": AUTOR,
            "dataset": "Iris (Fisher, 1936), exportado con sklearn.datasets.load_iris",
            "tipo": "pipeline",
            "actividad": "2.1",
            "comando": shlex.join(["python"] + sys.argv),
            "python_version": platform.python_version(),
            "sklearn_version": sklearn.__version__,
            "mlflow_version": mlflow.__version__,
            "sistema": platform.platform(),
            "fecha_entrenamiento": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": commit_git(),
            "contrato": f"{contrato['contract']['name']} v{contrato['contract']['version']}",
            "validacion_datos": reporte["resultado"],
            "sha256_train": reporte["archivos"]["train"]["sha256"],
            "sha256_test": reporte["archivos"]["test"]["sha256"],
        })

        # Modelo final: se entrena con las 120 flores y se evalúa una sola vez con las 30 de prueba
        modelo = crear_modelo(oob=True).fit(X_train, y_train)
        accuracy = modelo.score(X_test, y_test)
        train_accuracy = modelo.score(X_train, y_train)

        # La validación cruzada usa solo el entrenamiento; así la prueba no interviene en la comparación
        cv = cross_validate(
            crear_modelo(), X_train, y_train,
            cv=RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=CV_REPEATS, random_state=preparacion["semilla"]),
            scoring=["accuracy", "neg_log_loss"],
        )
        cv_mean = cv["test_accuracy"].mean()
        cv_std = cv["test_accuracy"].std()
        overfit_gap = train_accuracy - cv_mean  # cuánto acierta de más en las flores que ya conoce

        metricas = {
            "accuracy": accuracy,
            "f1_macro": f1_score(y_test, modelo.predict(X_test), average="macro"),
            "train_accuracy": train_accuracy,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "cv_log_loss": -cv["test_neg_log_loss"].mean(),
            "oob_accuracy": modelo.oob_score_,
            "overfit_gap": overfit_gap,
            "mean_depth": float(np.mean([arbol.get_depth() for arbol in modelo.estimators_])),
            "mean_leaves": float(np.mean([arbol.get_n_leaves() for arbol in modelo.estimators_])),
        }
        mlflow.log_metrics(metricas)

        # La firma lleva los nombres del contrato: el modelo documenta qué recibe y qué devuelve
        mlflow.sklearn.log_model(
            modelo,
            name=contrato["model"]["name"],
            signature=infer_signature(X_test, modelo.predict(X_test)),
            input_example=X_test.head(2),
            skops_trusted_types=["sklearn.tree._tree.Tree"],  # MLflow 3 guarda con skops y exige declarar este tipo
        )
        # Evidencias de trazabilidad: con qué entorno, contrato y datos validados se entrenó este modelo
        mlflow.log_artifact(str(comun.RAIZ / "requirements.txt"), artifact_path="entorno")
        mlflow.log_artifact(str(comun.CONTRATO), artifact_path="contrato")
        mlflow.log_artifact(str(comun.REPORTE_VALIDACION), artifact_path="calidad")
        mlflow.log_artifact(str(comun.PREPARACION), artifact_path="datos")

    comun.guardar_json({
        "run_id": run.info.run_id,
        "run_name": args.run_name,
        "experimento": experimento,
        "modelo": contrato["model"]["name"],
        "fecha_utc": comun.ahora_utc(),
        "metricas": {k: round(float(v), 6) for k, v in metricas.items()},
    }, comun.ULTIMO_RUN)

    log.info("Run: %s | Run ID: %s | experimento: %s", args.run_name, run.info.run_id, experimento)
    log.info("Exactitud en prueba: %.3f | entrenamiento: %.3f | validación cruzada: %.3f ± %.3f | brecha: %.3f",
             accuracy, train_accuracy, cv_mean, cv_std, overfit_gap)
    return 0


if __name__ == "__main__":
    sys.exit(main())
