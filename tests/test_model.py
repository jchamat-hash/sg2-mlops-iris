"""Pruebas del modelo recién entrenado (etapa 4 del pipeline).

Verifican que el run exista y use datos validados, que el modelo se cargue por su RUN_ID (como lo hará la
API de la actividad 3.1), que respete el contrato y que su desempeño supere los umbrales de aprobación.
"""

import os

import pytest

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.models import get_model_info  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

import comun  # noqa: E402

# Umbrales de aprobación del modelo
EXACTITUD_MINIMA = 0.80  # el de la sesión 4: por debajo, el modelo no sirve para clasificar
VC_MINIMA = 0.90         # en la actividad 1.1 todas las configuraciones superaron 0,91 en validación cruzada
BRECHA_MAXIMA = 0.10     # en la 1.1 la brecha más alta fue 0,058; por encima de 0,10 el modelo memoriza

CONTRATO = comun.leer_json(comun.CONTRATO)
ORDEN = CONTRATO["features"]["order"]
ETIQUETAS = CONTRATO["target"]["labels"]
NOMBRE_MODELO = CONTRATO["model"]["name"]

# Una flor conocida por especie (ids 1, 79 y 146 del conjunto original). Si el orden de las columnas
# cambiara entre entrenamiento e inferencia (training-serving skew), aquí fallaría la clasificación.
CASOS_REFERENCIA = [
    ({"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}, "setosa"),
    ({"sepal_length": 6.0, "sepal_width": 2.9, "petal_length": 4.5, "petal_width": 1.5}, "versicolor"),
    ({"sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.2, "petal_width": 2.3}, "virginica"),
]


def tabla(*entradas):
    """Arma la entrada del modelo con las columnas en el orden del contrato, como lo hará la API."""
    return pd.DataFrame([[entrada[c] for c in ORDEN] for entrada in entradas], columns=ORDEN)


@pytest.fixture(scope="module")
def run_info():
    assert comun.ULTIMO_RUN.exists(), "no hay un run que probar: primero se ejecuta run_pipeline.py"
    return comun.leer_json(comun.ULTIMO_RUN)


@pytest.fixture(scope="module")
def cliente():
    uri = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{(comun.RAIZ / 'mlflow.db').as_posix()}")
    mlflow.set_tracking_uri(uri)
    return MlflowClient()


@pytest.fixture(scope="module")
def run(cliente, run_info):
    return cliente.get_run(run_info["run_id"])


@pytest.fixture(scope="module")
def modelo(cliente, run_info):
    return mlflow.sklearn.load_model(f"runs:/{run_info['run_id']}/{NOMBRE_MODELO}")


@pytest.fixture(scope="module")
def prueba():
    datos = pd.read_csv(comun.TEST_CSV)
    return datos[ORDEN], datos[CONTRATO["target"]["name"]]


# ---------------------------------------------------------------- trazabilidad

def test_el_run_termino_correctamente(run):
    assert run.info.status == "FINISHED"


def test_el_modelo_se_entreno_con_los_datos_validados(run):
    etiquetas = run.data.tags
    assert etiquetas["validacion_datos"] == "APROBADO"
    assert etiquetas["sha256_train"] == comun.sha256_archivo(comun.TRAIN_CSV)
    assert etiquetas["sha256_test"] == comun.sha256_archivo(comun.TEST_CSV)


def test_el_run_guarda_contrato_validacion_datos_y_entorno(cliente, run_info):
    esperados = {"contrato/data_contract.json", "calidad/validacion_datos.json",
                 "datos/preparacion.json", "entorno/requirements.txt"}
    guardados = {a.path for carpeta in ("contrato", "calidad", "datos", "entorno")
                 for a in cliente.list_artifacts(run_info["run_id"], carpeta)}
    assert esperados <= guardados


# ---------------------------------------------------------------- coherencia con el contrato

def test_el_modelo_se_carga_por_su_run_id(modelo):
    assert hasattr(modelo, "predict")


def test_el_modelo_usa_las_variables_y_el_orden_del_contrato(modelo):
    assert list(modelo.feature_names_in_) == ORDEN


def test_la_firma_registrada_coincide_con_el_contrato(run_info):
    firma = get_model_info(f"runs:/{run_info['run_id']}/{NOMBRE_MODELO}").signature
    assert firma.inputs.input_names() == ORDEN


def test_las_predicciones_estan_en_el_dominio_del_contrato(modelo, prueba):
    X, _ = prueba
    assert set(modelo.predict(X)) <= set(CONTRATO["target"]["allowed_values"])


def test_el_ejemplo_valido_del_contrato_da_la_salida_documentada(modelo):
    esperado = CONTRATO["examples"]["valid_output"]
    clase = int(modelo.predict(tabla(CONTRATO["examples"]["valid_input"]))[0])
    assert clase == esperado["class_id"] and ETIQUETAS[str(clase)] == esperado["prediction"]


@pytest.mark.parametrize("entrada, especie", CASOS_REFERENCIA, ids=[e for _, e in CASOS_REFERENCIA])
def test_caso_de_referencia(modelo, entrada, especie):
    assert ETIQUETAS[str(int(modelo.predict(tabla(entrada))[0]))] == especie


# ---------------------------------------------------------------- desempeño

def test_exactitud_minima_en_prueba(modelo, prueba):
    X, y = prueba
    assert modelo.score(X, y) >= EXACTITUD_MINIMA


def test_la_exactitud_registrada_es_la_del_modelo(modelo, prueba, run):
    X, y = prueba
    assert modelo.score(X, y) == pytest.approx(run.data.metrics["accuracy"])


def test_validacion_cruzada_minima(run):
    assert run.data.metrics["cv_mean"] >= VC_MINIMA


def test_brecha_maxima_entre_entrenamiento_y_validacion_cruzada(run):
    assert run.data.metrics["overfit_gap"] <= BRECHA_MAXIMA
