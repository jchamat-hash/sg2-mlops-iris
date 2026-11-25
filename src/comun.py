"""Utilidades compartidas por las etapas del pipeline: rutas, registro de ejecución, huellas SHA-256 y JSON.

Solo usa la biblioteca estándar de Python, porque también lo importa el validador de datos.
"""

import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Todas las rutas parten de la raíz del proyecto: los scripts funcionan igual desde cualquier carpeta
RAIZ = Path(__file__).resolve().parent.parent
CONTRATO = RAIZ / "data_contract.json"
DIR_DATOS = RAIZ / "data"
TRAIN_CSV = DIR_DATOS / "train.csv"
TEST_CSV = DIR_DATOS / "test.csv"
PREPARACION = DIR_DATOS / "preparacion.json"
DIR_REPORTES = RAIZ / "reportes"
REPORTE_VALIDACION = DIR_REPORTES / "validacion_datos.json"
ULTIMO_RUN = DIR_REPORTES / "ultimo_run.json"
RESUMEN_PIPELINE = DIR_REPORTES / "pipeline_resumen.json"
DIR_LOGS = RAIZ / "logs"
LOG_PIPELINE = DIR_LOGS / "pipeline.log"

logging.addLevelName(logging.WARNING, "AVISO")


def configurar_logs(nombre):
    """Registro doble: mensajes breves en la terminal y una huella con fecha y hora en logs/pipeline.log.

    El archivo se acumula entre ejecuciones; así queda constancia de qué validaciones pasaron y cuáles fallaron.
    """
    DIR_LOGS.mkdir(exist_ok=True)
    logger = logging.getLogger(nombre)
    if logger.handlers:  # evita mensajes duplicados si se llama más de una vez
        return logger
    logger.setLevel(logging.INFO)
    consola = logging.StreamHandler(sys.stdout)
    consola.setFormatter(logging.Formatter("%(levelname)-6s %(message)s"))
    archivo = logging.FileHandler(LOG_PIPELINE, encoding="utf-8")
    archivo.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-6s | %(name)s | %(message)s"))
    logger.addHandler(consola)
    logger.addHandler(archivo)
    logger.propagate = False
    return logger


def sha256_archivo(ruta):
    """Huella digital del archivo: si cambia un solo byte, cambia la huella."""
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(65536), b""):
            h.update(bloque)
    return h.hexdigest()


def leer_json(ruta):
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def guardar_json(datos, ruta):
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, indent=2)
        f.write("\n")


def ahora_utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
