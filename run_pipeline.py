"""Orquestador del pipeline: un solo comando ejecuta las cuatro etapas en orden y se detiene en la primera que falle.

  1. Preparación de datos             src/prepare.py         -> data/train.csv y data/test.csv
  2. Validación y control de calidad  src/validate_data.py   -> aprobado o detenido
  3. Entrenamiento y registro         src/train.py           -> run en MLflow (modelo, métricas, trazabilidad)
  4. Pruebas automatizadas            pytest (tests/)        -> validador y modelo probados

Cada etapa corre en su propio proceso. Si termina con un código distinto de 0, el pipeline se detiene
(principio de fallar rápido): nunca se entrena con datos rechazados ni se aprueba un modelo que no pasó las pruebas.

Uso:
  python run_pipeline.py                     # pipeline completo
  python run_pipeline.py --sin-preparar      # usa los CSV existentes (para probar datos saboteados)
  python run_pipeline.py --max-depth 5 --run-name Prueba_depth5
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import comun  # noqa: E402

LINEA = "=" * 64


def parsear_argumentos():
    parser = argparse.ArgumentParser(description="Ejecuta el pipeline completo con un solo comando.")
    parser.add_argument("--sin-preparar", action="store_true",
                        help="omite la etapa 1 y valida los CSV tal como están")
    parser.add_argument("--run-name", default=f"Pipeline_{datetime.now():%Y%m%d_%H%M%S}")
    parser.add_argument("--n-estimators", default="100")
    parser.add_argument("--max-depth", default="3", help="profundidad máxima (None = sin límite)")
    parser.add_argument("--hipotesis", default=(
        "Configuración seleccionada en la actividad 1.1 (Run_3_depth3: 100 árboles, profundidad 3), "
        "entrenada con la partición estratificada y los datos validados del pipeline. "
        "Se espera una validación cruzada cercana a 0,95 y una brecha menor que 0,05."))
    return parser.parse_args()


def registrar_veredicto(aprobado):
    """Deja en el run de MLflow si el modelo pasó las pruebas: solo un modelo aprobado debería desplegarse."""
    if not comun.ULTIMO_RUN.exists():
        return None
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow  # se importa aquí: las etapas 1 y 2 no necesitan MLflow
    from mlflow.tracking import MlflowClient

    run = comun.leer_json(comun.ULTIMO_RUN)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    MlflowClient().set_tag(run["run_id"], "pruebas_modelo", "aprobado" if aprobado else "rechazado")
    return run["run_id"]


def main():
    args = parsear_argumentos()
    os.chdir(comun.RAIZ)
    py = sys.executable
    etapas = [
        (1, "Preparación de datos", [py, "src/prepare.py"]),
        (2, "Validación y control de calidad", [py, "src/validate_data.py"]),
        (3, "Entrenamiento y registro con MLflow", [py, "src/train.py", "--run-name", args.run_name,
                                                    "--n-estimators", args.n_estimators,
                                                    "--max-depth", args.max_depth, "--hipotesis", args.hipotesis]),
        (4, "Pruebas automatizadas", [py, "-m", "pytest"]),
    ]
    if args.sin_preparar:
        etapas = etapas[1:]

    # Las etapas hijas escriben en UTF-8 aunque la terminal de Windows use otra codificación
    entorno = dict(os.environ, PYTHONUTF8="1", MLFLOW_DISABLE_AGENT_HINT="1")
    comun.ULTIMO_RUN.unlink(missing_ok=True)  # el RUN_ID que se pruebe debe ser el de esta ejecución
    resumen = {"inicio": comun.ahora_utc(), "run_name": args.run_name, "sin_preparar": args.sin_preparar,
               "etapas": [], "resultado": "COMPLETADO", "etapa_fallida": None, "run_id": None}
    log = comun.configurar_logs("pipeline")
    log.info("Inicio del pipeline (%s)%s", args.run_name, " sin la etapa de preparación" if args.sin_preparar else "")
    inicio_total = time.perf_counter()

    for numero, nombre, comando in etapas:
        print(f"\n{LINEA}\nETAPA {numero}/4 · {nombre}\n{LINEA}", flush=True)
        inicio = time.perf_counter()
        codigo = subprocess.run(comando, env=entorno).returncode
        segundos = round(time.perf_counter() - inicio, 1)
        resumen["etapas"].append({"etapa": numero, "nombre": nombre, "codigo_salida": codigo, "segundos": segundos})
        if codigo != 0:
            resumen.update(resultado="DETENIDO", etapa_fallida=numero)
            if numero == 4:
                resumen["run_id"] = registrar_veredicto(aprobado=False)
            resumen["fin"] = comun.ahora_utc()
            comun.guardar_json(resumen, comun.RESUMEN_PIPELINE)
            consecuencia = {1: "no hay datos preparados", 2: "no se entrenó ningún modelo",
                            3: "el entrenamiento no terminó", 4: "el modelo quedó marcado como rechazado en MLflow"}
            print(f"\n{LINEA}\nERROR EN LA ETAPA {numero} ({nombre}): código de salida {codigo}.\n"
                  f"El pipeline se detuvo: {consecuencia[numero]}.\n{LINEA}", flush=True)
            log.error("Pipeline detenido en la etapa %d (%s), código %d", numero, nombre, codigo)
            return codigo

    resumen["run_id"] = registrar_veredicto(aprobado=True)
    resumen["fin"] = comun.ahora_utc()
    comun.guardar_json(resumen, comun.RESUMEN_PIPELINE)
    duracion = round(time.perf_counter() - inicio_total, 1)
    tiempos = " | ".join(f"{e['etapa']}: {e['segundos']} s" for e in resumen["etapas"])
    print(f"\n{LINEA}\nPIPELINE FINALIZADO CORRECTAMENTE en {duracion} s ({tiempos})\n"
          f"RUN_ID del modelo aprobado: {resumen['run_id']}\n"
          f"Para verlo en MLflow UI: make ui (macOS/Linux) o, con el entorno activado,\n"
          f"mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001 --workers 1\n{LINEA}", flush=True)
    log.info("Pipeline completado en %s s; RUN_ID %s", duracion, resumen["run_id"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
