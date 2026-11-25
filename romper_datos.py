"""Sabotea los datos preparados para comprobar qué etapa del pipeline detiene cada problema.

Reproduce los siete sabotajes del juego «¡Rompe el pipeline!» de la sesión 4 de Seminario de Grado II.

Uso:
  python romper_datos.py 3          # aplica el sabotaje 3 a los CSV de data/
  python run_pipeline.py --sin-preparar
  python romper_datos.py --todos    # corre los siete de punta a punta y resume qué etapa detuvo cada uno

Con --todos, los runs que llegan a entrenar quedan en el experimento "Sabotajes_pipeline" de MLflow y,
al final, los datos y los reportes vuelven a su estado limpio.
"""

import argparse
import csv
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import comun  # noqa: E402

MEDIDAS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def leer(ruta):
    with open(ruta, newline="", encoding="utf-8") as f:
        lector = csv.DictReader(f)
        return lector.fieldnames, list(lector)


def escribir(ruta, columnas, filas):
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=columnas)
        escritor.writeheader()
        escritor.writerows(filas)


# Cada sabotaje recibe las filas de entrenamiento y de prueba (listas de dict) y las modifica en el lugar
def texto_en_lugar_de_numero(train, test):
    train[1]["sepal_length"] = "cinco"


def celda_vacia(train, test):
    train[10]["petal_width"] = ""


def registro_en_milimetros(train, test):
    for col in MEDIDAS:
        train[60][col] = repr(round(float(train[60][col]) * 10, 1))


def especie_inexistente(train, test):
    train[100]["species"] = "3"


def sin_virginica(train, test):
    train[:] = [fila for fila in train if fila["species"] != "2"]


def etiquetas_erradas(train, test):
    azar = random.Random(42)  # con semilla: el sabotaje es el mismo en cada ejecución
    for fila in azar.sample(train, k=round(0.4 * len(train))):
        fila["species"] = str(azar.choice([c for c in (0, 1, 2) if c != int(fila["species"])]))


def prueba_copiada_en_entrenamiento(train, test):
    train.extend(dict(fila) for fila in test)


# número: (descripción, función, etapa que lo detuvo en el juego de la sesión 4)
ESCENARIOS = {
    1: ("«cinco» en lugar de un número", texto_en_lugar_de_numero, "Validación"),
    2: ("Celda vacía en petal_width", celda_vacia, "Validación"),
    3: ("Registro digitado en milímetros", registro_en_milimetros, "Validación"),
    4: ("Especie con código 3, que no existe", especie_inexistente, "Validación"),
    5: ("Faltan todas las flores virginica", sin_virginica, "Pruebas"),
    6: ("40 % de las etiquetas mal asignadas", etiquetas_erradas, "Pruebas"),
    7: ("Las flores de prueba copiadas en entrenamiento", prueba_copiada_en_entrenamiento, "Nadie"),
}
NOMBRE_ETAPA = {1: "1 · Preparación", 2: "2 · Validación", 3: "3 · Entrenamiento", 4: "4 · Pruebas"}


def aplicar(numero, ruta_train=comun.TRAIN_CSV, ruta_test=comun.TEST_CSV):
    columnas, train = leer(ruta_train)
    _, test = leer(ruta_test)
    ESCENARIOS[numero][1](train, test)
    escribir(ruta_train, columnas, train)


def evidencia(salida):
    """Extrae de la salida del pipeline la línea que explica por qué se detuvo."""
    lineas = salida.splitlines()
    fallidas = [l.split("::")[-1].split(" ")[0] for l in lineas if l.startswith("FAILED ")]
    if fallidas:
        texto = "1 prueba falló" if len(fallidas) == 1 else f"{len(fallidas)} pruebas fallaron"
        return f"{texto}: {', '.join(fallidas[:3])}" + (" ..." if len(fallidas) > 3 else "")
    errores = [l.split("- ", 1)[1].strip() for l in lineas if l.startswith("ERROR") and "  - " in l]
    if errores:
        return errores[0] + (f" (y {len(errores) - 1} más)" if len(errores) > 1 else "")
    return "sin errores"


def ejecutar_todos():
    py = sys.executable
    os.chdir(comun.RAIZ)
    respaldo = {ruta: ruta.read_bytes() for ruta in (comun.REPORTE_VALIDACION, comun.ULTIMO_RUN, comun.RESUMEN_PIPELINE)
                if ruta.exists()}
    entorno = dict(os.environ, MLFLOW_EXPERIMENT="Sabotajes_pipeline", PYTHONUTF8="1", MLFLOW_DISABLE_AGENT_HINT="1")
    tabla = []
    try:
        for numero, (descripcion, _, etapa_sesion) in ESCENARIOS.items():
            subprocess.run([py, "src/prepare.py"], check=True, capture_output=True, env=entorno)
            aplicar(numero)
            proceso = subprocess.run([py, "run_pipeline.py", "--sin-preparar", "--run-name", f"Sabotaje_{numero}"],
                                     capture_output=True, text=True, encoding="utf-8", env=entorno)
            resumen = comun.leer_json(comun.RESUMEN_PIPELINE)
            etapa = NOMBRE_ETAPA.get(resumen["etapa_fallida"], "Ninguna: terminó bien")
            tabla.append((numero, descripcion, etapa_sesion, etapa, evidencia(proceso.stdout)))
            print(f"Sabotaje {numero} ({descripcion}): lo detuvo la etapa {etapa}", flush=True)
    finally:
        # Datos limpios y reportes del último pipeline normal, como estaban antes
        subprocess.run([py, "src/prepare.py"], capture_output=True, env=entorno)
        for ruta, contenido in respaldo.items():
            ruta.write_bytes(contenido)

    lineas = ["# Resultado de los siete sabotajes", "",
              f"Generado con `python romper_datos.py --todos` el {comun.ahora_utc()} (UTC).", "",
              "| # | Sabotaje | Lo detuvo en la sesión 4 | Lo detiene este pipeline | Evidencia |",
              "|---|---|---|---|---|"]
    lineas += [f"| {n} | {d} | {s} | {e} | {ev} |" for n, d, s, e, ev in tabla]
    comun.DIR_REPORTES.mkdir(exist_ok=True)
    (comun.DIR_REPORTES / "sabotajes.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lineas[4:]) + "\n\nTabla guardada en reportes/sabotajes.md")
    detenidos = sum(1 for fila in tabla if not fila[3].startswith("Ninguna"))
    return 0 if detenidos == len(ESCENARIOS) else 1


def main():
    parser = argparse.ArgumentParser(description="Sabotea los datos para probar el pipeline.")
    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument("escenario", nargs="?", type=int, choices=sorted(ESCENARIOS), help="número de sabotaje (1 a 7)")
    grupo.add_argument("--todos", action="store_true", help="ejecuta los siete sabotajes de punta a punta")
    args = parser.parse_args()
    if args.todos:
        return ejecutar_todos()
    aplicar(args.escenario)
    print(f"Sabotaje {args.escenario} aplicado: {ESCENARIOS[args.escenario][0]}.\n"
          "Siguiente paso: python run_pipeline.py --sin-preparar")
    return 0


if __name__ == "__main__":
    sys.exit(main())
