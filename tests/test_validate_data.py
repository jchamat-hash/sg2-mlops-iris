"""Pruebas del contrato y del validador de datos (etapa 4 del pipeline).

Comprueban que cada tipo de dato malo se detecte, que el validador no dependa de librerías externas y que
los sabotajes de la sesión 4 que son problemas de datos se detengan en la validación.
"""

import ast
import csv
import math
import sys

import pytest

import comun
import prepare
import romper_datos
import validate_data as vd

CONTRATO = comun.leer_json(comun.CONTRATO)
VALIDA = CONTRATO["examples"]["valid_input"]
COLUMNAS = [CONTRATO["record_id"]["name"]] + CONTRATO["features"]["order"] + [CONTRATO["target"]["name"]]
FILAS_BASE = [[1, "5.1", "3.5", "1.4", "0.2", 0], [51, "7.0", "3.2", "4.7", "1.4", 1], [101, "6.3", "3.3", "6.0", "2.5", 2]]


def con(**cambios):
    """Copia del ejemplo válido del contrato con algunos campos cambiados."""
    entrada = dict(VALIDA)
    entrada.update(cambios)
    return entrada


def categorias(errores):
    return {categoria for categoria, _ in errores}


def escribir(ruta, filas, columnas=COLUMNAS):
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        escritor.writerow(columnas)
        escritor.writerows(filas)
    return ruta


# ---------------------------------------------------------------- requisito de la actividad

def test_el_validador_solo_usa_la_biblioteca_estandar():
    for archivo in ("validate_data.py", "comun.py"):
        arbol = ast.parse((comun.RAIZ / "src" / archivo).read_text(encoding="utf-8"))
        modulos = {a.name.split(".")[0] for n in ast.walk(arbol) if isinstance(n, ast.Import) for a in n.names}
        modulos |= {n.module.split(".")[0] for n in ast.walk(arbol) if isinstance(n, ast.ImportFrom) and n.module}
        externos = modulos - set(sys.stdlib_module_names) - {"comun"}
        assert not externos, f"{archivo} importa librerías externas: {externos}"


# ---------------------------------------------------------------- contrato

def test_el_contrato_es_coherente():
    assert vd.verificar_contrato(CONTRATO) == []


def test_el_ejemplo_valido_pasa_sin_advertencias():
    assert vd.validar_entrada(VALIDA, CONTRATO) == ([], [])


def test_el_ejemplo_invalido_falla_por_cada_razon_documentada():
    problemas, errores = vd.autoverificar_contrato(CONTRATO)
    assert problemas == []
    assert len(errores) == len(CONTRATO["examples"]["invalid_input_reasons"])


def test_un_contrato_contradictorio_se_detecta():
    roto = comun.leer_json(comun.CONTRATO)
    roto["features"]["fields"]["sepal_length"]["minimum"] = 20.0
    assert any("sepal_length" in p for p in vd.verificar_contrato(roto))


# ---------------------------------------------------------------- entradas de inferencia (JSON)

@pytest.mark.parametrize("entrada, categoria", [
    (con(sepal_length="5.1"), "tipo"),                       # texto que parece número
    (con(sepal_length="cinco"), "tipo"),
    (con(petal_width=True), "tipo"),                          # bool es subclase de int en Python
    (con(sepal_width=None), "nulo"),
    (con(petal_length=math.nan), "tipo"),
    (con(petal_length=math.inf), "tipo"),
    (con(petal_width=-1.0), "rango"),
    (con(sepal_length=58.0), "rango"),                        # registro en milímetros
    (con(color="azul"), "estructura"),                        # campo extra
    ({k: v for k, v in VALIDA.items() if k != "petal_width"}, "estructura"),  # campo faltante
    ([5.1, 3.5, 1.4, 0.2], "estructura"),                     # lista en lugar de objeto
])
def test_entrada_invalida_se_rechaza(entrada, categoria):
    errores, _ = vd.validar_entrada(entrada, CONTRATO)
    assert categoria in categorias(errores)


def test_las_peticiones_de_ejemplo_se_comportan_como_dicen():
    # Son las peticiones que se enviarán a la API de la actividad 3.1
    validas = comun.leer_json(comun.RAIZ / "ejemplos" / "peticiones_validas.json")
    invalidas = comun.leer_json(comun.RAIZ / "ejemplos" / "peticiones_invalidas.json")
    assert all(vd.validar_entrada(p, CONTRATO)[0] == [] for p in validas)
    assert all(vd.validar_entrada(p, CONTRATO)[0] for p in invalidas)


def test_los_enteros_se_aceptan_como_numeros():
    errores, _ = vd.validar_entrada(con(sepal_length=6, sepal_width=3, petal_length=5, petal_width=2), CONTRATO)
    assert errores == []


def test_fuera_del_rango_observado_es_advertencia_y_no_error():
    errores, avisos = vd.validar_entrada(con(sepal_length=8.5), CONTRATO)
    assert errores == [] and len(avisos) == 1


# ---------------------------------------------------------------- archivos CSV

def test_un_csv_correcto_no_tiene_errores(tmp_path):
    resultado = vd.validar_archivo(escribir(tmp_path / "train.csv", FILAS_BASE), CONTRATO, "train")
    assert resultado.errores == [] and resultado.filas == 3


@pytest.mark.parametrize("columna, valor, categoria", [
    ("sepal_width", "tres", "tipo"),
    ("sepal_width", "3,5", "tipo"),       # coma decimal de una exportación regional de Excel
    ("petal_width", "", "nulo"),
    ("petal_width", "NA", "nulo"),
    ("petal_length", "51", "rango"),
    ("species", "4", "dominio"),
    ("species", "2.0", "tipo"),
    ("id", "", "nulo"),
])
def test_celda_invalida_en_csv(tmp_path, columna, valor, categoria):
    filas = [list(fila) for fila in FILAS_BASE]
    filas[0][COLUMNAS.index(columna)] = valor
    resultado = vd.validar_archivo(escribir(tmp_path / "train.csv", filas), CONTRATO, "train")
    assert categoria in categorias(resultado.errores)


def test_una_columna_renombrada_se_detecta_con_pista(tmp_path):
    columnas = [c if c != "sepal_length" else "sepal_len" for c in COLUMNAS]
    resultado = vd.validar_archivo(escribir(tmp_path / "train.csv", FILAS_BASE, columnas), CONTRATO, "train")
    mensajes = " ".join(m for _, m in resultado.errores)
    assert "falta la columna obligatoria 'sepal_length'" in mensajes and "'sepal_len'" in mensajes


def test_un_id_repetido_es_error(tmp_path):
    resultado = vd.validar_archivo(escribir(tmp_path / "train.csv", FILAS_BASE + [FILAS_BASE[0]]), CONTRATO, "train")
    assert "id" in categorias(resultado.errores)


def test_un_archivo_sin_filas_es_error(tmp_path):
    resultado = vd.validar_archivo(escribir(tmp_path / "train.csv", []), CONTRATO, "train")
    assert "estructura" in categorias(resultado.errores)


# ---------------------------------------------------------------- sabotajes de la sesión 4

@pytest.fixture
def datos_limpios(tmp_path):
    """Partición limpia en una carpeta temporal, para sabotearla sin tocar data/."""
    columnas, entrenamiento, prueba = prepare.generar_particion(CONTRATO)
    rutas = tmp_path / "train.csv", tmp_path / "test.csv"
    prepare.escribir_csv(rutas[0], entrenamiento, columnas)
    prepare.escribir_csv(rutas[1], prueba, columnas)
    return rutas


def validar_conjunto(ruta_train, ruta_test):
    entrenamiento = vd.validar_archivo(ruta_train, CONTRATO, "train")
    prueba = vd.validar_archivo(ruta_test, CONTRATO, "test")
    errores_conjunto, _ = vd.validar_reglas_conjunto(entrenamiento, prueba, CONTRATO)
    return entrenamiento.errores + prueba.errores + errores_conjunto


def test_la_particion_limpia_cumple_el_contrato(datos_limpios):
    assert validar_conjunto(*datos_limpios) == []


@pytest.mark.parametrize("numero", [1, 2, 3, 4, 5, 7])
def test_la_validacion_detiene_el_sabotaje(datos_limpios, numero):
    romper_datos.aplicar(numero, *datos_limpios)
    assert validar_conjunto(*datos_limpios), f"el sabotaje {numero} pasó la validación"


def test_las_etiquetas_mal_asignadas_no_las_ve_la_validacion(datos_limpios):
    # Límite conocido: el contrato revisa formato y dominio, no si la etiqueta es la correcta.
    # Este sabotaje lo detienen las pruebas del modelo (tests/test_model.py).
    romper_datos.aplicar(6, *datos_limpios)
    assert validar_conjunto(*datos_limpios) == []
