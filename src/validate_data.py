"""Etapa 2 del pipeline: valida los datos contra el contrato (data_contract.json) sin librerías externas.

Revisa, en este orden:
  1. el propio contrato: que su estructura sea coherente y que sus ejemplos se comporten como dicen;
  2. cada archivo (train.csv y test.csv): columnas, tipos, nulos, rangos lógicos, especie válida e ids únicos;
  3. las reglas entre archivos: las tres especies presentes y ninguna flor de prueba dentro del entrenamiento.

Los errores detienen el pipeline antes de entrenar: se listan con fila y columna, se guardan en
reportes/validacion_datos.json y el script termina con código 1. Las advertencias (por ejemplo, un valor
posible pero fuera del rango que conoce el modelo) se reportan sin detener nada.

Uso:
  python src/validate_data.py                        # contrato + data/train.csv + data/test.csv
  python src/validate_data.py --json peticion.json   # una entrada de inferencia (objeto o lista de objetos)
"""

import argparse
import csv
import difflib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import comun

ENTERO = re.compile(r"[+-]?\d+")           # "2" es entero; "2.0" no
COMA_DECIMAL = re.compile(r"[+-]?\d+,\d+")  # "5,1": formato regional que el contrato no acepta
MAX_EN_PANTALLA = 25                        # el reporte JSON guarda todos; la terminal muestra los primeros


# ---------------------------------------------------------------- utilidades

def es_numero(valor):
    """True solo para int o float finitos. Excluye bool, que en Python es una subclase de int."""
    return isinstance(valor, (int, float)) and not isinstance(valor, bool) and math.isfinite(valor)


def cantidad(n, singular, plural):
    return f"{n} {singular if n == 1 else plural}"


def nombre_tipo(valor):
    nombres = {bool: "booleano", str: "texto", list: "lista", dict: "objeto", type(None): "nulo"}
    return nombres.get(type(valor), type(valor).__name__)


def revisar_valor(valor, regla):
    """Revisa un valor ya interpretado contra la regla de su campo.

    Devuelve (error, aviso): error es None o una tupla (categoría, detalle); aviso es None o un texto.
    """
    unidad = regla.get("unit", "")
    if valor is None:
        if regla.get("nullable", False):
            return None, None
        return ("nulo", "es nulo o está vacío"), None
    if isinstance(valor, bool):
        return ("tipo", f"es un booleano ({str(valor).lower()}), no un número"), None
    if not isinstance(valor, (int, float)):
        return ("tipo", f"el valor {valor!r} es de tipo {nombre_tipo(valor)}; se esperaba un número"), None
    if not math.isfinite(valor):
        return ("tipo", f"el valor {valor} no es un número finito"), None
    minimo, maximo = regla["minimum"], regla["maximum"]
    if not minimo <= valor <= maximo:
        return ("rango", f"el valor {valor} está fuera del rango lógico [{minimo}, {maximo}] {unidad}"), None
    bajo, alto = regla["observed_range"]
    if not bajo <= valor <= alto:
        return None, (f"el valor {valor} es posible, pero está fuera del rango observado [{bajo}, {alto}] {unidad}: "
                      "el modelo no vio flores así (posible deriva de datos)")
    return None, None


# ---------------------------------------------------------------- contrato

def verificar_contrato(contrato):
    """Revisa que el contrato tenga las secciones esperadas y que sus reglas no se contradigan."""
    secciones = ["contract", "model", "features", "target", "record_id", "missing_value_tokens",
                 "additional_fields_allowed", "dataset_rules", "inference", "examples"]
    faltan = [s for s in secciones if s not in contrato]
    if faltan:
        return [f"faltan secciones: {', '.join(faltan)}"]

    problemas = []
    orden = contrato["features"].get("order", [])
    campos = contrato["features"].get("fields", {})
    if not orden or len(set(orden)) != len(orden):
        problemas.append("features.order debe ser una lista de nombres sin repetir")
    if set(orden) != set(campos):
        problemas.append("features.order y features.fields deben nombrar los mismos campos")
    for nombre, regla in campos.items():
        if regla.get("type") != "number":
            problemas.append(f"{nombre}: type debe ser 'number'")
        minimo, maximo = regla.get("minimum"), regla.get("maximum")
        if not (es_numero(minimo) and es_numero(maximo) and minimo < maximo):
            problemas.append(f"{nombre}: minimum debe ser un número menor que maximum")
            continue
        observado = regla.get("observed_range")
        if not (isinstance(observado, list) and len(observado) == 2 and all(es_numero(v) for v in observado)
                and minimo <= observado[0] <= observado[1] <= maximo):
            problemas.append(f"{nombre}: observed_range debe quedar dentro de [minimum, maximum]")

    clases = contrato["target"].get("allowed_values", [])
    if not clases or not all(isinstance(c, int) and not isinstance(c, bool) for c in clases):
        problemas.append("target.allowed_values debe ser una lista de enteros")
    elif sorted(contrato["target"].get("labels", {})) != sorted(str(c) for c in clases):
        problemas.append("target.labels debe tener una etiqueta por cada valor permitido")

    ejemplos = contrato["examples"]
    for clave in ("valid_input", "valid_output", "invalid_input", "invalid_input_reasons"):
        if not isinstance(ejemplos.get(clave), dict):
            problemas.append(f"examples.{clave} debe ser un objeto JSON")
    if isinstance(ejemplos.get("valid_output"), dict) and \
            set(ejemplos["valid_output"]) != set(contrato["inference"].get("response", {})):
        problemas.append("examples.valid_output debe tener los mismos campos que inference.response")
    return problemas


def validar_entrada(entrada, contrato):
    """Valida una petición de inferencia (un objeto JSON ya leído). Devuelve (errores, avisos)."""
    if not isinstance(entrada, dict):
        return [("estructura", f"la entrada debe ser un objeto JSON; se recibió {nombre_tipo(entrada)}")], []
    errores, avisos = [], []
    campos = contrato["features"]["fields"]
    if not contrato["additional_fields_allowed"]:
        for extra in sorted(set(entrada) - set(campos)):
            errores.append(("estructura", f"campo '{extra}': no está definido en el contrato"))
    for nombre in contrato["features"]["order"]:
        regla = campos[nombre]
        if nombre not in entrada:
            if regla.get("required", True):
                errores.append(("estructura", f"campo '{nombre}': falta y es obligatorio"))
            continue
        error, aviso = revisar_valor(entrada[nombre], regla)
        if error:
            errores.append((error[0], f"campo '{nombre}': {error[1]}"))
        if aviso:
            avisos.append(f"campo '{nombre}': {aviso}")
    return errores, avisos


def autoverificar_contrato(contrato):
    """Prueba el contrato contra sí mismo: el ejemplo válido debe pasar y el inválido debe fallar por sus razones.

    Devuelve (problemas, errores_del_ejemplo_invalido).
    """
    problemas = verificar_contrato(contrato)
    if problemas:
        return problemas, []
    ejemplos = contrato["examples"]
    errores_validos, _ = validar_entrada(ejemplos["valid_input"], contrato)
    for _, detalle in errores_validos:
        problemas.append(f"el ejemplo válido no cumple el contrato: {detalle}")
    errores_invalidos, _ = validar_entrada(ejemplos["invalid_input"], contrato)
    if not errores_invalidos:
        problemas.append("el ejemplo inválido pasa la validación")
    for campo in ejemplos["invalid_input_reasons"]:
        if not any(f"'{campo}'" in detalle for _, detalle in errores_invalidos):
            problemas.append(f"el ejemplo inválido no produce el error que documenta para '{campo}'")
    return problemas, errores_invalidos


# ---------------------------------------------------------------- archivos CSV

class ResultadoArchivo:
    """Lo que se encontró en un archivo: errores, advertencias e insumos para las reglas entre archivos."""

    def __init__(self, ruta, rol):
        self.ruta = Path(ruta)
        self.rol = rol
        self.sha256 = None
        self.filas = 0
        self.errores = []        # tuplas (categoría, mensaje)
        self.advertencias = []
        self.conteo_clases = Counter()
        self.ids = []
        self.mediciones = {}     # (medidas..., especie) -> lista de ids

    def error(self, categoria, mensaje):
        self.errores.append((categoria, mensaje))

    def resumen(self, etiquetas):
        try:  # ruta relativa al proyecto: el reporte no expone carpetas personales del equipo
            ruta = self.ruta.resolve().relative_to(comun.RAIZ).as_posix()
        except ValueError:
            ruta = self.ruta.name
        return {
            "ruta": ruta,
            "sha256": self.sha256,
            "filas": self.filas,
            "errores": len(self.errores),
            "advertencias": len(self.advertencias),
            "conteo_por_especie": {etiquetas.get(str(k), str(k)): v for k, v in sorted(self.conteo_clases.items())},
        }


def interpretar_celda(texto, tokens_faltantes):
    """Convierte el texto de una celda en número. Devuelve (valor, error); un faltante se devuelve como None."""
    if texto is None:
        return None, None
    limpio = texto.strip()
    if limpio.lower() in tokens_faltantes:
        return None, None
    try:
        return float(limpio), None
    except ValueError:
        if COMA_DECIMAL.fullmatch(limpio):
            return None, ("tipo", f"el valor '{limpio}' usa coma decimal; el contrato exige punto "
                                  f"({limpio.replace(',', '.')})")
        return None, ("tipo", f"el valor '{limpio}' no es un número")


def interpretar_entero(texto, tokens_faltantes):
    """Para id y especie: solo enteros escritos sin decimales. Devuelve (valor, error)."""
    if texto is None or texto.strip().lower() in tokens_faltantes:
        return None, ("nulo", "es nulo o está vacío")
    limpio = texto.strip()
    if not ENTERO.fullmatch(limpio):
        return None, ("tipo", f"el valor '{limpio}' no es un número entero")
    return int(limpio), None


def validar_archivo(ruta, contrato, rol):
    """Valida un CSV de entrenamiento o de prueba contra el contrato."""
    res = ResultadoArchivo(ruta, rol)
    nombre = res.ruta.name
    if not res.ruta.is_file():
        res.error("estructura", f"{nombre}: no existe el archivo")
        return res
    res.sha256 = comun.sha256_archivo(res.ruta)

    orden = contrato["features"]["order"]
    campos = contrato["features"]["fields"]
    col_id = contrato["record_id"]["name"]
    objetivo = contrato["target"]
    col_objetivo = objetivo["name"]
    clases = set(objetivo["allowed_values"])
    tokens = {t.lower() for t in contrato["missing_value_tokens"]}
    esperadas = [col_id] + orden + [col_objetivo]

    # utf-8-sig tolera el BOM que agrega Excel al exportar a CSV
    with open(res.ruta, newline="", encoding="utf-8-sig") as f:
        lector = csv.DictReader(f)
        encabezado = lector.fieldnames
        if not encabezado:
            res.error("estructura", f"{nombre}: el archivo está vacío")
            return res
        repetidas = [c for c, n in Counter(encabezado).items() if n > 1]
        if repetidas:
            res.error("estructura", f"{nombre}: columnas repetidas en el encabezado: {', '.join(repetidas)}")
        faltan = [c for c in esperadas if c not in encabezado]
        sobran = [c for c in encabezado if c not in esperadas]
        for col in faltan:
            parecida = difflib.get_close_matches(col, sobran, n=1, cutoff=0.6)
            pista = f" (¿se renombró como '{parecida[0]}'?)" if parecida else ""
            res.error("estructura", f"{nombre}: falta la columna obligatoria '{col}'{pista}")
        if not contrato["additional_fields_allowed"]:
            for col in sobran:
                res.error("estructura", f"{nombre}: la columna '{col}' no está definida en el contrato")
        if not faltan and not sobran and list(encabezado) != esperadas:
            res.advertencias.append(f"{nombre}: las columnas vienen en otro orden; el entrenamiento las toma por nombre")

        vistos_ids = defaultdict(list)
        for numero, fila in enumerate(lector, start=1):
            res.filas = numero
            texto_id = (fila.get(col_id) or "").strip() if col_id in encabezado else ""
            prefijo = f"{nombre} · Fila {numero}" + (f" (id={texto_id})" if texto_id else "")
            if None in fila:
                res.error("estructura", f"{prefijo}: tiene más valores que columnas el encabezado")
            elif any(v is None for v in fila.values()):
                res.error("estructura", f"{prefijo}: tiene menos valores que columnas el encabezado")

            errores_antes = len(res.errores)
            if col_id in encabezado:
                valor_id, error = interpretar_entero(fila.get(col_id), tokens)
                if error:
                    res.error(error[0], f"{prefijo}, columna '{col_id}': {error[1]}")
                elif valor_id < contrato["record_id"].get("minimum", 1):
                    res.error("rango", f"{prefijo}, columna '{col_id}': debe ser mayor o igual que "
                                       f"{contrato['record_id'].get('minimum', 1)}")
                else:
                    vistos_ids[valor_id].append(numero)
                    res.ids.append(valor_id)

            medidas = []
            for col in orden:
                if col not in encabezado:
                    continue
                valor, error = interpretar_celda(fila.get(col), tokens)
                if not error:
                    error, aviso = revisar_valor(valor, campos[col])
                    if aviso:
                        res.advertencias.append(f"{prefijo}, columna '{col}': {aviso}")
                if error:
                    res.error(error[0], f"{prefijo}, columna '{col}': {error[1]}")
                medidas.append(valor)

            if col_objetivo in encabezado:
                especie, error = interpretar_entero(fila.get(col_objetivo), tokens)
                if not error and especie not in clases:
                    error = ("dominio", f"la especie {especie} no existe; valores permitidos: "
                                        f"{sorted(clases)} ({', '.join(objetivo['labels'].values())})")
                if error:
                    res.error(error[0], f"{prefijo}, columna '{col_objetivo}': {error[1]}")
                else:
                    res.conteo_clases[especie] += 1
                    if len(res.errores) == errores_antes and len(medidas) == len(orden):
                        res.mediciones.setdefault(tuple(medidas) + (especie,), []).append(texto_id)

    if res.filas == 0:
        res.error("estructura", f"{nombre}: no tiene filas de datos")
    for valor_id, filas in sorted(vistos_ids.items()):
        if len(filas) > 1:
            res.error("id", f"{nombre}: el id {valor_id} se repite en las filas {', '.join(map(str, filas))}")
    if contrato["dataset_rules"].get("duplicate_rows") == "warning":
        for _, ids in res.mediciones.items():
            if len(ids) > 1:
                res.advertencias.append(f"{nombre}: las flores con id {', '.join(ids)} tienen medidas y especie "
                                        "idénticas (es posible con medidas de 0,1 cm; no es un error)")
    return res


def validar_reglas_conjunto(entrenamiento, prueba, contrato):
    """Reglas que no se ven fila por fila: cobertura de especies y separación entre entrenamiento y prueba."""
    reglas = contrato["dataset_rules"]
    etiquetas = contrato["target"]["labels"]
    errores, avisos = [], []
    for res in (entrenamiento, prueba):
        if res.filas == 0:
            continue
        for clase in contrato["target"]["allowed_values"]:
            n = res.conteo_clases.get(clase, 0)
            if reglas.get("all_classes_present") and n == 0:
                errores.append(("cobertura", f"{res.ruta.name}: no hay ninguna flor de la especie {clase} "
                                             f"({etiquetas[str(clase)]}); el modelo no podría aprenderla o evaluarla"))
            elif n < reglas.get("min_rows_per_class", 0):
                errores.append(("cobertura", f"{res.ruta.name}: solo hay {n} flores de la especie {clase} "
                                             f"({etiquetas[str(clase)]}); el mínimo es {reglas['min_rows_per_class']}"))
    if not reglas.get("train_test_overlap_allowed", True):
        compartidos = sorted(set(entrenamiento.ids) & set(prueba.ids))
        if compartidos:
            muestra = ", ".join(map(str, compartidos[:10])) + (" ..." if len(compartidos) > 10 else "")
            errores.append(("fuga", f"{len(compartidos)} flores de prueba también están en entrenamiento "
                                    f"(ids {muestra}): es fuga de datos y la exactitud en prueba quedaría inflada"))
        repetidas = set(entrenamiento.mediciones) & set(prueba.mediciones)
        if repetidas:
            avisos.append(f"{len(repetidas)} filas de prueba tienen medidas y especie idénticas a filas de "
                          "entrenamiento; si no son la misma flor no es fuga, pero conviene revisarlas")
    return errores, avisos


# ---------------------------------------------------------------- ejecución

def parsear_argumentos():
    parser = argparse.ArgumentParser(description="Valida los datos contra el contrato (sin librerías externas).")
    parser.add_argument("--contrato", default=comun.CONTRATO, type=Path, help="ruta del contrato JSON")
    parser.add_argument("--train", default=comun.TRAIN_CSV, type=Path, help="CSV de entrenamiento")
    parser.add_argument("--test", default=comun.TEST_CSV, type=Path, help="CSV de prueba")
    parser.add_argument("--json", type=Path, help="valida una petición de inferencia en lugar de los CSV")
    return parser.parse_args()


def validar_peticion(ruta, contrato, log):
    """Modo --json: valida uno o varios objetos como los recibiría la API de inferencia."""
    with open(ruta, encoding="utf-8") as f:
        contenido = json.load(f)
    entradas = contenido if isinstance(contenido, list) else [contenido]
    rechazadas = 0
    for i, entrada in enumerate(entradas, start=1):
        errores, avisos = validar_entrada(entrada, contrato)
        estado = "RECHAZADA" if errores else "ACEPTADA"
        rechazadas += bool(errores)
        log.info("Entrada %d: %s", i, estado)
        for _, detalle in errores:
            log.error("  - %s", detalle)
        for aviso in avisos:
            log.warning("  - %s", aviso)
    return 1 if rechazadas else 0


def main():
    args = parsear_argumentos()
    log = comun.configurar_logs("validacion")
    try:
        contrato = comun.leer_json(args.contrato)
    except (OSError, json.JSONDecodeError) as exc:
        log.error("No se pudo leer el contrato %s: %s", args.contrato, exc)
        return 2

    # 1. El contrato se prueba a sí mismo antes de juzgar los datos
    ficha = contrato.get("contract", {})
    nombre_contrato = f"{ficha.get('name', '?')} v{ficha.get('version', '?')}"
    problemas, errores_ejemplo = autoverificar_contrato(contrato)
    if problemas:
        for problema in problemas:
            log.error("Contrato: %s", problema)
        log.error("El contrato no es coherente; se corrige antes de validar datos.")
        return 1
    log.info("Contrato %s: estructura coherente (%d variables, %d especies)", nombre_contrato,
             len(contrato["features"]["order"]), len(contrato["target"]["allowed_values"]))
    log.info("Ejemplo válido del contrato: aceptado")
    log.info("Ejemplo inválido del contrato: rechazado con %s, los que documenta el contrato",
             cantidad(len(errores_ejemplo), "error", "errores"))
    for _, detalle in errores_ejemplo:
        log.info("    %s", detalle)

    if args.json:
        return validar_peticion(args.json, contrato, log)

    # 2 y 3. Cada archivo y luego las reglas entre archivos
    entrenamiento = validar_archivo(args.train, contrato, "train")
    prueba = validar_archivo(args.test, contrato, "test")
    errores_conjunto, avisos_conjunto = validar_reglas_conjunto(entrenamiento, prueba, contrato)

    errores = [(entrenamiento.ruta.name, c, m) for c, m in entrenamiento.errores] + \
              [(prueba.ruta.name, c, m) for c, m in prueba.errores] + \
              [("conjunto", c, m) for c, m in errores_conjunto]
    avisos = entrenamiento.advertencias + prueba.advertencias + avisos_conjunto
    aprobado = not errores
    etiquetas = contrato["target"]["labels"]

    comun.guardar_json({
        "fecha_utc": comun.ahora_utc(),
        "contrato": {"nombre": ficha.get("name"), "version": ficha.get("version"),
                     "sha256": comun.sha256_archivo(args.contrato)},
        "archivos": {"train": entrenamiento.resumen(etiquetas), "test": prueba.resumen(etiquetas)},
        "errores_por_categoria": dict(Counter(c for _, c, _ in errores)),
        "errores": [{"origen": o, "categoria": c, "mensaje": m} for o, c, m in errores],
        "advertencias": avisos,
        "resultado": "APROBADO" if aprobado else "RECHAZADO",
    }, comun.REPORTE_VALIDACION)

    for res in (entrenamiento, prueba):
        log.info("%s: %s, %s, %s", res.ruta.name, cantidad(res.filas, "fila", "filas"),
                 cantidad(len(res.errores), "error", "errores"),
                 cantidad(len(res.advertencias), "advertencia", "advertencias"))
    log.info("Reglas entre archivos (especies presentes, sin fuga): %s, %s",
             cantidad(len(errores_conjunto), "error", "errores"),
             cantidad(len(avisos_conjunto), "advertencia", "advertencias"))
    for aviso in avisos[:MAX_EN_PANTALLA]:
        log.warning(aviso)

    if aprobado:
        log.info("Resultado: APROBADO. Los datos cumplen el contrato %s; el pipeline puede entrenar.", nombre_contrato)
        return 0
    log.error("Validación fallida: %s (%s)", cantidad(len(errores), "error", "errores"),
              ", ".join(f"{c}: {n}" for c, n in Counter(c for _, c, _ in errores).most_common()))
    for _, _, mensaje in errores[:MAX_EN_PANTALLA]:
        log.error("  - %s", mensaje)
    if len(errores) > MAX_EN_PANTALLA:
        log.error("  ... y %d errores más en reportes/validacion_datos.json", len(errores) - MAX_EN_PANTALLA)
    log.error("Resultado: RECHAZADO. El pipeline se detiene antes de entrenar.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
