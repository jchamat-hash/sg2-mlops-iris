# Pipeline reproducible y control de calidad de datos: clasificador de Iris

Seminario de Grado II · Especialización en Ciencia de Datos · Universidad Santo Tomás
Actividad 2.1 «Estandarización de Pipelines y Control de Calidad de Datos» · Jonathan Chamat · noviembre de 2026

Un solo comando prepara los datos, los valida contra un contrato explícito, entrena un Random Forest, lo registra en MLflow y lo somete a 54 pruebas automáticas. Si una etapa falla, el pipeline se detiene: no se entrena con datos rechazados ni se aprueba un modelo que no pasó las pruebas. El contrato de datos es, además, la especificación oficial de lo que recibirá la API de inferencia de la actividad 3.1.

## 1. Inicio rápido

**macOS o Linux** (requiere Python 3.13, `git` y `make`):

```bash
git clone https://github.com/jchamat-hash/sg2-mlops-iris.git
cd sg2-mlops-iris
make
```

`make` crea el entorno virtual `.venv`, instala las versiones exactas de `requirements.txt` y ejecuta el pipeline. La primera vez tarda unos minutos por la instalación; después, menos de un minuto.

**Windows (PowerShell)**, con los mismos pasos de la sesión 4:

```powershell
git clone https://github.com/jchamat-hash/sg2-mlops-iris.git
cd sg2-mlops-iris
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py
```

Si PowerShell bloquea la activación del entorno: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`.

En cualquier sistema, con el entorno activado, el comando único es `python run_pipeline.py`. Una ejecución correcta termina así (el RUN_ID y los tiempos cambian en cada ejecución):

```text
54 passed in 2.98s

================================================================
PIPELINE FINALIZADO CORRECTAMENTE en 26.0 s (1: 3.8 s | 2: 0.0 s | 3: 16.1 s | 4: 4.4 s)
RUN_ID del modelo aprobado: c6a3fec7bff948d1a7c99b04e8d215cd
```

## 2. Estructura del repositorio

**Tabla 1**
*Archivos del proyecto y responsabilidad de cada uno*

| Archivo | Responsabilidad |
|---|---|
| `run_pipeline.py` | Orquesta las cuatro etapas con un solo comando y se detiene en la primera que falle |
| `Makefile` | Atajos para macOS y Linux (`make`, `make ui`, `make sabotajes`); la primera vez crea el entorno |
| `data_contract.json` | Contrato de datos: nombres, tipos, nulos, rangos, especies válidas, reglas del conjunto y ejemplos de entrada válida e inválida |
| `src/prepare.py` | Exporta Iris con los nombres del contrato y genera `data/train.csv` y `data/test.csv` (80/20, estratificado) |
| `src/validate_data.py` | Valida el contrato y los datos con la biblioteca estándar de Python; también valida peticiones JSON |
| `src/train.py` | Entrena el Random Forest con los datos validados y lo registra en MLflow |
| `src/comun.py` | Rutas, registro de ejecución (logs) y huellas SHA-256 que comparten las etapas |
| `tests/test_validate_data.py` | 39 pruebas del contrato y del validador |
| `tests/test_model.py` | 15 pruebas del modelo: trazabilidad, coherencia con el contrato y desempeño mínimo |
| `romper_datos.py` | Aplica los siete sabotajes de la sesión 4 para comprobar qué etapa detiene cada uno |
| `ejemplos/` | Cinco peticiones de inferencia válidas y seis inválidas, las mismas que recibirá la API de la actividad 3.1 |
| `requirements.txt`, `pytest.ini`, `.gitignore` | Versiones exactas, configuración de las pruebas y archivos que no se versionan |

*Nota.* La estructura sigue el proyecto de referencia de la sesión 4, en el que cada archivo tiene una sola responsabilidad (Contreras Ortiz, 2026b). Lo que genera el pipeline (`data/`, `reportes/`, `logs/`, `mlflow.db` y `mlruns/`) no se versiona, porque se regenera con el comando único.

## 3. Requisitos

- Python 3.13. El proyecto se desarrolló en macOS con Apple Silicon y se verificó desde un clon limpio en Linux.
- `git`. En macOS y Linux, también `make`; en macOS viene con las herramientas de línea de comandos (`xcode-select --install`).
- Las librerías se instalan con versiones fijas (Tabla 2).

**Tabla 2**
*Versiones del entorno*

| Paquete | Versión | Uso en el proyecto |
|---|---|---|
| mlflow | 3.18.0 | Registro de experimentos, métricas, artefactos y del modelo |
| scikit-learn | 1.9.1 | Conjunto Iris, partición estratificada y Random Forest |
| numpy | 2.5.4 | Cálculo numérico |
| pandas | 3.1.1 | Lectura de los CSV ya validados para entrenar |
| pytest | 9.2.0 | Pruebas automatizadas (no participa en la validación de datos) |

*Nota.* La validación de datos no usa ninguna de estas librerías (sección 6).

## 4. Qué hace cada etapa

```text
python run_pipeline.py
 ├─ 1. Preparación     src/prepare.py         Iris -> data/train.csv (120 flores) y data/test.csv (30)
 ├─ 2. Validación      src/validate_data.py   contrato + CSV -> APROBADO, o se detiene (código 1)
 ├─ 3. Entrenamiento   src/train.py           datos validados -> run en MLflow con modelo y trazabilidad
 └─ 4. Pruebas         pytest                 54 pruebas -> modelo aprobado o rechazado (etiqueta en MLflow)
```

Cada etapa corre en su propio proceso. Si termina con un código de salida distinto de 0, el orquestador no ejecuta las siguientes, según el principio de fallar rápido de la sesión 4 (Contreras Ortiz, 2026b). El libro guía define el pipeline único como la ejecución de todo el flujo, desde la ingesta de datos hasta la validación, con una sola interfaz de comando, lo que reduce el error humano y garantiza que cada miembro del equipo obtenga resultados idénticos en las mismas condiciones (Contreras Ortiz et al., 2026, p. 111). También evita lo que Sculley et al. (2015) llaman *pipeline jungles*: preparaciones de datos que crecen como pasos sueltos y archivos intermedios difíciles de mantener.

Al terminar quedan cuatro registros: `reportes/validacion_datos.json` (el reporte de calidad), `reportes/ultimo_run.json` (el RUN_ID y las métricas del modelo), `reportes/pipeline_resumen.json` (duración y resultado de cada etapa) y `logs/pipeline.log`, que acumula la historia de todas las ejecuciones con fecha y hora.

Otros comandos útiles:

| Comando | Qué hace |
|---|---|
| `python run_pipeline.py --sin-preparar` | Valida, entrena y prueba con los CSV tal como están (sirve para probar datos saboteados) |
| `python run_pipeline.py --max-depth 5 --run-name Prueba_depth5` | Ejecuta el pipeline con otra configuración del modelo |
| `python src/validate_data.py` | Solo la validación del contrato y de los datos |
| `python -m pytest` | Solo las pruebas automatizadas |

## 5. Contrato de datos (`data_contract.json`)

El contrato es el acuerdo formal sobre cómo deben ser los datos: describe exactamente qué espera recibir el modelo y facilita su integración posterior con una API (Contreras Ortiz et al., 2026, p. 112). Las mismas reglas se aplican a los datos de entrenamiento y a las peticiones de inferencia.

**Tabla 3**
*Variables de entrada definidas en el contrato*

| Campo | Tipo | Unidad | Rango lógico (fuera, se rechaza) | Rango observado (fuera, se advierte) | Nombre original en scikit-learn |
|---|---|---|---|---|---|
| `sepal_length` | number | cm | 0,1 a 10,0 | 4,3 a 7,9 | `sepal length (cm)` |
| `sepal_width` | number | cm | 0,1 a 10,0 | 2,0 a 4,4 | `sepal width (cm)` |
| `petal_length` | number | cm | 0,1 a 10,0 | 1,0 a 6,9 | `petal length (cm)` |
| `petal_width` | number | cm | 0,1 a 10,0 | 0,1 a 2,5 | `petal width (cm)` |

*Nota.* Los cuatro campos son obligatorios (`required: true`) y no admiten nulos (`nullable: false`). El rango observado es el mínimo y el máximo de las 150 flores de Iris. El orden de la tabla es el orden en que el modelo recibe las variables.

**Política de rangos en dos niveles.** Un valor fuera del rango lógico no es una medida posible de una flor de iris en centímetros (por ejemplo, un registro digitado en milímetros) y se rechaza. Un valor dentro del rango lógico pero fuera del observado es una flor posible que el modelo no vio al entrenar: se acepta con una advertencia, que funciona como señal temprana de deriva de datos (*data drift*). Un rango demasiado estrecho rechazaría flores reales; uno demasiado amplio dejaría pasar errores de unidades.

**Otras reglas del contrato:**

- `species` es un entero de {0, 1, 2}, con las etiquetas setosa, versicolor y virginica. Solo existe en los datos de entrenamiento y prueba.
- `id` identifica cada flor (su número en el conjunto original) y sirve para rastrearla. No es una variable del modelo: Iris está ordenado por especie y el id delataría la clase.
- Se tratan como faltantes la celda vacía y los marcadores `na`, `n/a`, `nan`, `null` y `none`.
- No se admiten campos que el contrato no defina (`additional_fields_allowed: false`).
- Reglas sobre el conjunto (`dataset_rules`): cada archivo trae las tres especies con al menos 5 flores de cada una, ninguna flor de prueba puede estar en entrenamiento y las filas repetidas se reportan como advertencia.

**Ejemplos de entrada válida e inválida** (sección `examples`):

```json
"valid_input":   {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
"valid_output":  {"prediction": "setosa", "class_id": 0, "run_id": "<RUN_ID del modelo en MLflow>"},
"invalid_input": {"sepal_length": "5.1", "sepal_width": null, "petal_length": -1.4, "petal_width": true, "color": "azul"}
```

La entrada inválida falla de cinco maneras: un número escrito como texto, un nulo, un valor negativo, un booleano y un campo que no existe. Cada razón está documentada en `invalid_input_reasons`, y en cada ejecución el validador comprueba que rechaza exactamente esas cinco (autoverificación del contrato).

**Coherencia con el modelo y con la inferencia.** El entrenamiento toma las columnas en el orden del contrato, así que el modelo guarda esos nombres (`feature_names_in_`) y la firma registrada en MLflow tiene los mismos cuatro campos; dos pruebas lo verifican. Otra prueba exige que el ejemplo válido produzca la salida documentada (`setosa`, clase 0). La API de la actividad 3.1 usará los mismos nombres, rangos y la misma política de campos extra, lo que evita el sesgo entre entrenamiento y servicio (*training-serving skew*; Contreras Ortiz et al., 2026, p. 112). Las peticiones se pueden validar desde ya; la carpeta `ejemplos/` trae cinco válidas (una por especie, un caso en la frontera entre versicolor y virginica, y uno con números enteros) y seis inválidas (texto, booleano, nulo, campo extra, valores en milímetros y un campo faltante):

```bash
python src/validate_data.py --json ejemplos/peticiones_validas.json
python src/validate_data.py --json ejemplos/peticiones_invalidas.json
```

## 6. Validación de datos (`src/validate_data.py`)

La validación usa solo la biblioteca estándar de Python (`csv`, `json`, `math`, `re`, `difflib`, `collections` y `pathlib`), como exige la actividad. Una prueba automática revisa los `import` de `validate_data.py` y de `comun.py` contra la lista de módulos estándar del intérprete (`sys.stdlib_module_names`), así que cualquier dependencia externa haría fallar el pipeline. La semana 2 del portal del curso orienta esta etapa (MLOps Open Source - Santoto, 2026).

**Tabla 4**
*Controles de calidad del validador*

| Control | Ejemplos que detecta | Efecto |
|---|---|---|
| Estructura | Falta o sobra una columna, una columna renombrada (sugiere el nombre parecido), filas con más o menos valores, archivo vacío | Error |
| Tipo | `cinco`, `3,5` (coma decimal), `true`, `2.0` como especie, infinito | Error |
| Nulos | Celda vacía, `NA`, `n/a`, `nan`, `null`, `none` | Error |
| Rango lógico | `51` en `petal_length` (registro en milímetros), `-1.4` | Error |
| Rango observado | `sepal_length = 8.4` | Advertencia (posible deriva) |
| Especie | `species = 3` | Error |
| Identificador | `id` vacío, repetido o menor que 1 | Error |
| Cobertura de especies | Un archivo sin flores virginica, o con menos de 5 de una especie | Error |
| Fuga entre archivos | Una flor de prueba (mismo `id`) dentro del entrenamiento | Error |
| Filas repetidas | Dos flores con las mismas medidas y especie | Advertencia |

*Nota.* Un error detiene el pipeline antes de entrenar; una advertencia queda registrada, pero no lo detiene.

Cada error se informa con archivo, fila, id y columna, por ejemplo: `train.csv · Fila 61 (id=79), columna 'petal_width': el valor 15.0 está fuera del rango lógico [0.1, 10.0] cm`. El reporte completo queda en `reportes/validacion_datos.json`, con la huella SHA-256 de cada archivo, y el resumen queda en `logs/pipeline.log`. Así el pipeline deja la huella de qué validaciones pasaron y cuáles fallaron que pide el libro guía (Contreras Ortiz et al., 2026, p. 112). Es la misma idea que Breck et al. (2019) aplican a gran escala: los errores en los datos de entrada pueden anular cualquier mejora del entrenamiento o de la inferencia, por eso cada lote se compara con un esquema antes de usarse.

Un hallazgo en los datos limpios: el validador advierte que las flores 102 y 143 tienen medidas y especie idénticas. No es un error, porque con medidas de 0,1 cm dos flores pueden coincidir, y como ambas quedaron en entrenamiento no afectan la prueba.

## 7. Entrenamiento y trazabilidad en MLflow

`src/train.py` evoluciona el script de la actividad 1.1, que a su vez parte del de la sesión 2 (Contreras Ortiz, 2026a). Entrena la configuración seleccionada en la 1.1 (`Run_3_depth3`: 100 árboles, profundidad máxima 3, semilla 42), que tuvo la mejor validación cruzada con una brecha pequeña. La validación cruzada estratificada de 5 particiones repetida 10 veces usa solo el entrenamiento; las 30 flores de prueba se usan una sola vez. La brecha (`overfit_gap`) se calcula como exactitud en entrenamiento menos validación cruzada, igual que en la 1.1.

Antes de entrenar, `train.py` comprueba que la última validación haya aprobado exactamente estos archivos: compara la huella SHA-256 de `train.csv`, `test.csv` y del contrato con las del reporte. Si alguien edita un dato después de validar, no se entrena.

**Tabla 5**
*Qué registra cada run del pipeline en MLflow*

| Tipo | Contenido |
|---|---|
| Parámetros | `n_estimators`, `max_depth`, `model_seed`, `split_seed`, `test_size`, `split_estratificado`, `cv_folds`, `cv_repeats`, `filas_train`, `filas_test` |
| Métricas | `accuracy`, `f1_macro`, `train_accuracy`, `cv_mean`, `cv_std`, `cv_log_loss`, `oob_accuracy`, `overfit_gap`, `mean_depth`, `mean_leaves` |
| Etiquetas | autor, actividad, comando exacto, versiones de Python, scikit-learn y MLflow, sistema, fecha, `git_commit`, contrato y versión, `validacion_datos`, `sha256_train`, `sha256_test` y `pruebas_modelo` (aprobado o rechazado) |
| Artefactos | `iris_model` (con firma y ejemplo de entrada), `contrato/data_contract.json`, `calidad/validacion_datos.json`, `datos/preparacion.json`, `entorno/requirements.txt` |

*Nota.* La etiqueta `pruebas_modelo` la escribe `run_pipeline.py` al terminar la etapa 4. Solo un modelo marcado como aprobado debería desplegarse en la actividad 3.1. El registro de experimentos se hace con MLflow (Zaharia et al., 2018).

**Para ver los resultados en MLflow UI**, desde la carpeta del proyecto (la base `mlflow.db` se busca en la carpeta actual):

```bash
make ui
```

o, con el entorno activado:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001 --workers 1
```

Después se abre `http://127.0.0.1:5001`, se elige el modo **Model training** y se entra al experimento `Proyecto_Final_Grado`. Se usa el puerto 5001 porque en macOS el 5000 lo ocupa el Receptor AirPlay.

## 8. Pruebas automatizadas (`tests/`)

Validar los datos y probar el modelo son controles distintos: el primero ocurre antes de entrenar y revisa los datos contra el contrato; el segundo ocurre después y revisa el modelo contra umbrales (Contreras Ortiz, 2026b). Evaluar en varias etapas antes de desplegar es una práctica habitual de quienes operan modelos en producción (Shankar et al., 2024).

- `tests/test_validate_data.py` (39 pruebas): el validador no importa librerías externas; el contrato es coherente y sus ejemplos se comportan como dicen; cada tipo de entrada inválida (texto, booleano, nulo, NaN, infinito, fuera de rango, campo extra, campo faltante) se rechaza; las peticiones de `ejemplos/` se aceptan o se rechazan según su nombre; los CSV con celdas, columnas o ids defectuosos se rechazan; y los sabotajes 1 a 5 y 7 de la sesión 4 se detienen en la validación.
- `tests/test_model.py` (15 pruebas): el run terminó y se entrenó con los datos validados (mismas huellas); guarda contrato, reporte de validación, datos y entorno; el modelo se carga por su RUN_ID, como lo hará la API; usa las variables y el orden del contrato; la firma coincide; el ejemplo válido da la salida documentada; tres flores de referencia (una por especie) se clasifican bien; y se cumplen los umbrales de la Tabla 6.

**Tabla 6**
*Umbrales de aprobación del modelo*

| Criterio | Umbral | Justificación |
|---|---|---|
| Exactitud en prueba | ≥ 0,80 | Umbral del proyecto de la sesión 4 |
| Validación cruzada media | ≥ 0,90 | En la actividad 1.1 todas las configuraciones superaron 0,91 |
| Brecha (entrenamiento − validación cruzada) | ≤ 0,10 | En la 1.1 la brecha más alta fue 0,058; por encima de 0,10 el modelo memoriza |

## 9. Resultados de referencia

**Tabla 7**
*Métricas del modelo aprobado*

| Métrica | Valor |
|---|---|
| Exactitud en prueba (`accuracy`) | 0,967 (29 de 30 flores) |
| F1 macro en prueba | 0,967 |
| Exactitud en entrenamiento | 0,983 |
| Validación cruzada (5 × 10) | 0,958 ± 0,039 |
| Exactitud OOB | 0,950 |
| Brecha (entrenamiento − validación cruzada) | 0,026 |
| Log-loss en validación cruzada | 0,101 |
| Profundidad media y hojas por árbol | 2,98 y 4,98 |

*Nota.* Configuración de la actividad 1.1 (100 árboles, profundidad 3) con la partición estratificada del pipeline. Con las versiones de `requirements.txt` y las semillas indicadas, las métricas se repiten exactamente; el RUN_ID y los tiempos cambian en cada ejecución.

La única flor mal clasificada en la prueba es la 78 (6,7; 3,0; 5,0; 1,7 cm), una versicolor que el modelo asignó a virginica: su ancho de pétalo está en la zona donde las dos especies se superponen. Con 30 flores, cada error vale 3,3 puntos de exactitud, y el intervalo de Clopper-Pearson al 95 % para 29 aciertos de 30 va de 0,828 a 0,999. Por eso la validación cruzada (0,958) es la estimación más estable. En la actividad 1.1 la misma configuración obtuvo 1,000 en prueba con una partición no estratificada; la diferencia viene de la partición y no del modelo, como ya había mostrado el análisis de robustez de esa actividad, donde el 1,000 apareció solo en el 23 % de las particiones.

Para comprobar que los datos generados son idénticos, sus huellas deben coincidir con estas:

| Archivo | SHA-256 |
|---|---|
| `data/train.csv` | `31a232d594bfdd8d28c1f2a6f638e9b9bdfbf1a5b9fc06cff7d66d232ac66733` |
| `data/test.csv` | `4763d884a3966d528ca606a8d0cb84cc89412f03079d980533f6ffd60ed79929` |

En macOS: `shasum -a 256 data/*.csv`; en Linux: `sha256sum data/*.csv`; en Windows: `Get-FileHash data\*.csv`.

## 10. Siete sabotajes: qué etapa detiene cada uno

`romper_datos.py` reproduce los sabotajes del juego «¡Rompe el pipeline!» de la sesión 4 (Contreras Ortiz, 2026b). `make sabotajes` (o `python romper_datos.py --todos`) los ejecuta de punta a punta y guarda la tabla en `reportes/sabotajes.md`. Para uno solo: `make sabotaje N=3`, o `python romper_datos.py 3` seguido de `python run_pipeline.py --sin-preparar`.

**Tabla 8**
*Etapa que detiene cada sabotaje*

| # | Sabotaje | Lo detuvo en la sesión 4 | Lo detiene este pipeline | Evidencia |
|---|---|---|---|---|
| 1 | «cinco» en lugar de un número | Validación | 2 · Validación | `'cinco' no es un número` |
| 2 | Celda vacía en `petal_width` | Validación | 2 · Validación | `es nulo o está vacío` |
| 3 | Registro digitado en milímetros | Validación | 2 · Validación | 4 valores fuera del rango lógico |
| 4 | Especie con código 3 | Validación | 2 · Validación | `la especie 3 no existe` |
| 5 | Faltan todas las flores virginica | Pruebas | 2 · Validación | `no hay ninguna flor de la especie 2 (virginica)` |
| 6 | 40 % de las etiquetas mal asignadas | Pruebas | 4 · Pruebas | Falló `test_validacion_cruzada_minima` |
| 7 | Las flores de prueba copiadas en entrenamiento | Nadie | 2 · Validación | `30 flores de prueba también están en entrenamiento` |

*Nota.* La columna «Lo detuvo en la sesión 4» corresponde al pipeline de referencia de la clase.

Seis de los siete sabotajes se detienen en la validación, antes de entrenar. Dos de ellos pasaban esa etapa en el pipeline de la clase: el 5, que ahora detiene la regla de cobertura de especies, y el 7, que detiene la regla de separación entre entrenamiento y prueba. El sabotaje 7 es una fuga de datos del tipo que Kapoor y Narayanan (2023) clasifican como duplicados entre entrenamiento y prueba: infla la exactitud sin que el modelo sea mejor. El sabotaje 6 no lo puede ver un contrato, porque las etiquetas erradas siguen siendo valores válidos. Lo detuvieron las pruebas, pero no la exactitud: con el 40 % de las etiquetas cambiadas, la exactitud en prueba (0,833) todavía superó el umbral de 0,80; lo rechazó la validación cruzada (0,573 frente al mínimo de 0,90). Un solo umbral no habría bastado. Es la lección de la sesión 4: «un pipeline solo detecta lo que alguien le enseñó a revisar» (Contreras Ortiz, 2026b, diapositiva 26). Los runs saboteados que llegan a entrenar quedan en el experimento `Sabotajes_pipeline`, marcados como `rechazado`.

## 11. Reproducibilidad

El libro guía pide documentar el repositorio de código, las semillas, las versiones de software y sus configuraciones, los datos citados y las instrucciones para ejecutar (Contreras Ortiz et al., 2026, p. 91). La Tabla 9 indica dónde está cada elemento.

**Tabla 9**
*Elementos de reproducibilidad del proyecto*

| Elemento | Dónde está |
|---|---|
| Repositorio de código | Este repositorio; cada run guarda además el commit con el que se entrenó (`git_commit`) |
| Semillas | Partición 42 (`prepare.py`), bosque 42 y validación cruzada 42 (`train.py`) |
| Versiones de software | `requirements.txt`; cada run guarda las versiones de Python, scikit-learn y MLflow |
| Configuración | Argumentos de `run_pipeline.py` y `train.py`, registrados como parámetros y en la etiqueta `comando` |
| Datos citados | Iris (Fisher, 1936), en la copia de scikit-learn 1.9.1 (Pedregosa et al., 2011), con la huella SHA-256 de cada archivo |
| Instrucciones | Secciones 1 y 4 de este README |

La versión de los datos importa. La copia de Iris del repositorio UCI difiere del artículo de Fisher en las muestras 35 y 38 (Fisher, 1988); la de scikit-learn 1.9.1 que usa este proyecto trae los valores del artículo (4,9; 3,1; 1,5; 0,2 y 4,9; 3,6; 1,4; 0,1). Por eso cada run registra la huella de los archivos exactos con los que se entrenó.

El diseño sigue reglas básicas de investigación computacional reproducible: registrar cómo se produjo cada resultado, no manipular datos a mano, guardar las versiones exactas del software, versionar los scripts y anotar las semillas (Sandve et al., 2013). Son los mismos elementos que revisan las listas de verificación de reproducibilidad en aprendizaje automático (Pineau et al., 2021) y la automatización que caracteriza a MLOps (Kreuzberger et al., 2023).

## 12. Decisiones de diseño y limitaciones

**Decisiones:**

- El comando único es `python run_pipeline.py`, como en la sesión 4, porque funciona igual en macOS, Linux y Windows. El `Makefile` solo agrega atajos para macOS y Linux.
- La partición es estratificada (40 flores por especie en entrenamiento y 10 en prueba), a diferencia de la partición no estratificada de la actividad 1.1. Así las tres especies pesan lo mismo en la evaluación.
- Los rangos tienen dos niveles, rechazo y advertencia (sección 5).
- La validación usa la biblioteca estándar; pytest se usa solo para las pruebas automatizadas, igual que en el proyecto de la sesión 4.
- Las claves del contrato siguen el formato del ejemplo de la sesión 4 (`type`, `required`, `nullable`, `minimum`, `maximum`), y las descripciones están en español.

**Limitaciones:**

- Iris tiene 150 flores: la prueba solo tiene 30 y su intervalo de confianza es amplio (sección 9).
- El contrato no puede verificar que una etiqueta sea la correcta (sabotaje 6); eso depende de las pruebas del modelo.
- El rango observado sale de las mismas 150 flores, y la advertencia de deriva compara rangos, no distribuciones.
- MLflow guarda los datos en un SQLite local, pensado para una sola persona; en la actividad 3.1 el registro pasará a un servidor en Docker.
- Se verificó en macOS y Linux. En Windows no aplica el `Makefile`; los pasos de la sección 1 son los de la sesión 4.

## 13. Uso de inteligencia artificial

Usé Claude de Anthropic como asistente. Propuso la estructura del pipeline a partir del proyecto de la sesión 4, generó borradores del código, de las pruebas y de este README, y ayudó a verificar las referencias. Yo ejecuté el pipeline y los sabotajes en mi equipo, comprobé que los resultados coincidieran con los de este documento, revisé y ajusté el código y el texto, y aprobé las decisiones del proyecto: la configuración del modelo, que viene de mi análisis de la actividad 1.1, las reglas del contrato y los umbrales de aprobación.

## Referencias

Anthropic. (2026). *Claude* [Modelo de lenguaje grande]. https://claude.ai

Breck, E., Polyzotis, N., Roy, S., Whang, S. E., y Zinkevich, M. (2019). Data validation for machine learning. *Proceedings of Machine Learning and Systems, 1*. https://proceedings.mlsys.org/paper_files/paper/2019/hash/928f1160e52192e3e0017fb63ab65391-Abstract.html

Contreras Ortiz, M. S. (2026a, 28 de octubre). *Seminario de Grado II – Sesión 2: Baseline reproducible y trazabilidad con MLflow* [Diapositivas de PowerPoint]. Aula virtual, Universidad Santo Tomás.

Contreras Ortiz, M. S. (2026b, 19 de noviembre). *Seminario de Grado II – Sesión 4: Estandarización de pipelines y control de calidad de datos* [Diapositivas de PowerPoint]. Aula virtual, Universidad Santo Tomás.

Contreras Ortiz, M. S., Puerto Moreno, S. A., Suárez Rojas, A. L., y Vela Beltrán, D. A. (2026). *Métodos de investigación en data science*. Ediciones USTA Tunja.

Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems. *Annals of Eugenics, 7*(2), 179–188. https://doi.org/10.1111/j.1469-1809.1936.tb02137.x

Fisher, R. A. (1988). *Iris* [Conjunto de datos]. UCI Machine Learning Repository. https://doi.org/10.24432/C56C76

Kapoor, S., y Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns, 4*(9), Artículo 100804. https://doi.org/10.1016/j.patter.2023.100804

Kreuzberger, D., Kühl, N., y Hirschl, S. (2023). Machine learning operations (MLOps): Overview, definition, and architecture. *IEEE Access, 11*, 31866–31879. https://doi.org/10.1109/ACCESS.2023.3262138

MLOps Open Source - Santoto. (2026). *MLOps Open Source: Hands-on lab*. https://dialvebe.github.io/mlops-course-santoto/

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., y Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.

Pineau, J., Vincent-Lamarre, P., Sinha, K., Larivière, V., Beygelzimer, A., d'Alché-Buc, F., Fox, E., y Larochelle, H. (2021). Improving reproducibility in machine learning research (A report from the NeurIPS 2019 Reproducibility Program). *Journal of Machine Learning Research, 22*(164), 1–20. https://jmlr.org/papers/v22/20-303.html

Sandve, G. K., Nekrutenko, A., Taylor, J., y Hovig, E. (2013). Ten simple rules for reproducible computational research. *PLOS Computational Biology, 9*(10), Artículo e1003285. https://doi.org/10.1371/journal.pcbi.1003285

Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, D., Chaudhary, V., Young, M., Crespo, J.-F., y Dennison, D. (2015). Hidden technical debt in machine learning systems. En C. Cortes, N. D. Lawrence, D. D. Lee, M. Sugiyama y R. Garnett (Eds.), *Advances in Neural Information Processing Systems 28* (pp. 2503–2511). Curran Associates.

Shankar, S., Garcia, R., Hellerstein, J. M., y Parameswaran, A. G. (2024). "We have no idea how models will behave in production until production": How engineers operationalize machine learning. *Proceedings of the ACM on Human-Computer Interaction, 8*(CSCW1), Artículo 206. https://doi.org/10.1145/3653697

Zaharia, M., Chen, A., Davidson, A., Ghodsi, A., Hong, S. A., Konwinski, A., Murching, S., Nykodym, T., Ogilvie, P., Parkhe, M., Xie, F., y Zumar, C. (2018). Accelerating the machine learning lifecycle with MLflow. *IEEE Data Engineering Bulletin, 41*(4), 39–45. http://sites.computer.org/debull/A18dec/p39.pdf
