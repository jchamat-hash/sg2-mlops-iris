# Atajos para macOS y Linux. En Windows (sin make) se usa directamente: python run_pipeline.py
# Las recetas empiezan con TAB, no con espacios.

PY     ?= python3.13
VENV   ?= .venv
PYTHON := $(VENV)/bin/python
PUERTO ?= 5001
N      ?= 1

.DEFAULT_GOAL := pipeline
.PHONY: pipeline entorno validar pruebas sabotaje sabotajes ui limpiar ayuda

# Pipeline completo con un solo comando; la primera vez también crea el entorno
pipeline: $(VENV)/.instalado
	$(PYTHON) run_pipeline.py

entorno: $(VENV)/.instalado

# Se reinstala solo si cambia requirements.txt
$(VENV)/.instalado: requirements.txt
	$(PY) -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	touch $@

validar: $(VENV)/.instalado
	$(PYTHON) src/validate_data.py

pruebas: $(VENV)/.instalado
	$(PYTHON) -m pytest

# Ejemplo: make sabotaje N=3 (aplica el sabotaje 3 y corre el pipeline sin preparar datos nuevos)
sabotaje: $(VENV)/.instalado
	$(PYTHON) src/prepare.py
	$(PYTHON) romper_datos.py $(N)
	$(PYTHON) run_pipeline.py --sin-preparar

sabotajes: $(VENV)/.instalado
	$(PYTHON) romper_datos.py --todos

# En macOS el puerto 5000 lo ocupa el Receptor AirPlay; por eso 5001
ui: $(VENV)/.instalado
	MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false $(PYTHON) -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port $(PUERTO) --workers 1

limpiar:
	rm -rf data reportes logs mlruns mlflow.db .pytest_cache

ayuda:
	@echo "make                 pipeline completo (crea el entorno .venv la primera vez)"
	@echo "make validar         valida el contrato y los CSV de data/"
	@echo "make pruebas         corre las pruebas del validador y del modelo"
	@echo "make sabotaje N=3    aplica un sabotaje (1 a 7) y corre el pipeline"
	@echo "make sabotajes       corre los siete sabotajes y resume qué etapa detuvo cada uno"
	@echo "make ui              abre MLflow UI en http://127.0.0.1:$(PUERTO)"
	@echo "make limpiar         borra todo lo generado (datos, reportes, logs y MLflow)"
