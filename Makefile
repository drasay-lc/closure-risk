PY ?= python3 -W ignore

all: data features evaluate score scenarios

data:
	./scripts/fetch_ipeds.sh
	cd src && $(PY) build_panel.py

features:
	cd src && $(PY) features.py

evaluate:
	cd src && $(PY) evaluate.py

score:
	cd src && $(PY) score.py

scenarios:
	cd src && $(PY) scenarios.py

.PHONY: all data features evaluate score scenarios
