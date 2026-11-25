PY ?= python3 -W ignore

all: data features evaluate score scenarios dashboard backtest

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

dashboard:
	cd src && $(PY) export_dashboard.py

backtest:
	cd src && $(PY) closure_backtest.py

# Copy the static pages into the budget analysis site, e.g. make site SITE=../budgetanalysis
SITE ?= ../budgetanalysis
site:
	mkdir -p $(SITE)/closure-risk/track-record
	cp pages/index.html $(SITE)/closure-risk/index.html
	cp pages/track-record/* $(SITE)/closure-risk/track-record/

.PHONY: all data features evaluate score scenarios dashboard backtest site
