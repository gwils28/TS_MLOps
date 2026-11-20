# W47 - label-free monitoring of a demand forecaster. `make all` reproduces every result.
PY := uv run python
SHARDS := 4
# LightGBM's OpenMP spin-wait hangs when several processes share the cores (IMPLEMENTATION_NOTES P5)
export OMP_WAIT_POLICY := PASSIVE
export NML_DISABLE_USAGE_LOGGING := 1

.PHONY: all data guard e1 e2 e3 e5 figures report check test lint

all: guard e1 e2 e3 e5 figures report

data:     ## download 66 TLC files (~6 GB, sha256-checked) and rebuild the hourly series
	$(PY) -m w47.cli data

guard:    ## credibility check of the forecaster on the reference period
	$(PY) -m w47.guard

e1:       ## real shock (COVID): estimated vs realized, per phase (daily, then weekly chunks)
	$(PY) -m w47.e1_covid
	$(PY) -c "from w47.e1_covid import run, summarise; from w47.pipeline import TABLES; summarise(run('W')).to_csv(TABLES / 'e1_phases_W.csv', index=False)"

e2:       ## injected scenarios, 440 runs in $(SHARDS) parallel OS processes, then merged
	@for i in $$(seq 0 $$(($(SHARDS)-1))); do \
	  $(PY) -m w47.e2_injection --shard $$i --n-shards $(SHARDS) & \
	done; wait
	$(PY) -m w47.e2_injection --merge --n-shards $(SHARDS)

e3:       ## recall x delay x false alarms
	$(PY) -m w47.e3_detection

e5:       ## exploratory: retraining too early after the shock
	$(PY) -m w47.e5_retrain

figures:
	$(PY) -m w47.figures

report:   ## render report/ into docs/ (Quarto >= 1.6 on PATH)
	cd report && quarto render

test:
	uv run pytest -q

lint:
	uv run ruff check src tests

check: lint test
