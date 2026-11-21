PYTHON := .venv/bin/python

.PHONY: collect collect-check collect-metadata j2-prepare j2-mine j2-model j2-report j2
.PHONY: j3-dvc j3-quality j3-monitor j3-api j3

collect:
	$(PYTHON) src/collect.py

collect-check:
	$(PYTHON) src/check_collection.py

collect-metadata:
	$(PYTHON) src/collect.py --metadata-only

j2-prepare:
	$(PYTHON) src/preprocessing/clean_data.py
	$(PYTHON) src/preprocessing/merge_data.py
	$(PYTHON) src/features/build_features.py
	$(PYTHON) src/preprocessing/data_contract.py

j2-mine:
	$(PYTHON) src/mining/clustering.py
	$(PYTHON) src/mining/apriori.py

j2-model:
	$(PYTHON) src/mining/classification.py

j2-report:
	$(PYTHON) src/eda_report.py
	$(PYTHON) src/fairness_report.py

j2:
	$(MAKE) j2-prepare
	$(MAKE) j2-mine
	$(MAKE) j2-model
	$(MAKE) j2-report

j3-dvc:
	.venv/bin/dvc repro

j3-quality:
	$(PYTHON) src/quality_gate.py

j3-monitor:
	$(PYTHON) src/monitoring/drift_report.py

j3-api:
	.venv/bin/bentoml serve src.api.service:SegmentClassifier --host 0.0.0.0 --port 3000

j3:
	$(MAKE) j3-dvc
	$(MAKE) j3-quality
	$(MAKE) j3-monitor
