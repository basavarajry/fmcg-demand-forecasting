.PHONY: install data train api dashboard docker test lint clean

PYTHON ?= python

install:
	$(PYTHON) -m pip install -r requirements.txt
	cp -n .env.example .env 2>/dev/null || true

data:
	$(PYTHON) scripts/download_data.py --dataset synthetic

train:
	$(PYTHON) scripts/train.py --phases baseline ml tft

train-all:
	$(PYTHON) scripts/train.py --phases baseline ml deep tft

api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	streamlit run dashboards/app.py

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src tests
	black --check src tests

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache htmlcov reports/*.html

deploy-ec2:
	bash deployment/aws/deploy_ec2.sh
