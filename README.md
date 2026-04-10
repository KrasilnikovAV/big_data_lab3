# BBC News Classifier with ClickHouse

Проект лабораторной работы по Big Data/ML Ops:
- подготовка данных и обучение классической модели классификации;
- API сервис для инференса;
- запись результатов инференса и наборов данных в ClickHouse;
- тесты;
- DVC stage;
- Docker image;
- CI/CD pipeline на GitHub Actions.

## Структура

- `src/bbc_news/` - код подготовки данных, обучения, API.
- `tests/` - unit/e2e тесты.
- `config.ini` - гиперпараметры и пути.
- `dvc.yaml` - DVC pipeline stage.
- `Dockerfile`, `docker-compose.yml` - контейнеризация.
- `.github/workflows/ci.yml` - CI pipeline (PR в `main`).
- `.github/workflows/cd.yml` - CD/functional tests.
- `scenario.json` - сценарий функционального теста контейнера.
- `dev_sec_ops.yml` - манифест DevSecOps.
- `secrets/clickhouse.vault.yml` - зашифрованные ClickHouse-секреты в формате Ansible Vault.

## Быстрый старт

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

### Секреты (Ansible Vault)

Локальный `.env` и пользовательские `CLICKHOUSE_*`/`ANSIBLE_VAULT_*` переменные больше не используются.

1. Локальный пароль для Vault:

```bash
vi secrets/.vault_pass.txt
chmod 600 secrets/.vault_pass.txt
```

2. Подготовка временного файла с ClickHouse-секретами:

```bash
cat > secrets/clickhouse.secrets.yml <<'EOF'
CLICKHOUSE_USER: <clickhouse_user>
CLICKHOUSE_PASSWORD: <clickhouse_password>
EOF
```

3. Зашифровка `Ansible Vault` и удаление открытого файла:

```bash
./.venv/bin/ansible-vault encrypt secrets/clickhouse.secrets.yml \
  --output secrets/clickhouse.vault.yml \
  --vault-password-file secrets/.vault_pass.txt

rm secrets/clickhouse.secrets.yml
```

### Обучение модели

```bash
python scripts/train_model.py --config config.ini
```

Артефакты сохраняются в `artifacts/`:
- `model.joblib`
- `metrics.json`
- `submission.csv`

### Запуск API

Сервис использует только стандартные пути:
- `secrets/clickhouse.vault.yml`
- `secrets/.vault_pass.txt`

Поэтому локальный запуск выглядит так:

```bash
./.venv/bin/python -m uvicorn bbc_news.api:app --app-dir src --host 0.0.0.0 --port 8000
```

Пример запроса:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"texts":["Stock market gained today","Team won championship"]}'
```

Результаты инференса сохраняются в ClickHouse и доступны через:

```bash
curl http://localhost:8000/predictions?limit=5
```

### Тесты

```bash
PYTHONPATH=src pytest --cov=src --cov-report=term-missing --cov-report=xml
```

### DVC

```bash
dvc repro
```

### Docker

```bash
docker compose up -d --build
docker compose exec -T bbc-news-api python scripts/load_clickhouse_data.py --config /app/config.ini
python scripts/run_scenario.py --scenario scenario.json --base-url http://localhost:8000 --retries 5 --retry-delay 2
docker compose down
```

В `docker-compose` поднимаются три сервиса:
- `vault-bootstrap` - одноразовый контейнер, который запускает `scripts/bootstrap_clickhouse.py`, расшифровывает Ansible Vault и генерирует `clickhouse-user.xml` для ClickHouse.
- `clickhouse` - база данных для хранения результатов модели и наборов train/test.
- `bbc-news-api` - API сервиса модели, читающий ClickHouse-секреты напрямую из Vault по фиксированным путям.

## CI/CD

- CI запускается на `pull_request` в `main`.
- CI выполняет: обучение, тесты, сборку образа и push в DockerHub (если заданы secrets), подпись образа `cosign`, генерацию `dev_sec_ops.yml`.
- CD запускается вручную/по расписанию/после CI, создаёт временный `secrets/.vault_pass.txt` из GitHub Secret `ANSIBLE_VAULT_PASSWORD`, монтирует каталог `secrets` в контейнеры как `/run/secrets`, поднимает `docker compose`, загружает train/test данные в БД и выполняет функциональный сценарий из `scenario.json`.

## Ссылки

- GitHub: https://github.com/KrasilnikovAV/big_data_lab3
- DockerHub: https://hub.docker.com/r/kabeton2/bbc-news-classifier_v3
