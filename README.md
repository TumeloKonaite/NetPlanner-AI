# NetPlanner AI

![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.135%2B-009688?logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![Folium](https://img.shields.io/badge/Folium-interactive%20maps-77B829)
![pytest](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)
![Package Manager](https://img.shields.io/badge/package%20manager-uv-DE5FE9)

NetPlanner AI is a Python machine learning application for prioritizing telecom tower upgrades. It ingests network measurement data, trains a tower-level upgrade risk model, scores towers by upgrade probability, and publishes planning outputs through a FastAPI web interface.

The project produces two main planning artifacts:

- An interactive HTML tower upgrade map powered by Folium.
- A ranked `priority_sites.csv` export for operational planning.

## Features

- Tower-level feature engineering from raw telecom measurements.
- Upgrade risk labelling based on signal strength, download speed, and latency thresholds.
- Model training across Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting classifiers.
- Automated best-model selection using ROC-AUC.
- Reusable prediction pipeline for single-tower or batch scoring.
- Priority site planning with risk levels, upgrade probabilities, and ranked CSV output.
- FastAPI endpoints for map viewing, map regeneration, and CSV export.

## Project Structure

```text
.
|-- main.py                         # FastAPI application
|-- pyproject.toml                  # Project metadata and dependencies
|-- test_pipeline.py                # End-to-end pipeline checks
|-- data/
|   `-- telecom_dataset.csv         # Source telecom measurement data
|-- artifacts/                      # Generated training and model artifacts
|-- notebooks/                      # Exploration notebooks and generated maps
`-- src/
    |-- components/
    |   |-- data_ingestion.py       # Dataset validation and train/test split
    |   |-- data_transformation.py  # Tower aggregation and preprocessing
    |   |-- model_trainer.py        # Model training and evaluation
    |   `-- priority_site_planner.py # Ranked site and map generation
    `-- pipeline/
        |-- train_pipeline.py       # Full model training workflow
        |-- predict_pipeline.py     # Prediction API for trained artifacts
        `-- planning_pipeline.py    # Map and priority CSV workflow
```

## Requirements

- Python 3.12 or later
- `uv` for dependency management
- A source dataset at `data/telecom_dataset.csv`

The input dataset must include these columns:

```text
tower_id, latitude, longitude, city, operator, network_type,
signal_strength_dbm, coverage_quality, download_speed_mbps,
upload_speed_mbps, latency_ms, distance_to_tower_km, indoor_outdoor
```

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/TumeloKonaite/NetPlanner-AI.git
cd NetPlanner-AI
uv sync
```

If `uv` is not available, install it first:

```bash
pip install uv
```

## Usage

### Train the Model

Run the training pipeline to ingest data, transform tower measurements, evaluate models, and save the best model:

```bash
uv run python -m src.pipeline.train_pipeline
```

Generated artifacts are written to `artifacts/`, including:

- `raw_measurements.csv`
- `train.csv`
- `test.csv`
- `preprocessor.pkl`
- `model.pkl`
- `model_metrics.json`
- `schema.json`
- `feature_columns.json`

### Generate Planning Outputs

Run the end-to-end planning workflow:

```bash
uv run python test_pipeline.py
```

This validates the training, prediction, and planning pipelines. It also generates:

- `tower_upgrade_map.html`
- `priority_sites.csv`

### Run the Web Application

Start the FastAPI app:

```bash
uv run uvicorn main:app --reload
```

Open the app at:

```text
http://127.0.0.1:8000
```

Available routes:

- `/` - Landing page with links to planning outputs.
- `/tower-upgrade-map` - Displays the generated tower upgrade map.
- `/generate-tower-upgrade-map` - Regenerates the map and redirects to it.
- `/export-priority-sites` - Generates and downloads the ranked priority sites CSV.

## Modeling Workflow

1. `DataIngestion` validates `data/telecom_dataset.csv`, saves a raw copy, and creates a tower-aware train/test split.
2. `DataTransformation` aggregates raw measurements to tower level and creates the model target.
3. `ModelTrainer` evaluates candidate classifiers with grid search where configured.
4. The best model and preprocessing metadata are saved under `artifacts/`.
5. `PrioritySitePlanner` scores towers, assigns risk levels, ranks sites, and exports map and CSV planning outputs.

The current target rule marks a tower as requiring an upgrade when at least one of these conditions is true:

- Average signal strength is below `-90 dBm`.
- Average download speed is below `20 Mbps`.
- Average latency is above `100 ms`.

The current model features are:

- `upload_speed_mbps`
- `distance_to_tower_km`
- `num_measurements`

## Prediction Example

```python
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

sample = CustomData(
    upload_speed_mbps=8.5,
    distance_to_tower_km=5.2,
    num_measurements=3,
).get_data_as_data_frame()

predictor = PredictPipeline()
prediction = predictor.predict(sample)
probability = predictor.predict_proba(sample)

print(prediction, probability)
```

Run training before using the prediction pipeline so that `artifacts/model.pkl` and `artifacts/preprocessor.pkl` exist.

## Testing

Run the test suite:

```bash
uv run pytest
```

For a direct end-to-end pipeline check:

```bash
uv run python test_pipeline.py
```

## Configuration Notes

- `data/`, `artifacts/`, and `logs/` are ignored by Git because they contain local datasets, generated models, and runtime logs.
- The FastAPI app expects generated planning files in the project root when serving `/tower-upgrade-map` and `/export-priority-sites`.
- If map or model artifacts are missing, the planning pipeline attempts to create the required upstream artifacts automatically.

## License

No license file is currently included. Add a license before distributing or reusing this project outside its current context.
