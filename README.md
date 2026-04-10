# MLOps Capstone Project (Metaflow + MLflow)

This README is for the capstone flow in `flow_starter.py`.
## Setup (pip)

Run these from the capstone directory:

```bash
cd /path/to/OU-22971-Toolbox/MLOps/08_mlops_capstone_project
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install mlflow==3.8 metaflow nannyml pandas numpy scikit-learn pyarrow
```

Start the MLflow UI in a separate terminal:

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

Open: `http://localhost:5000`

## MLflow UI (experiment name)

Open `http://localhost:5000`, click **Experiments**, then open:

```
08_mlops_capstone
```

In this experiment, each Metaflow step logs its own MLflow run with the tag `pipeline_step`. Use the filter box (example: `tags.pipeline_step = model_gate`) and sort by **Start Time** to find the latest run for each step.

## Example runs

Replace `/path/to/...` with your data files.

### 1) Baseline run (no retrain, no promotion)
```bash
python flow_starter.py run \
  --reference-path /path/to/reference.parquet \
  --batch-path /path/to/batch_normal.parquet
```

After this run, in MLflow UI:
- Filter `tags.pipeline_step = model_gate`
  - Confirm tag `retrain_recommended=false`
  - Open `Artifacts/decision.json`
- Filter `tags.pipeline_step = batch_inference`
  - Open `Artifacts/inference/` and view `predictions.parquet` (or `predictions.csv` fallback)

### 2) Retrain + promotion run (automatic within the flow)
```bash
python flow_starter.py run \
  --reference-path /path/to/reference.parquet \
  --batch-path /path/to/batch_degraded.parquet \
  --retrain-threshold 0.05 \
  --min-improvement 0.01
```

After this run, in MLflow UI:
- Filter `tags.pipeline_step = retrain`
  - Metrics: `rmse_candidate`, `rmse_champion`
  - Open `Artifacts/candidate.json`
- Filter `tags.pipeline_step = candidate_acceptance`
  - Confirm tag `promotion_recommended=true`
  - Metrics: `rmse_candidate_ref`, `rmse_champion_ref`, `stability_ok`
  - Open `Artifacts/decision.json`
- Model Registry (left sidebar)
  - Open the model and verify a new version was created and the `@champion` alias moved

### 3) Failure + resumption run (workflow robustness)
Trigger a failure during retrain:
```bash
python flow_starter.py run \
  --reference-path /path/to/reference.parquet \
  --batch-path /path/to/batch_degraded.parquet \
  --fail-step retrain
```

Resume from the failed step:
```bash
python flow_starter.py resume retrain
```

After this run, in MLflow UI:
- Filter `tags.pipeline_step = retrain` (or `candidate_acceptance` if you failed there)
  - Find the **failed** run (status = Failed)
  - After `resume retrain`, find the **new successful** run for the same step
  - Open `Artifacts/decision.json` in the successful run to show the resumed outcome
