"""Capstone flow starter: manual monitoring loop for green taxi tip prediction."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import mlflow
import nannyml as nml
from mlflow.tracking import MlflowClient
from metaflow import FlowSpec, Parameter, step, current
from sklearn.linear_model import LinearRegression

TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "08_mlops_capstone"

RAW_DATETIME_COLS = ["lpep_pickup_datetime", "lpep_dropoff_datetime"]
TARGET_COL = "tip_amount"

EXPECTED_SCHEMA: Dict[str, str] = {
    "ehail_fee": "object",
    "RatecodeID": "float64",
    "store_and_fwd_flag": "object",
    "trip_type": "float64",
    "payment_type": "float64",
    "passenger_count": "float64",
    "congestion_surcharge": "float64",
    "DOLocationID": "int64",
    "PULocationID": "int64",
    "lpep_pickup_datetime": "datetime64[us]",
    "lpep_dropoff_datetime": "datetime64[us]",
    "VendorID": "int64",
    "extra": "float64",
    "fare_amount": "float64",
    "trip_distance": "float64",
    "tolls_amount": "float64",
    "tip_amount": "float64",
    "mta_tax": "float64",
    "total_amount": "float64",
    "improvement_surcharge": "float64",
}

RANGE_SPECS: List[Tuple[str, Optional[float], Optional[float]]] = [
    ("trip_distance", 0.0, 200.0),
    ("fare_amount", 0.0, 500.0),
    ("tip_amount", 0.0, 200.0),
    ("tolls_amount", 0.0, 200.0),
    ("total_amount", 0.0, 1000.0),
    ("passenger_count", 0.0, 10.0),
    ("duration_min", 0.0, 360.0),
]

SOFT_CATEGORICAL_COLS = [
    "store_and_fwd_flag",
    "payment_type",
    "trip_type",
    "RatecodeID",
    "PULocationID",
    "DOLocationID",
]
TIME_FEATURE_COLUMNS = ["pickup_hour", "pickup_dayofweek", "pickup_month"]
FEATURE_COLUMNS = [
    col
    for col in EXPECTED_SCHEMA.keys()
    if col not in set(RAW_DATETIME_COLS + [TARGET_COL])
] + ["duration_min"] + TIME_FEATURE_COLUMNS


def init_mlflow(model_name: str) -> None:
    """Basic MLflow setup for the flow."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    if mlflow.active_run() is not None:
        mlflow.set_tag("model_name", model_name)


@dataclass
class CheckResult:
    metrics: Dict[str, float]
    tables: Dict[str, pd.DataFrame]


def _expected_family(exp: str) -> str:
    exp = str(exp).strip().lower()
    if exp.startswith("datetime64"):
        return "datetime"
    if exp in {"object", "string"}:
        return "string"
    if exp.startswith("int") or exp.startswith("float") or exp in {"number", "numeric"}:
        return "numeric"
    if exp in {"bool", "boolean"}:
        return "bool"
    if exp in {"category"}:
        return "category"
    return "exact"


def _family_ok(actual_dtype: Any, expected: str) -> bool:
    t = pd.api.types
    fam = _expected_family(expected)
    if fam == "datetime":
        return t.is_datetime64_any_dtype(actual_dtype)
    if fam == "numeric":
        return t.is_numeric_dtype(actual_dtype)
    if fam == "string":
        return t.is_object_dtype(actual_dtype) or t.is_string_dtype(actual_dtype) or t.is_categorical_dtype(actual_dtype)
    if fam == "bool":
        return t.is_bool_dtype(actual_dtype)
    if fam == "category":
        return t.is_categorical_dtype(actual_dtype)
    return str(actual_dtype) == str(expected)


def load_taxi_table(path: Path) -> pd.DataFrame:
    """Load TLC-like data from a given path (parquet preferred), normalizing datetime columns if present."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")

    suf = path.suffix.lower()
    if suf == ".parquet":
        try:
            df = pd.read_parquet(path)
        except ImportError as e:
            raise ImportError(
                "Parquet support missing. Install 'pyarrow' (recommended) or 'fastparquet'. "
                "Example: conda install -c conda-forge pyarrow"
            ) from e
    elif suf == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix} (expected .parquet or .csv)")

    for c in RAW_DATETIME_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    return df


def run_integrity_checks(
    df_raw: pd.DataFrame,
    *,
    expected_schema: Optional[Dict[str, str]] = None,
    zone_lookup_path: Optional[Path] = None,
) -> CheckResult:
    """Run cheap schema/range/domain/datetime checks and return loggable tables + scalar metrics."""
    df = df_raw.copy()
    metrics: Dict[str, float] = {}
    tables: Dict[str, pd.DataFrame] = {}

    schema = expected_schema or EXPECTED_SCHEMA
    present_cols = set(df.columns)
    expected_cols = set(schema.keys())

    missing = sorted(expected_cols - present_cols)
    extra = sorted(present_cols - expected_cols)

    dtype_rows: List[Dict[str, Any]] = []
    bad_family = 0
    bad_exact = 0

    for col, exp_dtype in schema.items():
        if col not in df.columns:
            continue
        actual_dtype = df[col].dtype
        actual_str = str(actual_dtype)

        family_ok = _family_ok(actual_dtype, exp_dtype)
        exact_ok = (actual_str == str(exp_dtype))

        if not family_ok:
            bad_family += 1
        if not exact_ok:
            bad_exact += 1

        dtype_rows.append(
            {
                "column": col,
                "expected_dtype": str(exp_dtype),
                "actual_dtype": actual_str,
                "family_ok": bool(family_ok),
                "exact_match": bool(exact_ok),
            }
        )

    tables["schema_presence"] = pd.DataFrame({"missing_column": missing})
    tables["schema_extra_columns"] = pd.DataFrame({"extra_column": extra})
    tables["schema_dtypes"] = pd.DataFrame(
        dtype_rows,
        columns=["column", "expected_dtype", "actual_dtype", "family_ok", "exact_match"],
    )

    metrics["schema_missing_cols"] = float(len(missing))
    metrics["schema_extra_cols"] = float(len(extra))
    metrics["schema_bad_family_dtypes"] = float(bad_family)
    metrics["schema_bad_exact_dtypes"] = float(bad_exact)

    if df.shape[1] == 0:
        tables["missingness"] = pd.DataFrame(
            columns=["dtype", "missing_frac", "missing_count", "n_unique"]
        )
        metrics["missing_frac_mean"] = float("nan")
        metrics["missing_frac_max"] = float("nan")
    else:
        miss = pd.DataFrame(
            {
                "dtype": df.dtypes.astype(str),
                "missing_frac": df.isna().mean(),
                "missing_count": df.isna().sum(),
                "n_unique": df.nunique(dropna=False),
            }
        ).sort_values("missing_frac", ascending=False)
        tables["missingness"] = miss
        metrics["missing_frac_mean"] = float(np.nanmean(df.isna().mean().to_numpy()))
        metrics["missing_frac_max"] = float(np.nanmax(df.isna().mean().to_numpy()))

    dup = int(df.duplicated().sum()) if len(df) else 0
    metrics["duplicate_rows"] = float(dup)
    metrics["duplicate_rows_frac"] = float(dup / max(len(df), 1))

    def bad_frac_num(col: str, lo: Optional[float], hi: Optional[float]) -> Tuple[float, float, float]:
        if col not in df.columns:
            return np.nan, np.nan, np.nan
        x = pd.to_numeric(df[col], errors="coerce")
        valid = x.dropna()
        if valid.empty:
            return 1.0, np.nan, np.nan
        bad = pd.Series(False, index=valid.index)
        if lo is not None:
            bad |= valid < lo
        if hi is not None:
            bad |= valid > hi
        return float(bad.mean()), float(valid.min()), float(valid.max())

    rows: List[Dict[str, Any]] = []
    for col, lo, hi in RANGE_SPECS:
        bf, mn, mx = bad_frac_num(col, lo, hi)
        if not np.isnan(bf):
            rows.append(
                {"column": col, "lo": lo, "hi": hi, "bad_frac": bf, "min": mn, "max": mx}
            )

    if rows:
        rng = pd.DataFrame(rows).sort_values("bad_frac", ascending=False)
        tables["range_checks"] = rng
        metrics["range_worst_bad_frac"] = float(rng["bad_frac"].max())
        metrics["range_any_bad_cols"] = float((rng["bad_frac"] > 0).sum())

    dom_specs: List[Tuple[str, Iterable[Any]]] = [
        ("store_and_fwd_flag", ["Y", "N"]),
        ("payment_type", [1, 2, 3, 4, 5, 6]),
        ("trip_type", [1, 2]),
        ("RatecodeID", [1, 2, 3, 4, 5, 6]),
    ]

    drows: List[Dict[str, Any]] = []
    for col, allowed in dom_specs:
        if col not in df.columns:
            continue
        s = df[col]
        allowed_set = set(allowed)
        bad = ~s.isna() & ~s.isin(allowed_set)
        drows.append(
            {
                "column": col,
                "bad_frac": float(bad.mean()) if len(s) else 0.0,
                "bad_count": int(bad.sum()) if len(s) else 0,
                "n_unique": int(s.nunique(dropna=True)) if len(s) else 0,
            }
        )

    if drows:
        dom = pd.DataFrame(drows).sort_values("bad_frac", ascending=False)
        tables["domain_checks"] = dom
        metrics["domain_worst_bad_frac"] = float(dom["bad_frac"].max())
        metrics["domain_any_bad_cols"] = float((dom["bad_count"] > 0).sum())

    if all(c in df.columns for c in RAW_DATETIME_COLS):
        pickup = pd.to_datetime(df["lpep_pickup_datetime"], errors="coerce")
        dropoff = pd.to_datetime(df["lpep_dropoff_datetime"], errors="coerce")
        dur = (dropoff - pickup).dt.total_seconds() / 60.0

        metrics["duration_neg_frac"] = float((dur < 0).mean()) if len(dur) else 0.0
        metrics["duration_over_6h_frac"] = float((dur > 360).mean()) if len(dur) else 0.0
        metrics["duration_nan_frac"] = float(dur.isna().mean()) if len(dur) else 0.0

        tables["datetime_checks"] = pd.DataFrame(
            [
                {"check": "duration_negative", "frac": metrics["duration_neg_frac"]},
                {"check": "duration_over_6h", "frac": metrics["duration_over_6h_frac"]},
                {"check": "duration_nan", "frac": metrics["duration_nan_frac"]},
            ]
        )

    if zone_lookup_path and Path(zone_lookup_path).exists():
        zones = pd.read_csv(zone_lookup_path)
        if "LocationID" in zones.columns:
            valid_ids = set(
                pd.to_numeric(zones["LocationID"], errors="coerce")
                .dropna()
                .astype(int)
                .tolist()
            )
            for col in ["PULocationID", "DOLocationID"]:
                if col in df.columns:
                    s = pd.to_numeric(df[col], errors="coerce").dropna().astype(int)
                    bad = ~s.isin(valid_ids)
                    metrics[f"{col}_unknown_frac"] = float(bad.mean()) if len(s) else 0.0

    return CheckResult(metrics=metrics, tables=tables)


def _nannyml_result_table(result: Any, columns: List[str]) -> pd.DataFrame:
    ref = result.filter(period="reference").data
    analysis = result.filter(period="analysis").data
    rows: List[Dict[str, Any]] = []
    for col in columns:
        ref_vals = ref[(col, "value")] if (col, "value") in ref.columns else pd.Series(dtype=float)
        analysis_vals = (
            analysis[(col, "value")] if (col, "value") in analysis.columns else pd.Series(dtype=float)
        )
        ref_mean = float(ref_vals.mean()) if len(ref_vals) else float("nan")
        analysis_value = float(analysis_vals.iloc[-1]) if len(analysis_vals) else float("nan")
        diff = (
            analysis_value - ref_mean
            if not (np.isnan(analysis_value) or np.isnan(ref_mean))
            else float("nan")
        )
        upper = (
            float(analysis[(col, "upper_threshold")].iloc[-1])
            if (col, "upper_threshold") in analysis.columns and len(analysis)
            else float("nan")
        )
        lower = (
            float(analysis[(col, "lower_threshold")].iloc[-1])
            if (col, "lower_threshold") in analysis.columns and len(analysis)
            else float("nan")
        )
        alert = (
            bool(analysis[(col, "alert")].iloc[-1])
            if (col, "alert") in analysis.columns and len(analysis)
            else False
        )
        rows.append(
            {
                "column": col,
                "reference_mean": ref_mean,
                "analysis_value": analysis_value,
                "diff": diff,
                "upper_threshold": upper,
                "lower_threshold": lower,
                "alert": alert,
            }
        )
    return pd.DataFrame(rows)


def _soft_integrity_checks(ref: pd.DataFrame, batch: pd.DataFrame) -> CheckResult:
    """Run NannyML data quality checks for missingness and unseen categories."""
    metrics: Dict[str, float] = {}
    tables: Dict[str, pd.DataFrame] = {}

    if nml is None:
        metrics["nannyml_import_error"] = 1.0
        metrics["integrity_warn"] = 1.0
        tables["nannyml_error"] = pd.DataFrame(
            {"error": ["nannyml is not installed. Install it to enable soft gate checks."]}
        )
        return CheckResult(metrics=metrics, tables=tables)

    common_cols = [col for col in ref.columns if col in batch.columns]
    if not common_cols:
        metrics["nannyml_common_cols"] = 0.0
        metrics["integrity_warn"] = 0.0
        return CheckResult(metrics=metrics, tables=tables)

    chunk_size = max(len(batch), 1)

    try:
        missing_calc = nml.MissingValuesCalculator(column_names=common_cols, chunk_size=chunk_size)
        missing_calc.fit(ref[common_cols])
        missing_res = missing_calc.calculate(batch[common_cols])
        miss_tbl = _nannyml_result_table(missing_res, common_cols)
        tables["missingness_spike"] = miss_tbl
        metrics["missingness_spike_max"] = (
            float(np.nanmax(miss_tbl["diff"].to_numpy())) if not miss_tbl.empty else float("nan")
        )
        metrics["missingness_alert_count"] = (
            float(miss_tbl["alert"].sum()) if "alert" in miss_tbl.columns else 0.0
        )
    except Exception as exc:  # pragma: no cover - defensive soft gate
        metrics["missingness_nannyml_error"] = 1.0
        tables["missingness_nannyml_error"] = pd.DataFrame({"error": [str(exc)]})

    cat_cols = [col for col in SOFT_CATEGORICAL_COLS if col in common_cols]
    if cat_cols:
        ref_cat = ref[cat_cols].copy()
        batch_cat = batch[cat_cols].copy()
        for col in cat_cols:
            ref_cat[col] = ref_cat[col].astype("category")
            batch_cat[col] = batch_cat[col].astype("category")
        try:
            unseen_calc = nml.UnseenValuesCalculator(column_names=cat_cols, chunk_size=chunk_size)
            unseen_calc.fit(ref_cat)
            unseen_res = unseen_calc.calculate(batch_cat)
            unseen_tbl = _nannyml_result_table(unseen_res, cat_cols)
            tables["unseen_categories"] = unseen_tbl
            metrics["unseen_alert_count"] = (
                float(unseen_tbl["alert"].sum()) if "alert" in unseen_tbl.columns else 0.0
            )
            metrics["unseen_rate_max"] = (
                float(np.nanmax(unseen_tbl["analysis_value"].to_numpy()))
                if not unseen_tbl.empty
                else float("nan")
            )
        except Exception as exc:  # pragma: no cover - defensive soft gate
            metrics["unseen_nannyml_error"] = 1.0
            tables["unseen_nannyml_error"] = pd.DataFrame({"error": [str(exc)]})
    else:
        metrics["unseen_alert_count"] = 0.0
        metrics["unseen_rate_max"] = float("nan")

    integrity_warn = (
        metrics.get("missingness_alert_count", 0.0) > 0
        or metrics.get("unseen_alert_count", 0.0) > 0
        or metrics.get("missingness_nannyml_error", 0.0) > 0
        or metrics.get("unseen_nannyml_error", 0.0) > 0
        or metrics.get("nannyml_import_error", 0.0) > 0
    )
    metrics["integrity_warn"] = float(integrity_warn)
    metrics["retrain_boost"] = 1.0 if integrity_warn else 0.0

    return CheckResult(metrics=metrics, tables=tables)


def _hard_fail_reasons(batch: pd.DataFrame, checks: CheckResult) -> List[str]:
    reasons: List[str] = []
    if checks.metrics.get("schema_missing_cols", 0.0) > 0:
        reasons.append("missing_required_columns")
    if checks.metrics.get("schema_bad_family_dtypes", 0.0) > 0:
        reasons.append("invalid_dtypes")
    if checks.metrics.get("duration_neg_frac", 0.0) > 0:
        reasons.append("negative_duration")
    if checks.metrics.get("range_worst_bad_frac", 0.0) > 0:
        reasons.append("out_of_range_values")
    if "tip_amount" not in batch.columns or batch["tip_amount"].dropna().empty:
        reasons.append("target_missing")
    return reasons


def _build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    # Derive time-based features from pickup/dropoff timestamps.
    if all(c in data.columns for c in RAW_DATETIME_COLS):
        pickup = pd.to_datetime(data["lpep_pickup_datetime"], errors="coerce")
        dropoff = pd.to_datetime(data["lpep_dropoff_datetime"], errors="coerce")
        data["duration_min"] = (dropoff - pickup).dt.total_seconds() / 60.0
        data["pickup_hour"] = pickup.dt.hour
        data["pickup_dayofweek"] = pickup.dt.dayofweek
        data["pickup_month"] = pickup.dt.month
    else:
        data["duration_min"] = np.nan
        data["pickup_hour"] = np.nan
        data["pickup_dayofweek"] = np.nan
        data["pickup_month"] = np.nan

    # Ensure every expected feature column exists in the table.
    for col in FEATURE_COLUMNS:
        if col not in data.columns:
            data[col] = np.nan

    features = data[FEATURE_COLUMNS].copy()

    # Normalize categorical vs numeric columns with a simple fill policy.
    for col in FEATURE_COLUMNS:
        if col in SOFT_CATEGORICAL_COLS:
            features[col] = features[col].astype("object").fillna("missing")
        else:
            features[col] = pd.to_numeric(features[col], errors="coerce").fillna(0.0)

    return features


class MLFlowCapstoneFlow(FlowSpec):
    reference_path = Parameter("reference-path")
    batch_path = Parameter("batch-path")
    model_name = Parameter("model-name", default="green_taxi_tip_model")
    retrain_threshold = Parameter("retrain-threshold", default=0.2)
    min_improvement = Parameter("min-improvement", default=0.01)
    stability_max_regress_pct = Parameter("stability-max-regress-pct", default=0.02)
    block_on_integrity_warn = Parameter("block-on-integrity-warn", default=False)
    fail_step = Parameter("fail-step", default="")

    def _maybe_fail(self, step_name: str) -> None:
        # Optional failure injection for the resumption demo.
        if getattr(current, "origin_run_id", None):
            return
        if str(self.fail_step).strip().lower() == step_name:
            raise RuntimeError(f"Intentional failure for resumption demo at step: {step_name}")

    @step
    def start(self):
        init_mlflow(self.model_name)
        self.next(self.load_data)

    @step  # A
    def load_data(self):
        self.ref, self.batch = load_taxi_table(self.reference_path), load_taxi_table(self.batch_path)
        self.next(self.integrity_gate)

    @step  # B
    def integrity_gate(self):
        init_mlflow(self.model_name)
        with mlflow.start_run(run_name="integrity_gate"):
            mlflow.set_tag("pipeline_step", "integrity_gate")
            mlflow.set_tag("model_name", self.model_name)

            # Hard checks can reject the batch immediately.
            hard = run_integrity_checks(self.batch)
            for name, table in hard.tables.items():
                mlflow.log_table(table, artifact_file=f"integrity/hard/{name}.json")
            mlflow.log_metrics({f"hard_{k}": v for k, v in hard.metrics.items()})

            # Soft checks (NannyML) only raise warnings and inform retrain likelihood.
            soft = _soft_integrity_checks(self.ref, self.batch)
            for name, table in soft.tables.items():
                mlflow.log_table(table, artifact_file=f"integrity/soft/{name}.json")
            mlflow.log_metrics({f"soft_{k}": v for k, v in soft.metrics.items()})

            reasons = _hard_fail_reasons(self.batch, hard)
            ok = len(reasons) == 0
            integrity_warn = soft.metrics.get("integrity_warn", 0.0) > 0
            retrain_boost = soft.metrics.get("retrain_boost", 0.0)

            # Log a compact decision summary for auditability.
            decision = {
                "action": "continue" if ok else "reject_batch",
                "reasons": reasons,
                "integrity_warn": integrity_warn,
                "retrain_boost": retrain_boost,
            }
            mlflow.log_dict(decision, "decision.json")
            mlflow.set_tag("integrity_warn", str(integrity_warn).lower())
            mlflow.set_tag("integrity_ok", str(ok).lower())

        self.integrity_ok = ok
        self.integrity_warn = integrity_warn
        self.integrity_reasons = reasons
        self.retrain_boost = retrain_boost
        self.next({True: self.feature_engineering, False: self.end}, condition="integrity_ok")

    @step  # C
    def feature_engineering(self):
        init_mlflow(self.model_name)
        with mlflow.start_run(run_name="feature_engineering"):
            mlflow.set_tag("pipeline_step", "feature_engineering")
            mlflow.set_tag("model_name", self.model_name)

            self.ref_features = _build_feature_table(self.ref)
            self.batch_features = _build_feature_table(self.batch)

            self.ref_target = (
                pd.to_numeric(self.ref[TARGET_COL], errors="coerce")
                if TARGET_COL in self.ref.columns
                else None
            )
            self.batch_target = (
                pd.to_numeric(self.batch[TARGET_COL], errors="coerce")
                if TARGET_COL in self.batch.columns
                else None
            )

            feature_spec = {
                "feature_columns": FEATURE_COLUMNS,
                "dtypes": {col: str(self.ref_features[col].dtype) for col in FEATURE_COLUMNS},
                "fill_policy": {"categorical": "missing", "numeric": 0.0},
            }
            mlflow.log_dict(feature_spec, "feature_spec.json")
            mlflow.log_table(
                pd.DataFrame(
                    [
                        {"column": col, "dtype": str(self.ref_features[col].dtype)}
                        for col in FEATURE_COLUMNS
                    ]
                ),
                artifact_file="features/schema.json",
            )
            mlflow.log_metrics(
                {
                    "feature_count": float(len(FEATURE_COLUMNS)),
                    "ref_rows": float(len(self.ref_features)),
                    "batch_rows": float(len(self.batch_features)),
                }
            )

        self.next(self.load_champion)

    @step  # D
    def load_champion(self):
        init_mlflow(self.model_name)
        with mlflow.start_run(run_name="load_champion"):
            mlflow.set_tag("pipeline_step", "load_champion")
            mlflow.set_tag("model_name", self.model_name)

            client = MlflowClient()
            model_name = self.model_name

            try:
                champion = client.get_model_version_by_alias(model_name, "champion")
                champion_uri = f"models:/{model_name}@champion"
                self.champion_model_uri = champion_uri
                self.champion_version = champion.version
                self.champion_model = mlflow.pyfunc.load_model(champion_uri)
                self.champion_feature_cols = None
                self.champion_bootstrapped = False
                mlflow.set_tag("champion_loaded", "true")
                mlflow.log_dict(
                    {"action": "load_champion", "model_uri": champion_uri, "version": champion.version},
                    "champion.json",
                )
            except Exception:
                if self.ref_target is None or self.ref_target.dropna().empty:
                    raise ValueError("Bootstrap requires a non-empty target column.")

                y = self.ref_target
                valid = ~y.isna()
                X_train = self.ref_features.loc[valid].select_dtypes(include=[np.number])
                y_train = y.loc[valid]
                if X_train.empty:
                    raise ValueError("No numeric features available for bootstrap training.")

                model = LinearRegression()
                model.fit(X_train, y_train)

                mlflow.sklearn.log_model(model, "model")
                model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
                result = mlflow.register_model(model_uri, model_name)
                client.set_registered_model_alias(model_name, "champion", result.version)
                client.set_model_version_tag(model_name, result.version, "role", "champion")
                client.set_model_version_tag(model_name, result.version, "promotion_reason", "bootstrap")

                self.champion_model_uri = f"models:/{model_name}@champion"
                self.champion_version = result.version
                self.champion_model = model
                self.champion_feature_cols = X_train.columns.tolist()
                self.champion_bootstrapped = True
                mlflow.set_tag("champion_loaded", "false")
                mlflow.log_dict(
                    {
                        "action": "bootstrap_champion",
                        "model_uri": model_uri,
                        "version": result.version,
                        "feature_columns": X_train.columns.tolist(),
                    },
                    "champion.json",
                )

        self.next(self.model_gate)

    @step  # E
    def model_gate(self):
        init_mlflow(self.model_name)
        with mlflow.start_run(run_name="model_gate"):
            mlflow.set_tag("pipeline_step", "model_gate")
            mlflow.set_tag("model_name", self.model_name)
            mlflow.log_params(
                {
                    "retrain_threshold": float(self.retrain_threshold),
                    "min_improvement": float(self.min_improvement),
                    "stability_max_regress_pct": float(self.stability_max_regress_pct),
                    "block_on_integrity_warn": bool(self.block_on_integrity_warn),
                }
            )

            # If labels are missing, skip evaluation and downstream retrain logic.
            y = self.batch_target
            if y is None or y.dropna().empty:
                mlflow.set_tag("eval_available", "false")
                mlflow.log_metrics(
                    {
                        "rmse_champion": float("nan"),
                        "rmse_baseline": float("nan"),
                        "rmse_increase_pct": float("nan"),
                    }
                )
                mlflow.log_dict(
                    {"retrain_needed": False, "retrain_reason": "missing_target"},
                    "decision.json",
                )
                self.retrain_needed = False
                self.retrain_reason = "missing_target"
                self.rmse_champion = float("nan")
                self.rmse_baseline = float("nan")
                self.rmse_increase_pct = float("nan")
                self.next(self.batch_inference)
                return

            y = y.astype(float)
            X_eval = self.batch_features.copy()
            if self.champion_feature_cols:
                X_eval = X_eval[self.champion_feature_cols]

            try:
                preds = self.champion_model.predict(X_eval)
            except Exception:
                X_eval = X_eval.select_dtypes(include=[np.number])
                preds = self.champion_model.predict(X_eval)

            preds = np.asarray(preds, dtype=float)
            rmse_champion = float(np.sqrt(np.nanmean((preds - y.to_numpy()) ** 2)))

            baseline_value = (
                float(self.ref_target.dropna().mean())
                if self.ref_target is not None and not self.ref_target.dropna().empty
                else float(y.dropna().mean())
            )
            rmse_baseline = float(np.sqrt(np.nanmean((baseline_value - y.to_numpy()) ** 2)))
            rmse_increase_pct = (
                float((rmse_champion - rmse_baseline) / rmse_baseline)
                if rmse_baseline > 0
                else float("nan")
            )

            mlflow.log_metrics(
                {
                    "rmse_champion": rmse_champion,
                    "rmse_baseline": rmse_baseline,
                    "rmse_increase_pct": rmse_increase_pct,
                }
            )

            # Trigger retraining on degradation or integrity warnings.
            retrain_needed = bool(
                (not np.isnan(rmse_increase_pct) and rmse_increase_pct > float(self.retrain_threshold))
                or self.retrain_boost > 0
            )
            if retrain_needed:
                retrain_reason = (
                    "rmse_increase_pct"
                    if not np.isnan(rmse_increase_pct)
                    and rmse_increase_pct > float(self.retrain_threshold)
                    else "integrity_warn"
                )
            else:
                retrain_reason = "no_degradation"

            # Baseline bootstrap run should not retrain.
            if getattr(self, "champion_bootstrapped", False):
                retrain_needed = False
                retrain_reason = "no_degradation"

            mlflow.set_tag("retrain_recommended", str(retrain_needed).lower())
            mlflow.log_dict(
                {"retrain_needed": retrain_needed, "retrain_reason": retrain_reason},
                "decision.json",
            )

        self.retrain_needed = retrain_needed
        self.retrain_reason = retrain_reason
        self.rmse_champion = rmse_champion
        self.rmse_baseline = rmse_baseline
        self.rmse_increase_pct = rmse_increase_pct
        self.next({True: self.retrain, False: self.batch_inference}, condition="retrain_needed")

    @step  # F
    def retrain(self):
        # Trains a new candidate model from reference data.
        init_mlflow(self.model_name)
        with mlflow.start_run(run_name="retrain"):
            mlflow.set_tag("pipeline_step", "retrain")
            mlflow.set_tag("model_name", self.model_name)
            mlflow.log_param("trained_on_batches", Path(self.reference_path).name)
            mlflow.log_param("eval_batch_id", Path(self.batch_path).name)
            self._maybe_fail("retrain")

            if self.ref_target is None or self.ref_target.dropna().empty:
                raise ValueError("Retrain requires a non-empty target column.")

            # Expand the training window by including labeled batch data when available.
            y_ref = self.ref_target
            valid_ref = ~y_ref.isna()
            X_ref = self.ref_features.loc[valid_ref].select_dtypes(include=[np.number])
            y_ref = y_ref.loc[valid_ref]

            X_train = X_ref
            y_train = y_ref

            if self.batch_target is not None and not self.batch_target.dropna().empty:
                y_batch = self.batch_target
                valid_batch = ~y_batch.isna()
                X_batch = self.batch_features.loc[valid_batch].select_dtypes(include=[np.number])
                y_batch = y_batch.loc[valid_batch]
                for col in X_ref.columns:
                    if col not in X_batch.columns:
                        X_batch[col] = 0.0
                X_batch = X_batch[X_ref.columns]
                batch_weight = 5
                X_train = pd.concat([X_ref] + [X_batch] * batch_weight, ignore_index=True)
                y_train = pd.concat([y_ref] + [y_batch] * batch_weight, ignore_index=True)
            if X_train.empty:
                raise ValueError("No numeric features available for retraining.")
            self.candidate_feature_cols = X_train.columns.tolist()

            model = LinearRegression()
            model.fit(X_train, y_train)

            X_eval = self.batch_features.copy()
            for col in X_train.columns:
                if col not in X_eval.columns:
                    X_eval[col] = 0.0
            X_eval = X_eval[X_train.columns]
            preds = model.predict(X_eval)
            preds = np.asarray(preds, dtype=float)
            y_eval = self.batch_target.astype(float)
            rmse_candidate = float(np.sqrt(np.nanmean((preds - y_eval.to_numpy()) ** 2)))

            mlflow.log_metric("rmse_candidate", rmse_candidate)
            mlflow.log_metric("rmse_champion", self.rmse_champion)

            mlflow.sklearn.log_model(model, "model")
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            result = mlflow.register_model(model_uri, self.model_name)

            client = MlflowClient()
            client.set_model_version_tag(self.model_name, result.version, "role", "candidate")
            client.set_model_version_tag(self.model_name, result.version, "validation_status", "pending")
            client.set_model_version_tag(self.model_name, result.version, "decision_reason", self.retrain_reason)
            client.set_model_version_tag(self.model_name, result.version, "trained_on_batches", Path(self.reference_path).name)
            client.set_model_version_tag(self.model_name, result.version, "eval_batch_id", Path(self.batch_path).name)

            self.candidate_version = result.version
            self.candidate_model_uri = f"models:/{self.model_name}/{result.version}"
            self.candidate_rmse = rmse_candidate

            mlflow.log_dict(
                {
                    "action": "train_candidate",
                    "model_uri": model_uri,
                    "version": result.version,
                    "feature_columns": self.candidate_feature_cols,
                },
                "candidate.json",
            )

        self.next(self.candidate_acceptance)

    @step  # G
    def candidate_acceptance(self):
        # accept/reject candidate based on stability and integrity checks.
        init_mlflow(self.model_name)
        with mlflow.start_run(run_name="candidate_acceptance"):
            mlflow.set_tag("pipeline_step", "candidate_acceptance")
            mlflow.set_tag("model_name", self.model_name)
            mlflow.log_params(
                {
                    "min_improvement": float(self.min_improvement),
                    "stability_max_regress_pct": float(self.stability_max_regress_pct),
                    "block_on_integrity_warn": bool(self.block_on_integrity_warn),
                }
            )
            self._maybe_fail("candidate_acceptance")

            client = MlflowClient()
            promote = False # do not promote yet
            reason = "no_improvement" # no improvement reason
            integrity_block = bool(self.block_on_integrity_warn and self.integrity_warn) #if integrity warnings + block flag, promotion is blocked.

            # Stability check on reference data
            rmse_candidate_ref = float("nan")
            rmse_champion_ref = float("nan")
            stability_ok = True 
            if self.ref_target is not None and not self.ref_target.dropna().empty:
                y_ref = self.ref_target.astype(float)
                candidate_model = mlflow.pyfunc.load_model(self.candidate_model_uri) # load candidate model from uri
                champion_model = mlflow.pyfunc.load_model(f"models:/{self.model_name}@champion") # load champion model from uri
                # -- compute RMSE of candidate model on reference data --#
                X_ref_candidate = self.ref_features.copy() # copy reference features
                if getattr(self, "candidate_feature_cols", None):
                    X_ref_candidate = X_ref_candidate[self.candidate_feature_cols] # filter candidate features
                try:
                    cand_preds = candidate_model.predict(X_ref_candidate)
                except Exception:
                    cand_preds = candidate_model.predict(X_ref_candidate.select_dtypes(include=[np.number]))
                cand_preds = np.asarray(cand_preds, dtype=float)
                rmse_candidate_ref = float(np.sqrt(np.nanmean((cand_preds - y_ref.to_numpy()) ** 2))) # compute RMSE of candidate model on reference data

                # -- compute RMSE of champion model on reference data --#
                X_ref_champion = self.ref_features.copy() 
                if getattr(self, "champion_feature_cols", None):
                    X_ref_champion = X_ref_champion[self.champion_feature_cols]
                try:
                    champ_preds = champion_model.predict(X_ref_champion)
                except Exception:
                    champ_preds = champion_model.predict(X_ref_champion.select_dtypes(include=[np.number]))
                champ_preds = np.asarray(champ_preds, dtype=float)
                rmse_champion_ref = float(np.sqrt(np.nanmean((champ_preds - y_ref.to_numpy()) ** 2)))

                # -- check stability --#
                stability_ok = rmse_candidate_ref <= rmse_champion_ref * (1 + float(self.stability_max_regress_pct))

            mlflow.log_metrics(
                {
                    "rmse_candidate_ref": rmse_candidate_ref,
                    "rmse_champion_ref": rmse_champion_ref,
                    "stability_ok": float(stability_ok),
                }
            )

            # Promotion requires improvement and no integrity block.
            if integrity_block: # blocked due to data issues
                promote = False
                reason = "integrity_warn_blocked"
            elif self.candidate_rmse is None or np.isnan(self.candidate_rmse): # candidate metric missing
                promote = False
                reason = "candidate_metric_missing"
            elif self.rmse_champion is None or np.isnan(self.rmse_champion): # champion metric missing (no baseline to compare)
                promote = True
                reason = "no_champion_metric"
            else:
                improvement = (self.rmse_champion - self.candidate_rmse) / max(self.rmse_champion, 1e-12) # compute improvement percentage
                promote = improvement > float(self.min_improvement) # if improvement is greater than min improvement, promote
                reason = "improved_rmse" if promote else "insufficient_improvement"

            if promote and not stability_ok: # if promote and not stable, do not promote
                promote = False
                reason = "stability_regression"

            if promote:
                previous_champion = getattr(self, "champion_version", None)# get previous champion version
                client.set_registered_model_alias(self.model_name, "champion", self.candidate_version)# set champion alias to candidate version
                client.set_model_version_tag(self.model_name, self.candidate_version, "role", "champion")
                client.set_model_version_tag(self.model_name, self.candidate_version, "validation_status", "approved")
                client.set_model_version_tag(self.model_name, self.candidate_version, "promotion_reason", reason)
                client.set_model_version_tag(self.model_name,self.candidate_version,"promoted_at", datetime.now(timezone.utc).isoformat(),)
                if previous_champion is not None:
                    client.set_model_version_tag(self.model_name, previous_champion, "role", "previous_champion")
                    client.set_model_version_tag(self.model_name,previous_champion,"demoted_at", datetime.now(timezone.utc).isoformat(),)
            else:
                client.set_model_version_tag(self.model_name, self.candidate_version, "validation_status", "rejected")

            mlflow.set_tag("promotion_recommended", str(promote).lower())
            mlflow.log_dict(
                {
                    "promotion_recommended": promote,
                    "promotion_reason": reason,
                    "candidate_version": self.candidate_version,
                    "rmse_candidate": self.candidate_rmse,
                    "rmse_champion": self.rmse_champion,
                    "rmse_candidate_ref": rmse_candidate_ref,
                    "rmse_champion_ref": rmse_champion_ref,
                    "stability_ok": stability_ok,
                    "integrity_warn": bool(self.integrity_warn),
                    "block_on_integrity_warn": bool(self.block_on_integrity_warn),
                    "min_improvement": float(self.min_improvement),
                },
                "decision.json",
            )

        self.next(self.batch_inference)

    @step  # Inference demo artifact
    def batch_inference(self):
        init_mlflow(self.model_name)
        with mlflow.start_run(run_name="batch_inference"):
            mlflow.set_tag("pipeline_step", "batch_inference")
            mlflow.set_tag("model_name", self.model_name)

            # Log batch predictions for the required demo artifact.
            model_uri = f"models:/{self.model_name}@champion"
            model = mlflow.pyfunc.load_model(model_uri)

            X_pred = self.batch_features.copy()
            if getattr(self, "champion_feature_cols", None):
                X_pred = X_pred[self.champion_feature_cols]

            try:
                preds = model.predict(X_pred)
            except Exception:
                X_pred = X_pred.select_dtypes(include=[np.number])
                preds = model.predict(X_pred)

            preds = np.asarray(preds, dtype=float)
            out = pd.DataFrame({"prediction": preds})
            if self.batch_target is not None:
                out["actual_tip_amount"] = pd.to_numeric(self.batch_target, errors="coerce")

            output_path = Path("predictions.parquet")
            try:
                out.to_parquet(output_path, index=False)
                mlflow.log_artifact(str(output_path), artifact_path="inference")
                mlflow.set_tag("predictions_format", "parquet")
            except Exception as exc:
                fallback = Path("predictions.csv")
                out.to_csv(fallback, index=False)
                mlflow.log_artifact(str(fallback), artifact_path="inference")
                mlflow.set_tag("predictions_format", "csv")
                mlflow.log_dict({"parquet_error": str(exc)}, "inference/predictions_error.json")

        self.next(self.end)

    @step
    def end(self):
        # Terminal step for reject_batch or final flow completion.
        pass


if __name__ == "__main__":
    MLFlowCapstoneFlow()
