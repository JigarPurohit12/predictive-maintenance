"""Baselines first, then the model. Every run logged to MLflow.

    python -m training.train                 # every model, val + test
    python -m training.train --models rules logistic xgboost_tuned

The order is deliberate. A gradient-boosted model that barely beats "flag if the
24-hour vibration standard deviation is above its 99th percentile" is a
*finding*, not a failure — but you only know that if you measured the rule.
Reporting the tuned model alone tells you nothing about whether any of the
machinery earned its place.

Five models, which is the phase 3 acceptance criterion:

1. ``rules``        — a single-threshold rule on one feature. No fitting.
2. ``logistic``     — logistic regression on the four raw sensors alone.
3. ``histgb``       — sklearn's HistGradientBoostingClassifier.
4. ``xgboost``      — XGBoost at its defaults.
5. ``xgboost_tuned``— XGBoost with the handful of parameters that matter here.

**On SMOTE.** It is not used, and the ``smote`` ablation exists to show why.
Synthesising a positive by interpolating between two time-ordered rows from
different machines invents a machine-state that never existed, and doing it
before the split leaks across it. ``--models smote`` runs the ablation if
imbalanced-learn is installed; the honest interview answer is the measurement,
not the assertion.

The threshold is chosen on **validation** and then applied unchanged to test.
Picking it on test is how a cost curve turns into a number nobody should
believe.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import Settings, configure_logging, get_settings
from features.build_features import load_matrix
from features.contract import LABEL_COLUMN, RAW_SENSOR_FEATURES, FeatureContract, build_contract
from training.evaluate import (
    ModelResult,
    evaluate_scores,
    format_results,
    plot_cost_curve,
    plot_precision_recall,
    results_table,
    shap_summary,
    write_results,
)
from training.imbalance import cheapest_threshold, scale_pos_weight
from training.splits import split_frames, verify_time_splits

LOGGER = logging.getLogger(__name__)

RESULTS_FILENAME = "phase3_model_comparison.json"
SEED = 7

#: The feature the rules baseline thresholds on. Degrading equipment gets noisy
#: before it gets extreme, so the 24-hour standard deviation of vibration is the
#: single column a maintenance engineer would reach for first.
RULES_FEATURE = "vibration_std_24h"
RULES_PERCENTILE = 99.0


@dataclass
class FittedModel:
    """A trained estimator and how to get probabilities out of it."""

    name: str
    estimator: Any
    predict_proba: Callable[[pd.DataFrame], np.ndarray]
    params: dict[str, Any]
    #: True when SHAP's tree explainer can explain it.
    explainable: bool = False
    #: Which MLflow flavor serialises this estimator. The sklearn flavor now
    #: routes through skops, which refuses to pickle an XGBoost booster, so
    #: XGBoost models have to be logged through their own flavor.
    flavor: str = "sklearn"


# --- Model constructors -----------------------------------------------------


def fit_rules(train: pd.DataFrame, contract: FeatureContract, settings: Settings) -> FittedModel:
    """Flag when one feature exceeds its own training-set percentile.

    Nothing is fitted beyond a quantile, and that quantile is taken on *train*
    only — computing it over the whole dataset would leak the test distribution
    into the baseline and quietly make it look better than it is.
    """
    cutoff = float(np.percentile(train[RULES_FEATURE], RULES_PERCENTILE))

    def score(frame: pd.DataFrame) -> np.ndarray:
        # A rule has no probability, so the "score" is a rank: how far above the
        # cutoff, squashed into [0, 1] so it can share a PR curve with the rest.
        excess = (frame[RULES_FEATURE].to_numpy(dtype="float64") - cutoff) / max(abs(cutoff), 1e-9)
        return 1.0 / (1.0 + np.exp(-excess))

    return FittedModel(
        name="rules",
        estimator=None,
        predict_proba=score,
        params={"feature": RULES_FEATURE, "percentile": RULES_PERCENTILE, "cutoff": cutoff},
    )


def fit_logistic(train: pd.DataFrame, contract: FeatureContract, settings: Settings) -> FittedModel:
    """Logistic regression on the four raw sensors alone.

    The scaler is fitted inside a pipeline on train only. Fitting a scaler on
    the full dataset is leakage — quieter than a random split, and just as real.
    """
    columns = list(RAW_SENSOR_FEATURES)
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2_000,
                    class_weight="balanced",
                    random_state=SEED,
                ),
            ),
        ]
    )
    pipeline.fit(train[columns], train[LABEL_COLUMN])
    return FittedModel(
        name="logistic",
        estimator=pipeline,
        predict_proba=lambda frame: pipeline.predict_proba(frame[columns])[:, 1],
        params={"features": "raw sensors only", "n_features": len(columns), "class_weight": "balanced"},
    )


def fit_histgb(train: pd.DataFrame, contract: FeatureContract, settings: Settings) -> FittedModel:
    columns = list(contract.order)
    weight = scale_pos_weight(train[LABEL_COLUMN])
    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=SEED,
    )
    # HistGB has no scale_pos_weight, so the same idea arrives as sample weights.
    weights = np.where(train[LABEL_COLUMN] == 1, weight, 1.0)
    model.fit(train[columns], train[LABEL_COLUMN], sample_weight=weights)
    return FittedModel(
        name="histgb",
        estimator=model,
        predict_proba=lambda frame: model.predict_proba(frame[columns])[:, 1],
        params={"max_iter": 300, "learning_rate": 0.06, "scale_pos_weight": weight},
    )


def _xgboost(train: pd.DataFrame, contract: FeatureContract, name: str, **overrides: Any) -> FittedModel:
    from xgboost import XGBClassifier

    columns = list(contract.order)
    weight = scale_pos_weight(train[LABEL_COLUMN])
    params: dict[str, Any] = {
        "n_estimators": 300,
        "scale_pos_weight": weight,
        "eval_metric": "aucpr",
        "random_state": SEED,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    params.update(overrides)

    model = XGBClassifier(**params)
    model.fit(train[columns], train[LABEL_COLUMN])
    return FittedModel(
        name=name,
        estimator=model,
        predict_proba=lambda frame: model.predict_proba(frame[columns])[:, 1],
        params=params,
        explainable=True,
        flavor="xgboost",
    )


def fit_xgboost(train: pd.DataFrame, contract: FeatureContract, settings: Settings) -> FittedModel:
    return _xgboost(train, contract, "xgboost")


def fit_xgboost_tuned(train: pd.DataFrame, contract: FeatureContract, settings: Settings) -> FittedModel:
    """The handful of parameters that actually move this problem.

    Shallow trees and heavy subsampling, because the positives are few and the
    features are highly correlated with each other — twelve statistics of the
    same four sensors. Depth is where this overfits first.
    """
    return _xgboost(
        train,
        contract,
        "xgboost_tuned",
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.6,
        min_child_weight=5.0,
        reg_lambda=2.0,
        n_estimators=500,
    )


def fit_smote(train: pd.DataFrame, contract: FeatureContract, settings: Settings) -> FittedModel | None:
    """The ablation, not the recommendation. See the module docstring.

    Returns None when imbalanced-learn is absent, so the ablation is opt-in and
    a missing optional dependency never fails a training run.
    """
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        LOGGER.warning("imbalanced-learn is not installed; skipping the SMOTE ablation.")
        return None

    from xgboost import XGBClassifier

    columns = list(contract.order)
    resampled_x, resampled_y = SMOTE(random_state=SEED).fit_resample(train[columns], train[LABEL_COLUMN])
    LOGGER.info("SMOTE resampled %d rows to %d.", len(train), len(resampled_x))

    model = XGBClassifier(
        n_estimators=300, eval_metric="aucpr", random_state=SEED, n_jobs=-1, tree_method="hist"
    )
    model.fit(resampled_x, resampled_y)
    return FittedModel(
        name="smote",
        estimator=model,
        predict_proba=lambda frame: model.predict_proba(frame[columns])[:, 1],
        params={"resampler": "SMOTE", "resampled_rows": len(resampled_x)},
        explainable=True,
        flavor="xgboost",
    )


MODEL_BUILDERS: dict[str, Callable[[pd.DataFrame, FeatureContract, Settings], FittedModel | None]] = {
    "rules": fit_rules,
    "logistic": fit_logistic,
    "histgb": fit_histgb,
    "xgboost": fit_xgboost,
    "xgboost_tuned": fit_xgboost_tuned,
    "smote": fit_smote,
}

#: What runs when nothing is asked for. SMOTE is excluded on purpose: it is an
#: ablation you run to make a point, not part of the comparison.
DEFAULT_MODELS = ("rules", "logistic", "histgb", "xgboost", "xgboost_tuned")


# --- The run ----------------------------------------------------------------


def _span_hours(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    return float((frame["ts"].max() - frame["ts"].min()).total_seconds() / 3_600.0)


@dataclass
class TrainingRun:
    """Everything one model produced."""

    model: FittedModel
    results: list[ModelResult]
    validation_threshold: float
    artifacts: dict[str, Path]

    @property
    def test_result(self) -> ModelResult | None:
        return next((result for result in self.results if result.split == "test"), None)


def train_one(
    name: str,
    parts: dict[str, pd.DataFrame],
    contract: FeatureContract,
    settings: Settings,
    artifact_dir: Path,
) -> TrainingRun | None:
    """Fit on train, choose the threshold on validation, report on both."""
    builder = MODEL_BUILDERS[name]
    fitted = builder(parts["train"], contract, settings)
    if fitted is None:
        return None
    LOGGER.info("Fitted %s on %d training rows.", name, len(parts["train"]))

    results: list[ModelResult] = []
    artifacts: dict[str, Path] = {}

    validation = parts["val"]
    val_scores = fitted.predict_proba(validation)
    val_result, val_curve = evaluate_scores(
        name,
        "val",
        validation[LABEL_COLUMN],
        val_scores,
        settings,
        n_machines=int(validation["machine_id"].nunique()),
        span_hours=_span_hours(validation),
    )
    results.append(val_result)

    # The threshold is decided here, on validation, and never revisited.
    threshold = (
        settings.decision_threshold if not settings.derive_threshold else cheapest_threshold(val_curve).threshold
    )

    artifacts["cost_curve"] = plot_cost_curve(
        val_curve,
        cheapest_threshold(val_curve),
        f"{name} — expected cost vs threshold (validation)",
        artifact_dir / f"{name}_cost_curve.png",
    )
    artifacts["pr_curve"] = plot_precision_recall(
        np.asarray(validation[LABEL_COLUMN]),
        val_scores,
        f"{name} — precision/recall (validation)",
        artifact_dir / f"{name}_pr_curve.png",
    )

    if "test" in parts and not parts["test"].empty:
        test = parts["test"]
        test_result, _ = evaluate_scores(
            name,
            "test",
            test[LABEL_COLUMN],
            fitted.predict_proba(test),
            settings,
            n_machines=int(test["machine_id"].nunique()),
            span_hours=_span_hours(test),
            threshold=threshold,
            notes="threshold chosen on validation",
        )
        results.append(test_result)

    if fitted.explainable:
        path = shap_summary(
            fitted.estimator, parts["val"], list(contract.order), artifact_dir / f"{name}_shap.png"
        )
        if path is not None:
            artifacts["shap"] = path

    return TrainingRun(model=fitted, results=results, validation_threshold=threshold, artifacts=artifacts)


def log_to_mlflow(run: TrainingRun, contract: FeatureContract, settings: Settings) -> str | None:
    """Params, metrics, the feature order, the split dates and every plot.

    Returns the run id, or None when MLflow is unavailable. Tracking is worth
    having and never worth failing a training run over.
    """
    try:
        import mlflow
    except ImportError:  # pragma: no cover - mlflow is a declared dependency
        LOGGER.warning("mlflow is not installed; skipping tracking.")
        return None

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)

    with mlflow.start_run(run_name=run.model.name) as active:
        mlflow.log_params({f"model.{key}": value for key, value in run.model.params.items()})
        mlflow.log_params(
            {
                "horizon_hours": settings.prediction_horizon_hours,
                "cadence_hours": settings.feature_cadence_hours,
                "exclusion_hours": settings.post_failure_exclusion_hours,
                "gap_hours": settings.gap_hours,
                "window_sizes_hours": ",".join(str(size) for size in settings.window_sizes_hours),
                "train_end_ts": settings.train_end_ts.isoformat(),
                "val_start_ts": settings.val_start_ts.isoformat(),
                "val_end_ts": settings.val_end_ts.isoformat(),
                "test_start_ts": settings.test_start_ts.isoformat(),
                "cost_false_alarm": settings.cost_false_alarm,
                "cost_missed_failure": settings.cost_missed_failure,
                # The hash is what inference asserts against. Losing it means
                # losing the ability to prove a model was fed the right columns.
                "feature_hash": contract.hash,
                "n_features": contract.n_features,
            }
        )
        mlflow.log_param("decision_threshold", run.validation_threshold)
        # A rules baseline has no estimator to serialise. Recording that here
        # is what lets the registry skip it with an explanation rather than
        # failing inside MLflow when it looks for an artifact that was never
        # logged.
        mlflow.log_param("has_model", str(run.model.estimator is not None).lower())
        mlflow.log_param("model_flavor", run.model.flavor)

        for result in run.results:
            prefix = result.split
            mlflow.log_metrics(
                {
                    f"{prefix}.pr_auc": result.pr_auc,
                    f"{prefix}.pr_auc_lift": result.pr_auc_lift,
                    f"{prefix}.roc_auc": result.roc_auc,
                    f"{prefix}.recall": result.recall_at_threshold,
                    f"{prefix}.precision": result.precision_at_threshold,
                    f"{prefix}.expected_cost": result.expected_cost,
                    f"{prefix}.base_rate": result.base_rate,
                    **{f"{prefix}.{key}": value for key, value in result.confusion_matrix.items()},
                    **(
                        {f"{prefix}.recall_at_alarm_budget": result.recall_at_alarm_budget}
                        if result.recall_at_alarm_budget is not None
                        else {}
                    ),
                }
            )

        mlflow.log_dict({"feature_order": list(contract.order), "hash": contract.hash}, "feature_contract.json")
        for path in run.artifacts.values():
            mlflow.log_artifact(str(path))

        if run.model.estimator is not None:
            try:
                flavor = importlib.import_module(f"mlflow.{run.model.flavor}")
                flavor.log_model(run.model.estimator, name="model")
            except Exception as exc:  # noqa: BLE001 - tracking must not fail the run
                LOGGER.warning("Could not log the model artifact: %s", exc)

        return active.info.run_id


def train(
    settings: Settings | None = None,
    model_names: tuple[str, ...] = DEFAULT_MODELS,
    *,
    track: bool = True,
    artifact_dir: Path | None = None,
) -> tuple[list[ModelResult], dict[str, TrainingRun]]:
    """Fit every requested model and report them side by side."""
    settings = settings or get_settings()
    contract = build_contract(settings)

    matrix = load_matrix(settings)
    verify_time_splits(matrix, settings)
    parts = split_frames(matrix)
    for required in ("train", "val"):
        if required not in parts or parts[required].empty:
            raise ValueError(f"The feature matrix has no {required!r} rows; check the split dates in config.py.")

    artifact_dir = artifact_dir or Path("results") / "plots"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[ModelResult] = []
    runs: dict[str, TrainingRun] = {}
    for name in model_names:
        run = train_one(name, parts, contract, settings, artifact_dir)
        if run is None:
            continue
        runs[name] = run
        all_results.extend(run.results)
        if track:
            run_id = log_to_mlflow(run, contract, settings)
            LOGGER.info("Logged %s as MLflow run %s.", name, run_id)

    return all_results, runs


def best_run(runs: dict[str, TrainingRun]) -> str | None:
    """The winner, by validation PR-AUC. Only this one gets registered."""
    scored = {
        name: next((result.pr_auc for result in run.results if result.split == "val"), float("nan"))
        for name, run in runs.items()
    }
    scored = {name: value for name, value in scored.items() if not pd.isna(value)}
    return max(scored, key=lambda name: scored[name]) if scored else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_BUILDERS), default=list(DEFAULT_MODELS))
    parser.add_argument("--no-track", dest="track", action="store_false", help="skip MLflow logging")
    parser.add_argument("--results", default=str(Path("results") / RESULTS_FILENAME))
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings)

    results, runs = train(settings, tuple(args.models), track=args.track)
    print()
    print(format_results(results))

    winner = best_run(runs)
    if winner:
        print(f"\nBest by validation PR-AUC: {winner} (threshold {runs[winner].validation_threshold:.4f})")

    write_results(
        results,
        Path(args.results),
        extra={
            "feature_hash": build_contract(settings).hash,
            "best_model": winner,
            "decision_threshold": runs[winner].validation_threshold if winner else None,
            "cost_false_alarm": settings.cost_false_alarm,
            "cost_missed_failure": settings.cost_missed_failure,
            "table": results_table(results).to_dict(orient="records"),
        },
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
