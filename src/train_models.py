"""Train HW1 classifiers using only train-set cross-validation for selection.

Fixed grid: StandardScaler -> L2 Normalizer -> logistic regression or RBF SVC.
Official validation labels are used exclusively for reporting after model and
feature-route selection. Test labels are never used. No additional audio data
are needed. References: https://scikit-learn.org/stable/modules/pipeline.html
and https://scikit-learn.org/stable/modules/cross_validation.html .
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import os
from pathlib import Path
import platform
import time
from typing import Any
import warnings

# Must precede numerical imports for both direct execution and subprocess use.
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_variable] = "1"

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

try:
    from .model_utils import (
        CLASS_LABELS, aligned_probabilities, assert_feature_alignment,
        classification_metrics, load_feature_archive, rank_labels,
        save_bundle, select_features, write_json,
    )
except ImportError:
    from model_utils import (
        CLASS_LABELS, aligned_probabilities, assert_feature_alignment,
        classification_metrics, load_feature_archive, rank_labels,
        save_bundle, select_features, write_json,
    )


SEED = 42
N_SPLITS = 3
SELECTION_POLICY = {
    "data": "official train split only",
    "cross_validation": "3-fold StratifiedKFold(shuffle=True, random_state=42)",
    "objective": "OOF Top-1 accuracy + 0.5 * OOF Top-3 accuracy",
    "ties": "first candidate in the predeclared grid; routes classical, mert, concat",
    "scaling": "StandardScaler fitted inside each training fold, followed by per-recording L2 normalization",
    "validation": "reporting only; never used for candidate, hyperparameter, or route selection",
    "test": "prediction only; no test labels inspected or used",
}


def candidate_grid() -> list[dict[str, Any]]:
    return [
        {"candidate_id": f"logreg_C{C:g}", "model_type": "logistic_regression", "parameters": {"C": C, "max_iter": 5000, "random_state": SEED, "solver": "lbfgs"}}
        for C in (0.1, 1.0, 10.0, 100.0)
    ] + [
        {"candidate_id": f"svc_rbf_C{C:g}", "model_type": "svc_rbf", "parameters": {"C": C, "kernel": "rbf", "gamma": "scale", "probability": True, "random_state": SEED}}
        for C in (1.0, 10.0, 100.0)
    ]


def make_pipeline(candidate: dict[str, Any]) -> Pipeline:
    if candidate["model_type"] == "logistic_regression":
        classifier = LogisticRegression(**candidate["parameters"])
    elif candidate["model_type"] == "svc_rbf":
        classifier = SVC(**candidate["parameters"])
    else:
        raise ValueError("Unknown candidate model type")
    return Pipeline([
        ("standardize", StandardScaler()),
        ("l2_normalize", Normalizer(norm="l2")),
        ("classifier", classifier),
    ])


def fit_with_warnings(pipeline: Pipeline, X: np.ndarray, labels: np.ndarray) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipeline.fit(X, labels)
    messages = [str(warning.message) for warning in caught]
    for message in messages:
        print(f"  Fit warning: {message}", flush=True)
    return messages


def cross_validate_route(
    X: np.ndarray,
    labels: np.ndarray,
    classes: list[str],
    folds: list[tuple[np.ndarray, np.ndarray]],
    dataset: str,
    route: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for candidate in candidate_grid():
        started = time.monotonic()
        oof = np.empty((len(labels), len(classes)), dtype=np.float64)
        visits = np.zeros(len(labels), dtype=int)
        fold_metrics = []
        fit_warnings: list[str] = []
        for fold_index, (fit_indices, heldout_indices) in enumerate(folds):
            pipeline = make_pipeline(candidate)
            fit_warnings.extend(fit_with_warnings(pipeline, X[fit_indices], labels[fit_indices]))
            probabilities = aligned_probabilities(pipeline, X[heldout_indices], classes)
            oof[heldout_indices] = probabilities
            visits[heldout_indices] += 1
            fold_metrics.append({
                "fold": fold_index,
                "fitting_count": len(fit_indices),
                **classification_metrics(labels[heldout_indices], probabilities, classes),
            })
            print(
                f"CV {dataset}/{route}/{candidate['candidate_id']} fold {fold_index + 1}/{N_SPLITS}: "
                f"top1={fold_metrics[-1]['top1_accuracy']:.4f} top3={fold_metrics[-1]['top3_accuracy']:.4f}",
                flush=True,
            )
        if not np.all(visits == 1):
            raise RuntimeError("Each training recording must receive exactly one held-out prediction")
        metrics = classification_metrics(labels, oof, classes)
        summary = {
            **candidate,
            "feature_route": route,
            "feature_dimension": X.shape[1],
            "oof_metrics": metrics,
            "cv_selection_score": metrics["selection_score"],
            "fold_metrics": fold_metrics,
            "fit_warnings": fit_warnings,
            "elapsed_seconds": time.monotonic() - started,
        }
        summaries.append(summary)
        if best is None or summary["cv_selection_score"] > best["cv_selection_score"]:
            best = summary
    assert best is not None
    return summaries, best


def prediction_rows(ids: np.ndarray, probabilities: np.ndarray, classes: list[str], labels: np.ndarray | None = None) -> list[dict[str, Any]]:
    ranked = rank_labels(probabilities, classes)
    rows = []
    for index, (sample_id, top3) in enumerate(zip(ids, ranked)):
        row = {
            "sample_id": str(sample_id),
            "scores": probabilities[index].tolist(),
            "top1": top3[0],
            "top3": top3,
        }
        if labels is not None:
            truth = str(labels[index])
            row.update({"true_label": truth, "correct_top1": truth == top3[0], "correct_top3": truth in top3})
        rows.append(row)
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def train(classical_path: Path, mert_path: Path | None, output_dir: Path, model_dir: Path) -> dict[str, Any]:
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    classical = load_feature_archive(classical_path)
    mert = load_feature_archive(mert_path) if mert_path else None
    if mert is not None:
        assert_feature_alignment(classical, mert)
    observed_datasets = set(classical["datasets"].tolist())
    if observed_datasets != set(CLASS_LABELS):
        raise ValueError(f"Both datasets A and B are required, observed {observed_datasets}")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    routes = ["classical"] if mert is None else ["classical", "mert", "concat"]
    feature_dimensions = {"classical": classical["X"].shape[1]}
    feature_versions = {"classical": classical["feature_version"]}
    if mert is not None:
        feature_dimensions["mert"] = mert["X"].shape[1]
        feature_versions["mert"] = mert["feature_version"]
    route_arrays = {
        route: select_features(route, classical["X"], mert["X"] if mert else None)
        for route in routes
    }
    cv_results: dict[str, Any] = {"schema_version": 1, "selection_policy": SELECTION_POLICY, "datasets": {}}
    validation_results: dict[str, Any] = {"schema_version": 1, "score_column_order": CLASS_LABELS, "datasets": {}}
    test_scores: dict[str, Any] = {"schema_version": 1, "score_column_order": CLASS_LABELS, "datasets": {}}
    predictions: dict[str, dict[str, list[str]]] = {}
    audit: dict[str, Any] = {
        "schema_version": 1,
        "started_at_utc": started_at,
        "selection_policy": SELECTION_POLICY,
        "random_seed": SEED,
        "number_of_cv_folds": N_SPLITS,
        "candidate_grid": candidate_grid(),
        "feature_routes": routes,
        "feature_dimensions": feature_dimensions,
        "feature_versions": feature_versions,
        "input_archives": {"classical": {"path": str(classical_path), "sha256": sha256(classical_path)}},
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl")},
        "blas_threads": 1,
        "external_labeled_audio_used": False,
        "official_validation_used_for_selection_or_fitting": False,
        "test_labels_used": False,
        "datasets": {},
    }
    if mert_path:
        audit["input_archives"]["mert"] = {"path": str(mert_path), "sha256": sha256(mert_path)}

    for dataset, classes in CLASS_LABELS.items():
        dataset_mask = classical["datasets"] == dataset
        indices = {
            split: np.flatnonzero(dataset_mask & (classical["splits"] == split))
            for split in ("train", "validation", "test")
        }
        if any(len(value) == 0 for value in indices.values()):
            raise ValueError(f"Dataset {dataset} requires nonempty train, validation, and test splits")
        train_indices = indices["train"]
        labels_train = classical["labels"][train_indices]
        counts = {label: int(np.sum(labels_train == label)) for label in classes}
        if min(counts.values()) < N_SPLITS:
            raise ValueError(f"Dataset {dataset} needs at least {N_SPLITS} training examples per class: {counts}")
        folds = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(np.zeros(len(train_indices)), labels_train))
        assignments = np.empty(len(train_indices), dtype=int)
        for fold_index, (_, heldout_indices) in enumerate(folds):
            assignments[heldout_indices] = fold_index
        dataset_audit = {
            "split_counts": {split: len(value) for split, value in indices.items()},
            "train_class_counts": counts,
            "training_ids": classical["ids"][train_indices].tolist(),
            "cv_heldout_fold_by_training_id": {str(sample_id): int(fold) for sample_id, fold in zip(classical["ids"][train_indices], assignments)},
            "validation_ids": classical["ids"][indices["validation"]].tolist(),
            "test_ids": classical["ids"][indices["test"]].tolist(),
            "route_models": {},
        }
        dataset_cv: dict[str, Any] = {"classes": classes, "routes": {}}
        route_winners: dict[str, dict[str, Any]] = {}
        # All route/candidate choices are completed before consulting official
        # validation labels or evaluating a single official validation example.
        for route in routes:
            candidates, winner = cross_validate_route(route_arrays[route][train_indices], labels_train, classes, folds, dataset, route)
            route_winners[route] = winner
            dataset_cv["routes"][route] = {"candidates": candidates, "selected_candidate_id": winner["candidate_id"], "selected_cv_score": winner["cv_selection_score"]}
        selected_route = max(routes, key=lambda route: route_winners[route]["cv_selection_score"])
        selected_winner = route_winners[selected_route]
        dataset_cv.update({"selected_route": selected_route, "selected_candidate_id": selected_winner["candidate_id"], "selected_cv_score": selected_winner["cv_selection_score"]})
        cv_results["datasets"][dataset] = dataset_cv
        write_json(output_dir / "cv_results.json", cv_results)
        dataset_validation: dict[str, Any] = {"classes": classes, "selected_route": selected_route, "routes": {}}
        selected_bundle: dict[str, Any] | None = None
        for route in routes:
            winner = route_winners[route]
            pipeline = make_pipeline(winner)
            final_fit_warnings = fit_with_warnings(pipeline, route_arrays[route][train_indices], labels_train)
            bundle = {
                "schema_version": 1,
                "dataset": dataset,
                "classes": classes,
                "label_to_index": {label: index for index, label in enumerate(classes)},
                "feature_route": route,
                "feature_dimensions": feature_dimensions,
                "feature_versions": feature_versions,
                "pipeline": pipeline,
                "config": {"candidate_id": winner["candidate_id"], "model_type": winner["model_type"], "parameters": winner["parameters"], "seed": SEED, "standardize": True, "normalize": "l2"},
                "selection": {"cv_score": winner["cv_selection_score"], "policy": SELECTION_POLICY, "is_final_selected_route": route == selected_route},
                "fitted_on_split": "train",
                "training_sample_count": len(train_indices),
                "fit_warnings": final_fit_warnings,
            }
            model_path = model_dir / f"{dataset}_{route}.joblib"
            save_bundle(model_path, bundle)
            dataset_audit["route_models"][route] = {"path": str(model_path), "sha256": sha256(model_path), "candidate_id": winner["candidate_id"]}
            val_indices = indices["validation"]
            val_labels = classical["labels"][val_indices]
            val_probabilities = aligned_probabilities(pipeline, route_arrays[route][val_indices], classes)
            rows = prediction_rows(classical["ids"][val_indices], val_probabilities, classes, val_labels)
            metrics = classification_metrics(val_labels, val_probabilities, classes)
            dataset_validation["routes"][route] = {"candidate_id": winner["candidate_id"], "model_path": str(model_path), "cv_selection_score": winner["cv_selection_score"], "metrics": metrics, "predictions": rows, "errors": [row for row in rows if not row["correct_top1"]]}
            print(f"Validation {dataset}/{route}: top1={metrics['top1_accuracy']:.4f} top3={metrics['top3_accuracy']:.4f}; selection remains train-CV-only", flush=True)
            if route == selected_route:
                selected_bundle = bundle
        assert selected_bundle is not None
        selected_path = model_dir / f"{dataset}_selected.joblib"
        save_bundle(selected_path, selected_bundle)
        dataset_audit.update({"selected_route": selected_route, "selected_candidate_id": selected_winner["candidate_id"], "selected_model_path": str(selected_path), "selected_model_sha256": sha256(selected_path)})
        test_indices = indices["test"]
        probabilities_test = aligned_probabilities(selected_bundle["pipeline"], route_arrays[selected_route][test_indices], classes)
        test_rows = prediction_rows(classical["ids"][test_indices], probabilities_test, classes)
        dataset_predictions = {row["sample_id"]: row["top3"] for row in test_rows}
        expected_ids = set(classical["ids"][test_indices].tolist())
        if set(dataset_predictions) != expected_ids or len(dataset_predictions) != len(test_indices):
            raise RuntimeError(f"Dataset {dataset} test prediction coverage is not exact")
        predictions[f"dataset_{dataset}"] = dataset_predictions
        test_scores["datasets"][dataset] = {"selected_route": selected_route, "classes": classes, "predictions": test_rows}
        validation_results["datasets"][dataset] = dataset_validation
        audit["datasets"][dataset] = dataset_audit
        write_json(output_dir / "validation_results.json", validation_results)
        write_json(output_dir / "training_audit.json", audit)

    audit["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    audit["elapsed_seconds"] = time.monotonic() - started
    audit["prediction_counts"] = {dataset: len(rows) for dataset, rows in predictions.items()}
    write_json(output_dir / "predictions.json", predictions)
    write_json(output_dir / "test_scores.json", test_scores)
    write_json(output_dir / "training_audit.json", audit)
    print(f"Saved predictions and metrics to {output_dir}; checkpoints to {model_dir}", flush=True)
    return {"cv_results": cv_results, "validation_results": validation_results, "predictions": predictions, "training_audit": audit}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classical", type=Path, required=True)
    parser.add_argument("--mert", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        train(
            args.classical.expanduser().resolve(),
            args.mert.expanduser().resolve() if args.mert else None,
            args.output_dir.expanduser().resolve(),
            args.model_dir.expanduser().resolve(),
        )


if __name__ == "__main__":
    main()
