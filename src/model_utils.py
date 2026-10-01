"""Shared feature alignment, probability ordering, and model-bundle inference."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


CLASS_LABELS = {
    "A": ["1960s", "1970s", "1980s", "1990s", "2000s", "2010s"],
    "B": ["US", "UK", "Brazil", "Spain", "Germany", "Italy"],
}
FEATURE_ROUTES = ("classical", "mert", "concat")


def canonical_dataset(value: str) -> str:
    value = str(value).strip()
    aliases = {"A": "A", "B": "B", "dataset_A": "A", "dataset_B": "B", "Dataset A": "A", "Dataset B": "B"}
    if value not in aliases:
        raise ValueError(f"Unknown dataset: {value!r}")
    return aliases[value]


def load_feature_archive(path: str | Path) -> dict[str, Any]:
    """Load features without using or exposing any test labels.

    NumPy decompresses the labels array as a storage unit, but only the
    non-test entries are inspected or copied into the returned metadata.
    """
    with np.load(path, allow_pickle=False) as archive:
        required = {"X", "ids", "datasets", "splits", "labels"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{path}: missing fields {sorted(missing)}")
        X = np.asarray(archive["X"], dtype=np.float32)
        ids = np.asarray(archive["ids"], dtype=str)
        datasets = np.asarray([canonical_dataset(x) for x in archive["datasets"]], dtype=str)
        raw_splits = np.asarray(archive["splits"], dtype=str)
        split_aliases = {"train": "train", "training": "train", "validation": "validation", "valid": "validation", "val": "validation", "test": "test"}
        if any(x not in split_aliases for x in raw_splits):
            raise ValueError(f"{path}: unrecognized split names")
        splits = np.asarray([split_aliases[x] for x in raw_splits], dtype=str)
        labels = np.full(len(ids), "", dtype="<U64")
        supervised = splits != "test"
        raw_labels = archive["labels"]
        if raw_labels.shape != ids.shape:
            raise ValueError(f"{path}: labels shape differs from ids shape")
        labels[supervised] = np.asarray(raw_labels[supervised], dtype=str)
        version = str(archive["feature_version"].item()) if "feature_version" in archive else "unspecified"
        names = np.asarray(archive["names"], dtype=str) if "names" in archive else None
    if X.ndim != 2 or not len(X) or X.shape[1] == 0:
        raise ValueError(f"{path}: X must be a nonempty recording-by-feature matrix")
    if not np.isfinite(X).all():
        raise ValueError(f"{path}: X contains non-finite values")
    for name, values in (("ids", ids), ("datasets", datasets), ("splits", splits), ("labels", labels)):
        if values.ndim != 1 or len(values) != len(X):
            raise ValueError(f"{path}: invalid {name} shape")
    keys = list(zip(datasets.tolist(), ids.tolist()))
    if len(keys) != len(set(keys)):
        raise ValueError(f"{path}: duplicate dataset/sample_id entries")
    for dataset, classes in CLASS_LABELS.items():
        observed = set(labels[(datasets == dataset) & supervised].tolist())
        if not observed.issubset(classes):
            raise ValueError(f"{path}: unknown supervised labels for {dataset}: {sorted(observed.difference(classes))}")
    if names is not None and names.shape != (X.shape[1],):
        raise ValueError(f"{path}: feature-name count does not match X")
    return {"X": X, "ids": ids, "datasets": datasets, "splits": splits, "labels": labels, "feature_version": version, "names": names}


def assert_feature_alignment(classical: dict[str, Any], mert: dict[str, Any]) -> None:
    for key in ("ids", "datasets", "splits", "labels"):
        if not np.array_equal(classical[key], mert[key]):
            raise ValueError(f"Classical and MERT feature archives are not aligned: {key}")
    if len(classical["X"]) != len(mert["X"]):
        raise ValueError("Classical and MERT feature row counts differ")


def select_features(route: str, classical: np.ndarray | None, mert: np.ndarray | None = None) -> np.ndarray:
    if route == "classical":
        if classical is None:
            raise ValueError("Classical features are required by this model")
        selected = classical
    elif route == "mert":
        if mert is None:
            raise ValueError("MERT features are required by this model")
        selected = mert
    elif route == "concat":
        if classical is None or mert is None:
            raise ValueError("Concatenation requires classical and MERT features")
        if len(classical) != len(mert):
            raise ValueError("Classical and MERT feature row counts differ")
        selected = np.concatenate((classical, mert), axis=1)
    else:
        raise ValueError(f"Unknown feature route: {route}")
    selected = np.asarray(selected, dtype=np.float32)
    if selected.ndim != 2 or not np.isfinite(selected).all():
        raise ValueError("Features must be a finite recording-by-feature matrix")
    return selected


def aligned_probabilities(pipeline: Any, X: np.ndarray, classes: list[str]) -> np.ndarray:
    """Predict probabilities and reorder columns to the assignment label order."""
    model_classes = [str(label) for label in pipeline.classes_]
    if set(model_classes) != set(classes) or len(model_classes) != len(classes):
        raise ValueError(f"Model classes {model_classes} differ from expected {classes}")
    raw = np.asarray(pipeline.predict_proba(X), dtype=np.float64)
    probabilities = raw[:, [model_classes.index(label) for label in classes]]
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise ValueError("Classifier returned invalid probabilities")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Classifier probabilities do not sum to one")
    return probabilities


def rank_labels(probabilities: np.ndarray, classes: list[str], k: int = 3) -> list[list[str]]:
    probabilities = np.asarray(probabilities)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(classes):
        raise ValueError("Probability matrix has the wrong number of columns")
    # Stable sorting makes ties reproducible in the official class order.
    order = np.argsort(-probabilities, axis=1, kind="stable")[:, :k]
    return [[classes[index] for index in row] for row in order]


def classification_metrics(labels: np.ndarray | list[str], probabilities: np.ndarray, classes: list[str]) -> dict[str, Any]:
    labels = [str(label) for label in labels]
    if not labels or len(labels) != len(probabilities):
        raise ValueError("Metrics require matching nonempty labels and predictions")
    ranked = rank_labels(probabilities, classes)
    confusion = np.zeros((len(classes), len(classes)), dtype=int)
    class_index = {label: index for index, label in enumerate(classes)}
    correct1 = correct3 = 0
    adjacent_errors = 0
    for truth, top3 in zip(labels, ranked):
        true_index = class_index[truth]
        predicted_index = class_index[top3[0]]
        confusion[true_index, predicted_index] += 1
        correct1 += int(truth == top3[0])
        correct3 += int(truth in top3)
        adjacent_errors += int(abs(true_index - predicted_index) == 1)
    counts = confusion.sum(axis=1)
    recalls = np.divide(np.diag(confusion), counts, out=np.zeros(len(classes), dtype=float), where=counts > 0)
    top1, top3 = correct1 / len(labels), correct3 / len(labels)
    result = {
        "sample_count": len(labels),
        "top1_accuracy": top1,
        "top3_accuracy": top3,
        "selection_score": top1 + 0.5 * top3,
        "classes": list(classes),
        "confusion_matrix_counts": confusion.tolist(),
        "confusion_matrix_rows": "true_label",
        "confusion_matrix_columns": "predicted_label",
        "per_class_support": counts.tolist(),
        "per_class_recall": recalls.tolist(),
        "error_count": len(labels) - correct1,
    }
    if classes == CLASS_LABELS["A"]:
        errors = len(labels) - correct1
        result["adjacent_decade_error_count"] = adjacent_errors
        result["adjacent_decade_fraction_of_errors"] = adjacent_errors / errors if errors else 0.0
    return result


def load_bundle(path: str | Path) -> dict[str, Any]:
    """Load a trusted local joblib checkpoint; never load untrusted pickle files."""
    import joblib

    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or bundle.get("schema_version") != 1:
        raise ValueError("Unsupported model bundle format")
    for key in ("pipeline", "feature_route", "classes", "dataset", "feature_dimensions", "config"):
        if key not in bundle:
            raise ValueError(f"Model bundle missing {key}")
    return bundle


def predict_bundle(bundle: dict[str, Any], classical: np.ndarray | None, mert: np.ndarray | None = None) -> np.ndarray:
    route = bundle["feature_route"]
    for name, values in (("classical", classical), ("mert", mert)):
        if route in (name, "concat"):
            if values is None or values.ndim != 2:
                raise ValueError(f"Model requires a {name} feature matrix")
            if values.shape[1] != bundle["feature_dimensions"][name]:
                raise ValueError(f"Unexpected {name} feature dimension")
    X = select_features(route, classical, mert)
    if len(X) == 0:
        return np.empty((0, len(bundle["classes"])), dtype=np.float64)
    return aligned_probabilities(bundle["pipeline"], X, bundle["classes"])


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, suffix=".json", delete=False) as out:
            temporary = out.name
            json.dump(value, out, ensure_ascii=False, indent=2, allow_nan=False)
            out.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def save_bundle(path: str | Path, bundle: dict[str, Any]) -> None:
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".joblib", delete=False) as out:
            temporary = out.name
        joblib.dump(bundle, temporary, compress=3)
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
