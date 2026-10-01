"""Independently check Apple Accelerate matmul against non-BLAS einsum.

This performs no fitting and never uses labels. All supplied feature rows are
checked, including train/validation/test. Logistic-regression probabilities
are independently reconstructed from saved coefficients using einsum and
softmax. For nonlinear SVC models, the matrix arithmetic check uses fixed
random coefficients; it is not an independent verification of SVC inference.

Run from the HW1 folder after generating the two feature archives:
python src/verify_numerics.py --output /tmp/numerical_verification.json

Reference: https://github.com/numpy/numpy/issues/29820
"""

from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import platform
import warnings

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import joblib
import numpy as np
from scipy.special import softmax
from threadpoolctl import threadpool_limits

PROJECT = Path(__file__).resolve().parents[1]
ISSUE_URL = "https://github.com/numpy/numpy/issues/29820"


def capture_matmul(left, right):
    # Include underflow in the diagnostic, unlike NumPy's usual default.
    with np.errstate(all="warn"), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = left @ right
    return result, [str(item.message) for item in caught]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classical", type=Path, default=PROJECT / "artifacts/classical_features.npz")
    parser.add_argument("--mert", type=Path, default=PROJECT / "artifacts/mert_features.npz")
    parser.add_argument("--model-dir", type=Path, default=PROJECT / "artifacts/models")
    parser.add_argument("--output", type=Path, default=PROJECT / "results/numerical_verification.json")
    args = parser.parse_args()
    arrays = {}
    features = {}
    reference_ids = reference_datasets = None
    for name, path in (("classical", args.classical), ("mert", args.mert)):
        with np.load(path, allow_pickle=False) as data:
            X = np.asarray(data["X"])
            ids = np.asarray(data["ids"], dtype=str)
            datasets = np.asarray(data["datasets"], dtype=str)
        datasets = np.asarray([value.removeprefix("dataset_") for value in datasets])
        if reference_ids is None:
            reference_ids, reference_datasets = ids, datasets
        else:
            assert np.array_equal(ids, reference_ids), "Feature sample IDs differ"
            assert np.array_equal(datasets, reference_datasets), "Feature datasets differ"
        assert X.ndim == 2 and np.isfinite(X).all(), f"Invalid {name} features"
        arrays[name] = X
        features[name] = {
            "path": str(path), "shape": list(X.shape), "dtype": str(X.dtype),
            "all_finite": bool(np.isfinite(X).all()),
            "minimum": float(X.min()), "maximum": float(X.max()),
        }
    config_output = io.StringIO()
    with contextlib.redirect_stdout(config_output):
        np.show_config()
    configuration = config_output.getvalue()
    report = {
        "schema_version": 1,
        "record_type": "fresh_independent_numerical_check",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "numpy_version": np.__version__, "python_version": platform.python_version(),
            "platform": platform.platform(),
            "blas_backend": "Apple Accelerate" if "name: accelerate" in configuration else "see configuration",
            "numpy_configuration": configuration,
        },
        "upstream_issue_url": ISSUE_URL,
        "feature_archives": features,
        "unique_recordings": len(reference_ids),
        "coverage_by_dataset": {d: int(np.sum(reference_datasets == d)) for d in ("A", "B")},
        "labels_used": False, "training_performed": False,
        "identity_checks": [], "models": [],
    }
    with threadpool_limits(limits=1):
        for size in (14, 15, 16, 32):
            identity = np.eye(size, dtype=np.float64)
            product, messages = capture_matmul(identity, identity)
            exact = bool(np.array_equal(product, identity))
            assert exact, "Identity matmul returned incorrect values"
            report["identity_checks"].append({"size": size, "exact_identity_result": exact, "warnings": messages})
        for path in sorted(args.model_dir.glob("*.joblib")):
            if "selected" in path.name:
                continue
            bundle = joblib.load(path)
            pipeline = bundle["pipeline"]
            classifier = pipeline.named_steps["classifier"]
            route = bundle["feature_route"]
            if route == "concat":
                X = np.concatenate((arrays["classical"], arrays["mert"]), axis=1)
            else:
                X = arrays[route]
            X = X[reference_datasets == bundle["dataset"]]
            z = pipeline[:-1].transform(X)
            assert np.isfinite(z).all(), "Non-finite transformed features"
            is_logistic = type(classifier).__name__ == "LogisticRegression"
            coefficients = classifier.coef_ if is_logistic else np.random.default_rng(100).normal(size=(6, z.shape[1]))
            model_result = {
                "checkpoint": path.name, "dataset": bundle["dataset"],
                "feature_route": route, "classifier": type(classifier).__name__,
                "recordings_checked": len(X), "transformed_all_finite": True,
                "transformed_max_abs": float(np.abs(z).max()),
                "transformed_max_l2_norm": float(np.linalg.norm(z, axis=1).max()),
                "arithmetic_coefficient_source": "actual fitted LR coefficients" if is_logistic else "seed-100 random coefficients; not SVC prediction verification",
                "arithmetic_checks": [],
            }
            for dtype in (np.float32, np.float64):
                a, weights = np.asarray(z, dtype=dtype), np.asarray(coefficients, dtype=dtype)
                scores, messages = capture_matmul(a, weights.T)
                reference = np.einsum("ij,kj->ik", a, weights, optimize=False)
                residual = softmax(reference, axis=1) - 1 / 6
                gradient, gradient_messages = capture_matmul(residual.T, a)
                gradient_reference = np.einsum("ik,ij->kj", residual, a, optimize=False)
                tolerance = 1e-4 if dtype == np.float32 else 1e-10
                np.testing.assert_allclose(scores, reference, rtol=tolerance, atol=tolerance)
                np.testing.assert_allclose(gradient, gradient_reference, rtol=tolerance, atol=tolerance)
                assert np.isfinite(scores).all() and np.isfinite(gradient).all()
                model_result["arithmetic_checks"].append({
                    "dtype": np.dtype(dtype).name, "all_finite": True,
                    "max_abs_matmul_einsum_error": float(np.abs(scores - reference).max()),
                    "reference_max_abs": float(np.abs(reference).max()),
                    "max_abs_gradient_einsum_error": float(np.abs(gradient - gradient_reference).max()),
                    "warnings": messages, "gradient_warnings": gradient_messages,
                })
            if is_logistic:
                logits = np.einsum("ij,kj->ik", z.astype(np.float64), classifier.coef_, optimize=False) + classifier.intercept_
                reference_probabilities = softmax(logits, axis=1)
                with np.errstate(all="warn"), warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    probabilities = pipeline.predict_proba(X)
                np.testing.assert_allclose(probabilities, reference_probabilities, rtol=1e-12, atol=1e-12)
                same_ranks = bool(np.array_equal(np.argsort(probabilities, axis=1), np.argsort(reference_probabilities, axis=1)))
                assert same_ranks and np.isfinite(probabilities).all()
                model_result["independent_probability_check"] = {
                    "all_finite": True,
                    "max_abs_error": float(np.abs(probabilities - reference_probabilities).max()),
                    "all_six_class_rankings_identical": same_ranks,
                    "max_abs_fitted_coefficient": float(np.abs(classifier.coef_).max()),
                    "warnings": [str(item.message) for item in caught],
                }
            report["models"].append(model_result)
    assert report["models"], "No route checkpoints were found"
    report["passed"] = True
    report["interpretation"] = "All tested products, gradients and LR probabilities match independent non-BLAS einsum references; no numerical corruption observed. Warnings alone are not evidence of overflow."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Verified {len(reference_ids)} recordings and {len(report['models'])} route checkpoints; wrote {args.output}")


if __name__ == "__main__":
    main()
