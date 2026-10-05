"""Check HW1 prediction JSON against the official test IDs and label sets.

The format example is not an answer key. This validator reads IDs from the
test split, never reads hidden labels, and optionally verifies ranking against
saved model scores. The prediction JSON alone cannot prove confidence order.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

LABELS = {
    "dataset_A": ["1960s", "1970s", "1980s", "1990s", "2000s", "2010s"],
    "dataset_B": ["US", "UK", "Brazil", "Spain", "Germany", "Italy"],
}
TEST_COUNTS = {"dataset_A": 132, "dataset_B": 102}


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path):
    with Path(path).open(encoding="utf-8-sig") as source:
        return json.load(source, object_pairs_hook=reject_duplicate_keys)


def validate_test_ids(expected_ids):
    if set(expected_ids) != set(LABELS):
        raise ValueError("Expected test IDs for exactly dataset_A and dataset_B")
    for dataset, ids in expected_ids.items():
        ids = list(ids)
        if len(ids) != TEST_COUNTS[dataset]:
            raise ValueError(f"{dataset}: expected {TEST_COUNTS[dataset]} test IDs, found {len(ids)}")
        prefix = dataset[-1] + "_"
        if any(not isinstance(sid, str) or not sid.startswith(prefix)
               or sid.endswith(".wav") for sid in ids):
            raise ValueError(f"{dataset}: IDs must have prefix {prefix} and no .wav extension")
        if len(set(ids)) != len(ids):
            raise ValueError(f"{dataset}: duplicate test IDs in the data")


def read_test_ids(data_root):
    expected_ids = {}
    for dataset in LABELS:
        root = Path(data_root) / dataset
        manifest = root / "manifest.csv"
        if manifest.is_file():
            with manifest.open(encoding="utf-8-sig", newline="") as source:
                rows = csv.DictReader(source)
                if not {"sample_id", "split"}.issubset(rows.fieldnames or []):
                    raise ValueError(f"{manifest}: missing sample_id or split column")
                expected_ids[dataset] = [r["sample_id"] for r in rows if r["split"] == "test"]
        else:
            expected_ids[dataset] = [p.stem for p in sorted((root / "test").glob("*.wav"))]
    validate_test_ids(expected_ids)
    return expected_ids


def scores_from_audit(audit):
    scores = {}
    for dataset, labels in LABELS.items():
        item = audit["datasets"][dataset[-1]]
        columns = item["classes"]
        if len(columns) != 6 or set(columns) != set(labels):
            raise ValueError(f"{dataset}: score audit has incorrect classes")
        order = [columns.index(label) for label in labels]
        scores[dataset] = {}
        for row in item["predictions"]:
            sid = row["sample_id"]
            if sid in scores[dataset] or len(row["scores"]) != 6:
                raise ValueError(f"{dataset}: duplicate ID or invalid scores for {sid}")
            scores[dataset][sid] = [row["scores"][i] for i in order]
    return scores


def validate_predictions(predictions, expected_ids, scores=None):
    validate_test_ids(expected_ids)
    if not isinstance(predictions, dict) or set(predictions) != set(LABELS):
        raise ValueError("JSON must contain exactly dataset_A and dataset_B")
    if scores is not None and set(scores) != set(LABELS):
        raise ValueError("Scores must cover both datasets")
    for dataset, labels in LABELS.items():
        entries = predictions[dataset]
        if not isinstance(entries, dict):
            raise ValueError(f"{dataset}: predictions must be an object keyed by sample ID")
        wanted = set(expected_ids[dataset])
        missing, extra = wanted - set(entries), set(entries) - wanted
        if missing or extra:
            raise ValueError(f"{dataset}: missing IDs {sorted(missing)}; extra IDs {sorted(extra)}")
        if scores is not None and set(scores[dataset]) != wanted:
            raise ValueError(f"{dataset}: model scores do not cover exactly the test split")
        for sid, top3 in entries.items():
            if not isinstance(top3, list) or len(top3) != 3:
                raise ValueError(f"{sid}: expected a list of three labels")
            if any(not isinstance(label, str) or label not in labels for label in top3):
                raise ValueError(f"{sid}: unknown label in {top3}")
            if len(set(top3)) != 3:
                raise ValueError(f"{sid}: Top-3 labels must be distinct")
            if scores is not None:
                values = scores[dataset][sid]
                if len(values) != 6 or any(isinstance(v, bool) or not isinstance(v, (int, float))
                                           or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
                    raise ValueError(f"{sid}: invalid class probabilities")
                if not math.isclose(sum(values), 1.0, abs_tol=1e-5):
                    raise ValueError(f"{sid}: probabilities do not sum to one")
                ranking = sorted(range(6), key=lambda i: (-values[i], i))
                if top3 != [labels[i] for i in ranking[:3]]:
                    raise ValueError(f"{sid}: Top-3 does not match descending model confidence")
    return {dataset: len(predictions[dataset]) for dataset in LABELS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--scores-json", type=Path,
                        help="Optional test_scores.json to verify confidence order")
    args = parser.parse_args()
    try:
        scores = scores_from_audit(load_json(args.scores_json)) if args.scores_json else None
        counts = validate_predictions(load_json(args.prediction), read_test_ids(args.data_root), scores)
    except (ValueError, KeyError, OSError, TypeError) as error:
        parser.exit(1, f"Prediction validation failed: {error}\n")
    print(json.dumps({"valid": True, "test_counts": counts,
                      "confidence_order_verified": scores is not None}, indent=2))


if __name__ == "__main__":
    main()
