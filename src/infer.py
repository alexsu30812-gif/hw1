"""Reproduce the submission from audio, saved classifiers and a pinned encoder.

Example: python src/infer.py --data-root /path/to/datasets \
  --model-dir artifacts/models --output r14942154.json --device cpu
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from classical_features import extract_classical, FEATURE_VERSION as CLASSICAL_VERSION
from model_utils import load_bundle, predict_bundle, rank_labels, write_json
from mert_features import MertExtractor, FEATURE_VERSION as MERT_VERSION
from validate_predictions import validate_predictions, validate_test_ids

MERT_ID = 'm-a-p/MERT-v1-95M'
MERT_REVISION = '12af15fef9d0ac838c3f475bfbbf26d2060dd4f5'


def get_test_records(data_root):
    records = []
    seen = set()
    for dataset in ('A', 'B'):
        root = (data_root / f'dataset_{dataset}').resolve()
        manifest = root / 'manifest.csv'
        if manifest.is_file():
            with manifest.open(encoding='utf-8-sig', newline='') as f:
                rows = [r for r in csv.DictReader(f) if r['split'] == 'test']
            # The label column is deliberately unused during inference.
            items = [(r['sample_id'], (root / r['audio_path']).resolve()) for r in rows]
        else:
            folder = root / 'test'
            items = [(p.stem, p.resolve()) for p in sorted(folder.glob('*.wav'))]
        if not items:
            raise ValueError(f'No test audio found for dataset {dataset}')
        for sample_id, path in items:
            if sample_id in seen or not path.is_relative_to(root):
                raise ValueError(f'Duplicate ID or unsafe audio path: {sample_id}')
            if not path.is_file():
                raise FileNotFoundError(path)
            seen.add(sample_id)
            records.append({'dataset': dataset, 'sample_id': sample_id, 'path': path})
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True, type=Path)
    p.add_argument('--model-dir', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--mert-model', type=Path)
    p.add_argument('--cache-dir', type=Path, default=Path('.inference_cache'))
    p.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    p.add_argument('--no-feature-cache', action='store_true')
    args = p.parse_args()
    records = get_test_records(args.data_root)
    expected_ids = {
        f'dataset_{d}': [r['sample_id'] for r in records if r['dataset'] == d]
        for d in ('A', 'B')
    }
    validate_test_ids(expected_ids)
    bundles = {d: load_bundle(args.model_dir / f'{d}_selected.joblib') for d in ('A','B')}
    for d, bundle in bundles.items():
        if bundle['dataset'] != d:
            raise ValueError('Wrong checkpoint dataset')
        for route, version in [('classical', CLASSICAL_VERSION), ('mert', MERT_VERSION)]:
            if bundle['feature_route'] in (route, 'concat'):
                if bundle['feature_versions'][route] != version:
                    raise ValueError(f'Checkpoint/extractor version mismatch for {route}')
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    needs_mert = any(b['feature_route'] in ('mert','concat') for b in bundles.values())
    mert = None
    if needs_mert:
        if args.mert_model:
            model_path = args.mert_model
        else:
            from huggingface_hub import snapshot_download
            model_path = snapshot_download(MERT_ID, revision=MERT_REVISION,
                                           cache_dir=str(args.cache_dir / 'huggingface'),
                                           allow_patterns=['*.json','*.py','pytorch_model.bin'])
        mert = MertExtractor(model_path, args.device)
    predictions = {'dataset_A': {}, 'dataset_B': {}}
    with threadpool_limits(limits=1):
        for index, record in enumerate(records, 1):
            bundle = bundles[record['dataset']]
            digest = hashlib.sha256(record['path'].read_bytes()).hexdigest()
            vectors = {}
            for route, version, extractor in [('classical',CLASSICAL_VERSION,extract_classical),
                                               ('mert',MERT_VERSION,mert)]:
                if bundle['feature_route'] not in (route, 'concat'):
                    vectors[route] = None
                    continue
                tag = hashlib.sha256((digest+version).encode()).hexdigest()
                cached = args.cache_dir / f'{route}_{tag}.npy'
                if cached.exists() and not args.no_feature_cache:
                    vector = np.load(cached, allow_pickle=False)
                else:
                    vector = extractor(record['path'])
                    if not args.no_feature_cache:
                        np.save(cached, vector, allow_pickle=False)
                vectors[route] = vector[None, :]
            scores = predict_bundle(bundle, vectors['classical'], vectors['mert'])
            top3 = rank_labels(scores, bundle['classes'])[0]
            predictions[f"dataset_{record['dataset']}"][record['sample_id']] = top3
            if index % 20 == 0 or index == len(records):
                print(f'Inference {index}/{len(records)}', flush=True)
    validate_predictions(predictions, expected_ids)
    write_json(args.output, predictions)
    print(f'Saved {args.output}', flush=True)


if __name__ == '__main__':
    main()
