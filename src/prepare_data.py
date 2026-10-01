"""Read the official manifests, audit audio hashes and create experiment records."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import wave

LABELS = {
    'dataset_A': ['1960s', '1970s', '1980s', '1990s', '2000s', '2010s'],
    'dataset_B': ['US', 'UK', 'Brazil', 'Spain', 'Germany', 'Italy'],
}
EXPECTED = {'dataset_A': {'train': 1026, 'validation': 132, 'test': 132},
            'dataset_B': {'train': 798, 'validation': 102, 'test': 102}}


def prepare(data_root: Path, allow_missing: bool = False):
    records, audit, seen_ids, seen_hashes = [], {}, set(), {}
    for dataset, classes in LABELS.items():
        root = (data_root / dataset).resolve()
        manifest = root / 'manifest.csv'
        rows = list(csv.DictReader(manifest.open(encoding='utf-8-sig', newline='')))
        counts = Counter(r['split'] for r in rows)
        if dict(counts) != EXPECTED[dataset]:
            raise ValueError(f'{dataset}: unexpected split counts {counts}')
        missing = []
        for row in rows:
            sample_id, split, label = row['sample_id'], row['split'], row['label']
            if sample_id in seen_ids:
                raise ValueError(f'Duplicate ID: {sample_id}')
            seen_ids.add(sample_id)
            if split == 'test':
                if label:
                    raise ValueError('Test labels must be hidden; refuse to use a labeled test manifest')
            elif label not in classes:
                raise ValueError(f'Unknown label {label}')
            path = (root / row['audio_path']).resolve()
            if not path.is_relative_to(root):
                raise ValueError('Manifest path escapes dataset directory')
            record = dict(dataset=dataset, sample_id=sample_id, split=split,
                          label=label or None, path=str(path), sha256=row['sha256'])
            if not path.is_file():
                missing.append(sample_id)
                if allow_missing:
                    continue
                raise FileNotFoundError(path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != row['sha256']:
                raise ValueError(f'Audio SHA256 mismatch: {sample_id}')
            with wave.open(str(path)) as wav:
                if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 24000):
                    raise ValueError(f'Unexpected WAV format: {sample_id}')
                duration = wav.getnframes() / wav.getframerate()
                if abs(duration - float(row['duration_seconds'])) > 0.01:
                    raise ValueError(f'Unexpected duration: {sample_id}')
            if digest in seen_hashes and seen_hashes[digest]['split'] != split:
                raise ValueError(f'Duplicate audio across splits: {sample_id}')
            seen_hashes[digest] = record
            records.append(record)
        audit[dataset] = {
            'counts': dict(counts), 'labels': classes,
            'label_counts': {split: dict(Counter(r['label'] for r in rows if r['split'] == split))
                             for split in ['train', 'validation']},
            'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
            'missing_audio': missing,
        }
    return records, audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True, type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--allow-missing', action='store_true', help='Development only; produces records_available.json')
    args = p.parse_args()
    records, audit = prepare(args.data_root, args.allow_missing)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = '_available' if args.allow_missing else ''
    (args.output_dir / f'records{suffix}.json').write_text(json.dumps(records, indent=2))
    (args.output_dir / f'data_audit{suffix}.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps({'verified_audio_files': len(records), 'complete': not args.allow_missing}))


if __name__ == '__main__':
    main()
