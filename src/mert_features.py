"""Frozen MERT-v1-95M statistics over six deterministic 5-second chunks.

Take transformer hidden layers 7 and 12 (1-based transformer block numbers),
then population mean and standard deviation over all frame representations
from all chunks. This fixed design produces 3072 features per recording.
No labels or split-specific statistics are used in extraction.
Model: https://huggingface.co/m-a-p/MERT-v1-95M
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

FEATURE_VERSION = 'mert95m-12af15f-layers7and12-globalmeanstd-5sec-fp32-v1'
LAYERS = (7, 12)
SAMPLE_RATE = 24000
CHUNK_SAMPLES = 5 * SAMPLE_RATE


def feature_names():
    return [f'mert_layer{layer}_{stat}_{i:03d}'
            for layer in LAYERS for stat in ('mean', 'std') for i in range(768)]


class MertExtractor:
    def __init__(self, model_path, device='cpu', batch_chunks=6):
        import torch
        from transformers import AutoModel, Wav2Vec2FeatureExtractor
        self.torch, self.device = torch, device
        torch.set_num_threads(4)
        torch.manual_seed(42)
        if device == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError('MPS unavailable in this process')
        self.batch_chunks = batch_chunks
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            str(model_path), local_files_only=True)
        # Only the pinned, downloaded MERT source files are trusted here.
        self.model = AutoModel.from_pretrained(
            str(model_path), trust_remote_code=True, local_files_only=True,
            attn_implementation='eager').eval().to(device)

    def __call__(self, path):
        import librosa
        import soundfile as sf
        y, sr = sf.read(str(path), dtype='float32', always_2d=True)
        y = y.mean(axis=1)
        if sr != SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
        if not len(y) or not np.isfinite(y).all():
            raise ValueError(f'Invalid audio: {path}')
        if len(y) != 30 * SAMPLE_RATE:
            raise ValueError(f'Expected a 30-second excerpt: {path}, samples={len(y)}')
        chunks = list(y.reshape(6, CHUNK_SAMPLES))
        frame_features = {layer: [] for layer in LAYERS}
        with self.torch.inference_mode():
            for start in range(0, len(chunks), self.batch_chunks):
                batch = self.processor(chunks[start:start+self.batch_chunks],
                                       sampling_rate=SAMPLE_RATE, return_tensors='pt',
                                       padding=True)
                batch = {k: v.to(self.device) for k, v in batch.items()}
                output = self.model(**batch, output_hidden_states=True)
                for layer in LAYERS:
                    values = output.hidden_states[layer].detach().cpu().float().numpy()
                    frame_features[layer].append(values.reshape(-1, values.shape[-1]))
                del output, batch
        pooled = []
        for layer in LAYERS:
            frames = np.concatenate(frame_features[layer], axis=0)
            pooled.extend([frames.mean(axis=0, dtype=np.float64),
                           frames.std(axis=0, dtype=np.float64)])
        vector = np.concatenate(pooled).astype('float32')
        if vector.shape != (3072,) or not np.isfinite(vector).all():
            raise ValueError('Invalid MERT features')
        return vector


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records-json', required=True, type=Path)
    p.add_argument('--model-path', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--cache-dir', type=Path)
    p.add_argument('--device', choices=['cpu','mps','cuda'], default='cpu')
    p.add_argument('--batch-chunks', type=int, default=6)
    args = p.parse_args()
    records = json.loads(args.records_json.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache = args.cache_dir or args.output.with_suffix('.cache')
    cache.mkdir(parents=True, exist_ok=True)
    extractor = None
    values = []
    started = time.monotonic()
    for i, record in enumerate(records, 1):
        audio_path = Path(record['path'])
        if not audio_path.is_absolute():
            audio_path = args.records_json.parent / audio_path
        stat = audio_path.stat()
        fingerprint = f'{FEATURE_VERSION}|{audio_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}'
        key = hashlib.sha256(fingerprint.encode()).hexdigest()
        cached = cache / f'{key}.npy'
        if cached.exists():
            value = np.load(cached, allow_pickle=False)
        else:
            if extractor is None:
                extractor = MertExtractor(args.model_path, args.device, args.batch_chunks)
            value = extractor(audio_path)
            tmp = cached.with_suffix('.tmp')
            with tmp.open('wb') as out:
                np.save(out, value, allow_pickle=False)
            tmp.replace(cached)
        if value.shape != (3072,) or not np.isfinite(value).all():
            raise ValueError(f'Invalid cached feature: {cached}')
        values.append(value)
        if i % 25 == 0 or i == len(records):
            print(f'MERT {i}/{len(records)}, elapsed {time.monotonic()-started:.1f}s', flush=True)
    np.savez_compressed(args.output, X=np.stack(values),
                        ids=np.asarray([r['sample_id'] for r in records]),
                        datasets=np.asarray([r['dataset'] for r in records]),
                        splits=np.asarray([r['split'] for r in records]),
                        labels=np.asarray([r.get('label') or '' for r in records]),
                        names=np.asarray(feature_names()), feature_version=np.asarray(FEATURE_VERSION))
    print(f'Saved {args.output}', flush=True)


if __name__ == '__main__':
    main()
