"""Compact, label-independent audio features for HW1.

The implementation uses the standard librosa feature API:
https://librosa.org/doc/0.11.0/feature.html
https://librosa.org/doc/0.11.0/generated/librosa.onset.onset_strength.html

No recording-level gain normalization is applied. Log-mel power is measured
against a fixed reference, so loudness and timbral information remain available
to a downstream classifier. Fit any dataset-level scaler on training data only.

CLI records are a JSON list containing dataset, sample_id, path, split, label.
Relative audio paths are resolved relative to the records JSON directory.
Per-file caches permit restarting interrupted extraction without recomputing
completed recordings. The final NPZ preserves the original records order.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import numpy as np


FEATURE_VERSION = "classical-310-v1-librosa011-sr24000-fft2048-hop512"
SAMPLE_RATE = 24_000
N_FFT = 2_048
HOP_LENGTH = 512
N_MELS = 64
N_MFCC = 20
RHYTHM_BPMS = (60, 90, 120, 150, 180)


def feature_names() -> list[str]:
    """Return stable feature names in exactly the extraction order (310)."""
    groups = [
        ("mfcc", N_MFCC),
        ("mfcc_delta", N_MFCC),
        ("mfcc_delta2", N_MFCC),
        ("logmel", N_MELS),
        ("spectral_centroid", 1),
        ("spectral_bandwidth", 1),
        ("spectral_rolloff85", 1),
        ("spectral_rolloff95", 1),
        ("spectral_flatness", 1),
        ("spectral_contrast", 7),
        ("chroma", 12),
        ("rms", 1),
        ("zcr", 1),
        ("onset_strength", 1),
    ]
    names = [
        f"{name}_{stat}_{i:02d}"
        for name, count in groups
        for stat in ("mean", "std")
        for i in range(count)
    ]
    names += ["tempo_bpm"]
    names += [f"onset_autocorrelation_{bpm}bpm" for bpm in RHYTHM_BPMS]
    names += ["onset_peaks_per_second", "rms_p90_minus_p10"]
    return names


def _pool(values: np.ndarray) -> np.ndarray:
    """Mean and population standard deviation over time, channel by channel."""
    values = np.atleast_2d(values)
    return np.concatenate(
        (values.mean(axis=-1, dtype=np.float64), values.std(axis=-1, dtype=np.float64))
    )


def extract_classical(path: str | Path) -> np.ndarray:
    """Extract a finite float32 vector using only the contents of an audio file.

    The expected input is 24 kHz mono WAV. Other sampling rates are resampled
    to 24 kHz, and multichannel inputs are averaged. Signals are not peak/RMS
    normalized. Silence has tempo 0 instead of a prior-driven tempo estimate.
    """
    import librosa
    from scipy.signal import find_peaks
    import soundfile as sf

    signal, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if signal.shape[0] == 0:
        raise ValueError(f"Empty audio file: {path}")
    signal = np.mean(signal, axis=1, dtype=np.float32)
    if not np.isfinite(signal).all():
        raise ValueError(f"Audio contains non-finite samples: {path}")
    if sample_rate != SAMPLE_RATE:
        signal = librosa.resample(signal, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    duration = len(signal) / SAMPLE_RATE
    # Delta features require at least nine frames. This only affects unusually
    # short inputs; the supplied 30-second recordings never require padding.
    min_samples = 8 * HOP_LENGTH
    if len(signal) < min_samples:
        signal = np.pad(signal, (0, min_samples - len(signal)))

    magnitude = np.abs(
        librosa.stft(signal, n_fft=N_FFT, hop_length=HOP_LENGTH, window="hann")
    ).astype(np.float32, copy=False)
    power = magnitude * magnitude
    mel_power = librosa.feature.melspectrogram(
        S=power,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=0.0,
        fmax=SAMPLE_RATE / 2,
    )
    # ref=1.0 retains absolute level. An 80 dB floor limits numerical noise.
    logmel = librosa.power_to_db(mel_power, ref=1.0, top_db=80.0)
    mfcc = librosa.feature.mfcc(S=logmel, n_mfcc=N_MFCC)
    centroid = librosa.feature.spectral_centroid(S=magnitude, sr=SAMPLE_RATE)
    bandwidth = librosa.feature.spectral_bandwidth(
        S=magnitude, sr=SAMPLE_RATE, centroid=centroid
    )
    rms = librosa.feature.rms(
        y=signal, frame_length=N_FFT, hop_length=HOP_LENGTH
    )
    onset = librosa.onset.onset_strength(
        S=logmel, sr=SAMPLE_RATE, hop_length=HOP_LENGTH
    )

    pooled = [
        _pool(mfcc),
        _pool(librosa.feature.delta(mfcc, order=1)),
        _pool(librosa.feature.delta(mfcc, order=2)),
        _pool(logmel),
        _pool(centroid),
        _pool(bandwidth),
        _pool(
            librosa.feature.spectral_rolloff(
                S=magnitude, sr=SAMPLE_RATE, roll_percent=0.85
            )
        ),
        _pool(
            librosa.feature.spectral_rolloff(
                S=magnitude, sr=SAMPLE_RATE, roll_percent=0.95
            )
        ),
        _pool(librosa.feature.spectral_flatness(S=magnitude, power=2.0)),
        _pool(
            librosa.feature.spectral_contrast(
                S=magnitude, sr=SAMPLE_RATE, fmin=200.0, n_bands=6
            )
        ),
        _pool(
            librosa.feature.chroma_stft(
                S=power,
                sr=SAMPLE_RATE,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                n_chroma=12,
                tuning=0.0,
            )
        ),
        _pool(rms),
        _pool(
            librosa.feature.zero_crossing_rate(
                signal, frame_length=N_FFT, hop_length=HOP_LENGTH
            )
        ),
        _pool(onset),
    ]

    if np.max(onset, initial=0.0) > 1e-8:
        tempo = float(
            librosa.feature.tempo(
                onset_envelope=onset,
                sr=SAMPLE_RATE,
                hop_length=HOP_LENGTH,
                start_bpm=120.0,
                max_tempo=240.0,
                ac_size=8.0,
            )[0]
        )
        peaks, _ = find_peaks(
            onset,
            height=float(onset.mean() + 0.5 * onset.std()),
            distance=max(1, int(0.15 * SAMPLE_RATE / HOP_LENGTH)),
        )
        peak_rate = float(len(peaks)) / max(duration, 1e-8)
    else:
        tempo = 0.0
        peak_rate = 0.0

    # Positive onset-envelope autocorrelation at several fixed musical rates.
    # This is an input-derived rhythm descriptor, not a label or learned prior.
    onset64 = onset.astype(np.float64)
    onset_energy = float(np.dot(onset64, onset64))
    periodicity = []
    for bpm in RHYTHM_BPMS:
        lag = int(round(60.0 * SAMPLE_RATE / (bpm * HOP_LENGTH)))
        value = 0.0
        if onset_energy > 1e-12 and lag < len(onset64):
            value = float(np.dot(onset64[:-lag], onset64[lag:]) / onset_energy)
        periodicity.append(value)
    rms_range = float(np.percentile(rms, 90) - np.percentile(rms, 10))
    vector = np.concatenate(
        pooled + [np.asarray([tempo] + periodicity + [peak_rate, rms_range])]
    ).astype(np.float32)
    if vector.shape != (len(feature_names()),) or not np.isfinite(vector).all():
        raise ValueError(f"Invalid feature vector for {path}: shape={vector.shape}")
    return vector


def _atomic_npy(path: Path, vector: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npy", delete=False) as out:
            temporary = out.name
            np.save(out, vector, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def _cache_path(audio_path: Path, cache_dir: Path) -> Path:
    stat = audio_path.stat()
    identity = json.dumps(
        [FEATURE_VERSION, str(audio_path), stat.st_size, stat.st_mtime_ns],
        ensure_ascii=False,
    )
    key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return cache_dir / f"{key}.npy"


def _read_cache(path: Path) -> np.ndarray | None:
    try:
        vector = np.load(path, allow_pickle=False)
        if vector.shape == (len(feature_names()),) and np.isfinite(vector).all():
            return vector.astype(np.float32, copy=False)
    except (OSError, ValueError, EOFError):
        pass
    return None


def _extract_and_cache(path: str, cache_path: str) -> np.ndarray:
    vector = extract_classical(path)
    _atomic_npy(Path(cache_path), vector)
    return vector


def _load_records(records_path: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    with records_path.open(encoding="utf-8") as source:
        records = json.load(source)
    if not isinstance(records, list) or not records:
        raise ValueError("records JSON must contain a nonempty list")
    paths: list[Path] = []
    seen: set[tuple[str, str]] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Record {index} must be an object")
        for key in ("dataset", "sample_id", "path", "split"):
            if key not in record or not str(record[key]).strip():
                raise ValueError(f"Record {index} has no valid {key}")
        identifier = (str(record["dataset"]), str(record["sample_id"]))
        if identifier in seen:
            raise ValueError(f"Duplicate dataset/sample_id: {identifier}")
        seen.add(identifier)
        audio_path = Path(record["path"]).expanduser()
        if not audio_path.is_absolute():
            audio_path = records_path.parent / audio_path
        audio_path = audio_path.resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(f"Record {index} audio does not exist: {audio_path}")
        paths.append(audio_path)
    return records, paths


def _save_output(path: Path, records: list[dict[str, Any]], matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as out:
            temporary = out.name
            np.savez_compressed(
                out,
                X=matrix,
                ids=np.asarray([str(r["sample_id"]) for r in records], dtype=str),
                datasets=np.asarray([str(r["dataset"]) for r in records], dtype=str),
                splits=np.asarray([str(r["split"]) for r in records], dtype=str),
                labels=np.asarray(
                    ["" if r.get("label") is None else str(r["label"]) for r in records],
                    dtype=str,
                ),
                names=np.asarray(feature_names(), dtype=str),
                feature_version=np.asarray(FEATURE_VERSION),
            )
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Per-recording cache; default: OUTPUT.npz.cache/",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    # Prevent each extraction worker from launching its own large BLAS pool.
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[variable] = "1"
    records, paths = _load_records(args.records_json.expanduser().resolve())
    output = args.output.expanduser().resolve()
    cache_dir = args.cache_dir or output.with_suffix(output.suffix + ".cache")
    cache_dir = cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    matrix = np.empty((len(records), len(feature_names())), dtype=np.float32)
    pending: list[tuple[int, Path, Path]] = []
    completed = 0
    for index, audio_path in enumerate(paths):
        cache_path = _cache_path(audio_path, cache_dir)
        cached = _read_cache(cache_path)
        if cached is None:
            pending.append((index, audio_path, cache_path))
        else:
            matrix[index] = cached
            completed += 1
    print(
        f"Features: {matrix.shape[1]}; recordings: {len(records)}; "
        f"cached: {completed}; workers: {args.workers}",
        flush=True,
    )
    started = time.monotonic()
    initial_completed = completed

    def progress(index: int) -> None:
        nonlocal completed
        completed += 1
        extracted = completed - initial_completed
        elapsed = time.monotonic() - started
        remaining = (len(records) - completed) * elapsed / max(extracted, 1)
        print(
            f"[{completed}/{len(records)}] {records[index]['dataset']}/"
            f"{records[index]['sample_id']} elapsed={elapsed:.1f}s ETA={remaining:.1f}s",
            flush=True,
        )

    if args.workers == 1:
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=1):
            for index, audio_path, cache_path in pending:
                matrix[index] = _extract_and_cache(str(audio_path), str(cache_path))
                progress(index)
    elif pending:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_extract_and_cache, str(audio_path), str(cache_path)): index
                for index, audio_path, cache_path in pending
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    matrix[index] = future.result()
                except Exception as error:
                    for remaining_future in futures:
                        remaining_future.cancel()
                    raise RuntimeError(
                        f"Feature extraction failed for {paths[index]}; "
                        "completed recordings remain cached"
                    ) from error
                progress(index)
    _save_output(output, records, matrix)
    print(f"Saved {matrix.shape} to {output}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted; completed recordings remain cached.", file=sys.stderr)
        raise SystemExit(130)
