#!/usr/bin/env python3
"""Run an audio-language model on every record in an HW1 evaluation split.

Official implementations used (no remote Python code):
https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct
https://huggingface.co/Qwen/Qwen2.5-Omni-3B
https://huggingface.co/docs/transformers/v4.57.1/model_doc/qwen2_audio

Input --records-json is a JSON list of {dataset, sample_id, path, label}.
``label`` is optional for test and is NEVER supplied to the model. Paths may
be absolute or relative to --data-root. An optional ``split`` field must
match --split. Both datasets must occur in the records file.

Outputs: alm.json (metrics and coverage), alm_predictions.json (ranked labels),
alm_samples.jsonl (raw responses and parsing), and alm_run_config.json.
Rerunning the same command resumes completed records. The model, prompt,
records, code and decoding configuration must match the previous run.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any


LABELS = {
    "dataset_A": ["1960s", "1970s", "1980s", "1990s", "2000s", "2010s"],
    "dataset_B": ["US", "UK", "Brazil", "Spain", "Germany", "Italy"],
}
PROMPT_VERSION = "audio_only_ranked_json_v1"
PROMPT_REASON = (
    "One fixed task-specific prompt was selected before evaluation to bound "
    "compute on a 24 GiB Mac. It names all six allowed labels, distinguishes "
    "release market from language or nationality, requests ranked JSON, and "
    "uses no labelled demonstrations. The formatting retry is not a separately "
    "selected classification prompt. No validation answers are used in prompts."
)
FALLBACK_DESCRIPTION = (
    "After at most one formatting retry by default, retain valid unique labels "
    "from the last parseable JSON ranking and fill missing positions in the "
    "published fixed class order. If no ranking is parseable, use the first "
    "three classes. All such rows are marked fallback, counted in all-sample "
    "metrics, and also reported separately. Ground truth never enters fallback."
)


def prompt_for(dataset: str, retry: bool = False) -> str:
    if dataset == "dataset_A":
        task = (
            "Listen to this music recording and estimate its release decade "
            "in the United States. Consider audible instrumentation, rhythm, "
            "vocal style, and recording/production characteristics."
        )
    else:
        task = (
            "Listen to this music recording, released in the 1980s, and estimate "
            "its release market. The target is the market of the release, not "
            "the artist's nationality or simply the language of the lyrics. "
            "Consider the combined musical and production evidence."
        )
    text = (
        task + "\nAllowed labels: " + json.dumps(LABELS[dataset]) + ".\n"
        "Rank the three most likely different labels from most to least likely. "
        "Even if uncertain, choose three. Return only a JSON object with key "
        '"top3" whose value is an array of exactly three distinct allowed label '
        "strings. Use the exact spellings above. Do not include explanations, "
        "confidence numbers, song titles, artist names, or markdown."
    )
    if retry:
        text += (
            "\nFORMAT CORRECTION: Your previous output was invalid. Return only "
            "the requested JSON object. Each of the three values must be a "
            "different member of the allowed labels."
        )
    return text


def canonical_label(value: Any, allowed: list[str]) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip().casefold()
    # Case and surrounding whitespace are harmless; no geographic inference.
    return next((label for label in allowed if label.casefold() == value), None)


def parse_ranking(response: str, allowed: list[str]) -> dict[str, Any]:
    """Recover JSON from a response; never infer ranks from free-form prose."""
    decoder = json.JSONDecoder()
    partial: list[str] = []
    errors: list[str] = []
    candidates: list[Any] = []
    stripped = response.strip()
    # Prefer complete output; embedded JSON permits harmless code fences/prose.
    try:
        candidates.append(json.loads(stripped))
    except json.JSONDecodeError:
        for match in re.finditer(r"[\[{]", stripped):
            try:
                value, _ = decoder.raw_decode(stripped[match.start():])
                candidates.append(value)
            except json.JSONDecodeError:
                pass
    for candidate in candidates:
        values = candidate.get("top3") if isinstance(candidate, dict) else candidate
        if not isinstance(values, list):
            errors.append("JSON must be an array or an object containing top3")
            continue
        canonical = [canonical_label(v, allowed) for v in values]
        unique = list(dict.fromkeys(v for v in canonical if v is not None))
        if len(unique) > len(partial):
            partial = unique[:3]
        if len(values) == 3 and None not in canonical and len(unique) == 3:
            return {
                "valid": True, "top3": unique, "partial": unique,
                "error": None,
                "strict_json": stripped.startswith("{") and stripped.endswith("}"),
            }
        errors.append("Need exactly three distinct allowed labels")
    return {
        "valid": False, "top3": None, "partial": partial,
        "error": errors[0] if errors else "No parseable JSON ranking",
        "strict_json": False,
    }


def fallback_ranking(partial: list[str], allowed: list[str]) -> list[str]:
    return list(dict.fromkeys([v for v in partial if v in allowed] + allowed))[:3]


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_records(path: Path, root: Path, split: str) -> list[dict[str, Any]]:
    value = json.loads(path.read_text())
    if not isinstance(value, list) or not value:
        raise ValueError("--records-json must contain a nonempty JSON list")
    seen: set[tuple[str, str]] = set()
    records = []
    for item in value:
        dataset, sample_id = item["dataset"], str(item["sample_id"])
        if dataset not in LABELS:
            raise ValueError(f"Unknown dataset: {dataset}")
        if item.get("split", split) != split:
            raise ValueError(f"Record {sample_id} belongs to a different split")
        key = (dataset, sample_id)
        if key in seen:
            raise ValueError(f"Duplicate record: {key}")
        seen.add(key)
        audio_path = Path(item["path"])
        if not audio_path.is_absolute():
            audio_path = root / audio_path
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        label = item.get("label")
        if label is not None and label not in LABELS[dataset]:
            raise ValueError(f"Invalid evaluation label for {key}: {label}")
        if split != "test" and label is None:
            raise ValueError(f"Evaluation label missing for {key}")
        records.append({
            "dataset": dataset, "sample_id": sample_id,
            "path": str(audio_path.resolve()), "label": label,
        })
    if {item["dataset"] for item in records} != set(LABELS):
        raise ValueError("The records file must contain both dataset_A and dataset_B")
    return records


def load_completed(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    completed: dict[tuple[str, str], dict[str, Any]] = {}
    if not path.exists():
        return completed
    # A terminated process may leave one incomplete last line. Repair only that
    # final line, and preserve every completed response before resuming.
    content = path.read_bytes()
    lines = content.splitlines(keepends=True)
    offset = 0
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if index != len(lines) - 1:
                raise ValueError(f"Corrupt nonfinal JSONL line {index + 1}")
            with path.open("r+b") as handle:
                handle.truncate(offset)
            break
        key = (row["dataset"], row["sample_id"])
        if key in completed:
            raise ValueError(f"Duplicate completed result {key}")
        completed[key] = row
        offset += len(line)
    # Ensure the next appended object starts on its own line.
    if completed and path.stat().st_size and not path.read_bytes().endswith(b"\n"):
        with path.open("ab") as handle:
            handle.write(b"\n")
    return completed


def metrics_for(rows: list[dict[str, Any]], labels: list[str]) -> dict[str, Any]:
    labelled = [row for row in rows if row.get("label") is not None]
    matrix = [[0 for _ in labels] for _ in labels]
    correct1 = correct3 = 0
    for row in labelled:
        truth = row["label"]
        prediction = row["top3"]
        matrix[labels.index(truth)][labels.index(prediction[0])] += 1
        correct1 += prediction[0] == truth
        correct3 += truth in prediction
    normalized = [
        [count / sum(row) if sum(row) else 0.0 for count in row]
        for row in matrix
    ]
    n = len(labelled)
    return {
        "n_labelled": n, "top1_accuracy": correct1 / n if n else None,
        "top3_accuracy": correct3 / n if n else None,
        "labels": labels, "confusion_matrix_counts": matrix,
        "confusion_matrix_row_normalized": normalized,
        "confusion_matrix_axes": {"rows": "true", "columns": "predicted_top1"},
    }


def write_summary(
    output_dir: Path, config: dict[str, Any], records: list[dict[str, Any]],
    completed: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    rows = [completed[(r["dataset"], r["sample_id"])] for r in records
            if (r["dataset"], r["sample_id"]) in completed]
    summary: dict[str, Any] = {
        "schema_version": 1, "config": config,
        "complete": len(rows) == len(records), "n_expected": len(records),
        "n_completed": len(rows), "datasets": {},
    }
    predictions: dict[str, dict[str, list[str]]] = {key: {} for key in LABELS}
    for dataset, labels in LABELS.items():
        subset = [row for row in rows if row["dataset"] == dataset]
        n = len(subset)
        invalid = sum(not row["attempts"][0]["parse"]["valid"] for row in subset)
        fallbacks = sum(row["used_fallback"] for row in subset)
        attempts = [attempt for row in subset for attempt in row["attempts"]]
        expected = sum(record["dataset"] == dataset for record in records)
        summary["datasets"][dataset] = {
            "n_expected": expected, "n_completed": n, "complete": n == expected,
            "first_response_invalid_count": invalid,
            "first_response_invalid_rate": invalid / n if n else None,
            "fallback_count": fallbacks,
            "fallback_rate": fallbacks / n if n else None,
            "generation_attempt_count": len(attempts),
            "invalid_attempt_count": sum(not a["parse"]["valid"] for a in attempts),
            "metrics_all_samples": metrics_for(subset, labels),
            "metrics_without_fallback": metrics_for(
                [row for row in subset if not row["used_fallback"]], labels
            ),
            "total_inference_seconds": sum(row["seconds"] for row in subset),
        }
        predictions[dataset] = {row["sample_id"]: row["top3"] for row in subset}
    atomic_json(output_dir / "alm_predictions.json", predictions)
    atomic_json(output_dir / "alm.json", summary)
    return summary


class AudioLanguageModel:
    """Audio is decoded and passed to the processor, never replaced by text."""

    def __init__(self, args: argparse.Namespace):
        import torch
        from transformers import AutoConfig, AutoProcessor

        if args.device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable; select --device cpu explicitly")
        torch.manual_seed(args.seed)
        self.torch = torch
        self.device = args.device
        self.dtype = getattr(torch, args.dtype)
        self.max_new_tokens = args.max_new_tokens
        local = {"local_files_only": not args.allow_download}
        config = AutoConfig.from_pretrained(args.model_path, **local)
        self.model_type = config.model_type
        loading = {
            **local, "torch_dtype": self.dtype,
            "attn_implementation": args.attention,
            "low_cpu_mem_usage": True,
        }
        if self.model_type == "qwen2_audio":
            from transformers import Qwen2AudioForConditionalGeneration

            model_class = Qwen2AudioForConditionalGeneration
        elif self.model_type in {"qwen2_5_omni", "qwen2_5_omni_thinker"}:
            from transformers import Qwen2_5OmniThinkerForConditionalGeneration

            # The public Thinker class loads the `thinker.*` checkpoint prefix.
            # Do not instantiate the speech-generation Talker or vocoder.
            model_class = Qwen2_5OmniThinkerForConditionalGeneration
            loading["config"] = getattr(config, "thinker_config", config)
        else:
            raise ValueError(f"Unsupported audio-language model: {self.model_type}")
        self.processor = AutoProcessor.from_pretrained(args.model_path, **local)
        print(f"Loading {model_class.__name__} ({args.dtype}, {args.device})", flush=True)
        self.model = model_class.from_pretrained(args.model_path, **loading)
        self.model.eval().to(args.device)
        # Clear sampling settings inherited from instruction model defaults.
        generation = self.model.generation_config
        generation.do_sample = False
        generation.temperature = 1.0
        generation.top_p = 1.0
        generation.top_k = 50
        generation.num_beams = 1
        generation.repetition_penalty = 1.0
        self.sampling_rate = int(self.processor.feature_extractor.sampling_rate)
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        print(
            f"Loaded {self.parameter_count:,} parameters; audio sample rate "
            f"{self.sampling_rate}; greedy decoding", flush=True,
        )

    def load_audio(self, path: str):
        import librosa
        import numpy as np

        waveform, _ = librosa.load(path, sr=self.sampling_rate, mono=True)
        if len(waveform) == 0 or not np.isfinite(waveform).all():
            raise ValueError(f"Empty or nonfinite waveform: {path}")
        # The released examples are 30 s. Fail instead of silently discarding
        # audio from a mistakenly supplied longer recording.
        if self.model_type == "qwen2_audio" and len(waveform) > 30 * self.sampling_rate:
            raise ValueError("Qwen2-Audio input exceeds 30 seconds; check the data")
        return waveform

    def generate(self, waveform, prompt: str) -> str:
        audio_content = {"type": "audio"}
        if self.model_type == "qwen2_audio":
            audio_content["audio_url"] = "recording.wav"
        else:
            audio_content["audio"] = "recording.wav"
        conversation = [{"role": "user", "content": [
            audio_content, {"type": "text", "text": prompt},
        ]}]
        if self.model_type != "qwen2_audio":
            conversation.insert(0, {"role": "system", "content": [{
                "type": "text", "text": (
                    "You are Qwen, a virtual human developed by the Qwen Team, "
                    "Alibaba Group, capable of perceiving auditory and visual "
                    "inputs, as well as generating text and speech."
                ),
            }]})
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False,
        )
        audio_keyword = "audios" if self.model_type == "qwen2_audio" else "audio"
        inputs = self.processor(
            text=text, **{audio_keyword: [waveform]},
            sampling_rate=self.sampling_rate, return_tensors="pt", padding=True,
        )
        if "input_features" not in inputs or inputs["input_features"].numel() == 0:
            raise RuntimeError("The processor produced no audio features")
        inputs = {
            key: value.to(self.device, dtype=self.dtype)
            if value.is_floating_point() else value.to(self.device)
            for key, value in inputs.items()
        }
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                num_beams=1, use_cache=True,
            )
        answer_tokens = generated[:, inputs["input_ids"].shape[1]:]
        response = self.processor.batch_decode(
            answer_tokens, skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        del generated, answer_tokens, inputs
        return response


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--records-json", type=Path)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    parser.add_argument("--split", choices=["validation", "test", "train"], default="validation")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--attention", choices=["sdpa", "eager"], default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-retries", type=int, choices=[0, 1], default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--allow-download", action="store_true", help="Allow missing Hugging Face files to download")
    parser.add_argument("--max-new-samples", type=int, help="Smoke test only; summary is marked incomplete until all records finish")
    parser.add_argument("--summarize-only", action="store_true", help="Rebuild outputs from saved JSONL without loading weights")
    args = parser.parse_args()
    if args.max_new_tokens < 1 or (args.max_new_samples is not None and args.max_new_samples < 1):
        parser.error("Token and sample limits must be positive")
    return args


def main() -> int:
    args = arguments()
    records_path = args.records_json or args.data_root / f"records_{args.split}.json"
    records = load_records(records_path, args.data_root, args.split)
    config = {
        "model_path": str(Path(args.model_path).resolve()) if Path(args.model_path).exists() else args.model_path,
        "split": args.split, "device": args.device, "dtype": args.dtype,
        "attention": args.attention, "max_new_tokens": args.max_new_tokens,
        "max_retries": args.max_retries, "seed": args.seed,
        "decoding": "greedy, num_beams=1, repetition_penalty=1.0",
        "prompt_version": PROMPT_VERSION, "prompt_selection_reason": PROMPT_REASON,
        "prompts": {dataset: prompt_for(dataset) for dataset in LABELS},
        "retry_prompts": {dataset: prompt_for(dataset, retry=True) for dataset in LABELS},
        "labels": LABELS, "fallback_policy": FALLBACK_DESCRIPTION,
        "audio_preprocessing": "whole excerpt, mono, resample to processor sampling rate; no augmentation",
        "records_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "alm_run_config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        if previous != config:
            changed = [key for key in config if config[key] != previous.get(key)]
            raise ValueError(f"Resume configuration differs ({changed}); use a new output directory")
    else:
        atomic_json(config_path, config)
    samples_path = args.output_dir / "alm_samples.jsonl"
    completed = load_completed(samples_path)
    allowed_keys = {(row["dataset"], row["sample_id"]) for row in records}
    if set(completed) - allowed_keys:
        raise ValueError("Saved results contain samples absent from the records file")
    summary = write_summary(args.output_dir, config, records, completed)
    if args.summarize_only or summary["complete"]:
        print(json.dumps({key: summary[key] for key in ["complete", "n_expected", "n_completed"]}))
        return 0 if summary["complete"] else 2

    model = AudioLanguageModel(args)
    atomic_json(args.output_dir / "alm_runtime.json", {
        "model_type": model.model_type, "parameter_count": model.parameter_count,
        "processor_sampling_rate": model.sampling_rate,
        "torch_version": model.torch.__version__,
        "python_version": sys.version,
    })
    new_count = 0
    for record in records:
        key = (record["dataset"], record["sample_id"])
        if key in completed:
            continue
        if args.max_new_samples is not None and new_count >= args.max_new_samples:
            break
        started = time.perf_counter()
        attempts = []
        try:
            waveform = model.load_audio(record["path"])
            for attempt_index in range(args.max_retries + 1):
                response = model.generate(
                    waveform, prompt_for(record["dataset"], retry=attempt_index > 0),
                )
                parsed = parse_ranking(response, LABELS[record["dataset"]])
                attempts.append({"attempt": attempt_index + 1, "raw_response": response, "parse": parsed})
                if parsed["valid"]:
                    break
        except Exception as error:
            # Infrastructure failures are not fabricated predictions. Stop,
            # retain prior work and let the corrected command resume this row.
            append_jsonl(args.output_dir / "alm_errors.jsonl", {
                "dataset": record["dataset"], "sample_id": record["sample_id"],
                "error_type": type(error).__name__, "error": str(error),
                "attempts": attempts,
            })
            write_summary(args.output_dir, config, records, completed)
            raise
        parsed = attempts[-1]["parse"]
        used_fallback = not parsed["valid"]
        top3 = parsed["top3"] if parsed["valid"] else fallback_ranking(
            parsed["partial"], LABELS[record["dataset"]],
        )
        row = {
            "dataset": record["dataset"], "sample_id": record["sample_id"],
            "label": record["label"], "top3": top3,
            "used_fallback": used_fallback, "attempts": attempts,
            "audio_seconds": len(waveform) / model.sampling_rate,
            "seconds": time.perf_counter() - started,
        }
        append_jsonl(samples_path, row)
        completed[key] = row
        new_count += 1
        summary = write_summary(args.output_dir, config, records, completed)
        print(
            f"[{len(completed)}/{len(records)}] {key[0]}/{key[1]} "
            f"{top3} fallback={used_fallback} {row['seconds']:.1f}s", flush=True,
        )
        del waveform
        gc.collect()
        if args.device == "mps":
            model.torch.mps.empty_cache()
    print(f"Saved {len(completed)}/{len(records)} records to {args.output_dir / 'alm.json'}")
    return 0 if summary["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
