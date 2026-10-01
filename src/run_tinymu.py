#!/usr/bin/env python3
"""Evaluate TinyMU using real audio and constrained answer likelihoods.

The official 229M TinyMU consists of MATPAC++ + projector + SmolLM2-135M.
This wrapper restores ALL trained weights from tinymu.pt['model']; no encoder
or language-model initialization weights, network access, or training needed.
For each audio/prompt pair, it teacher-forces each of the six possible answers
and ranks their mean token log probabilities (including EOS). This is a
constrained audio-language-model classification experiment, not free-form JSON
generation. No true label is used to compute a score or select an answer.

Inputs: --records-json list of {dataset,sample_id,path,label}, both datasets.
Outputs use run_alm.py's alm.json schema, plus candidate scores in JSONL.
The input is the first 10 seconds, resampled to 16 kHz, as in upstream inference.
The one short prompt per task is fixed in advance, with no validation tuning.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

from run_alm import LABELS, append_jsonl, atomic_json, load_completed, load_records, write_summary


PROJECT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "385133d3df4460b749d3d774c93edd4016bd4bac"
WEIGHTS_REVISION = "0735fc50bc8b881d687dedccdd48b742927611b3"
SMOL_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
PROMPTS = {
    "dataset_A": (
        "Which US release decade best matches this music? Choose one: "
        "1960s, 1970s, 1980s, 1990s, 2000s, 2010s. Answer with only the decade."
    ),
    "dataset_B": (
        "Which release market best matches this 1980s music? Choose one: "
        "US, UK, Brazil, Spain, Germany, Italy. Predict the release market, "
        "not the artist's nationality. Answer with only the market."
    ),
}


def load_source(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import upstream source {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # Needed by dataclasses in the official source.
    spec.loader.exec_module(module)
    return module


class TinyMUEvaluator:
    def __init__(self, args):
        import torch
        import yaml
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = args.device
        if args.device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable; run with --device cpu")
        torch.manual_seed(args.seed)
        torch.set_num_threads(args.cpu_threads)
        config = yaml.safe_load(args.config_path.read_text())
        model_config = config["model"]
        if model_config["encoder"]["audioenc_name"].lower() != "matpac":
            raise ValueError("Expected official TinyMU MATPAC encoder")
        if model_config["decoder"].get("use_lora", False):
            raise ValueError("This wrapper expects the official complete TinyMU checkpoint")
        source = args.vendor_root / "src" / "models"
        matpac = load_source("tinymu_official_matpac", source / "matpac" / "model.py")
        projectors = load_source("tinymu_official_projector", source / "projector.py")

        # Match the exact upstream module tree to make strict state loading
        # verify the adapter, audio encoder, trained LM and tied embeddings.
        self.model = torch.nn.Module()
        self.model.audio_encoder = torch.nn.Module()
        self.model.audio_encoder.enc = matpac.matpac_wrapper(pull_time_dimension=False)
        self.model.projector = projectors.get_projector(**model_config["projector"])
        self.model.text_decoder = torch.nn.Module()
        text_config = AutoConfig.from_pretrained(str(args.text_model_path), local_files_only=True)
        self.model.text_decoder.lm = AutoModelForCausalLM.from_config(
            text_config, attn_implementation="eager",
        )
        self.model.text_decoder.embed_fn = self.model.text_decoder.lm.model.embed_tokens
        self.ds_rate = model_config["encoder"]["ds_rate"]
        self.max_prompt_tokens = model_config["decoder"]["max_text_token_len"]
        self.tokenizer = AutoTokenizer.from_pretrained(str(args.text_model_path), local_files_only=True)
        self.tokenizer.add_special_tokens({"pad_token": "!"})
        print("Loading all trained TinyMU weights from the local official checkpoint", flush=True)
        checkpoint = torch.load(
            str(args.model_path), map_location="cpu", mmap=True, weights_only=False,
        )
        if "model" not in checkpoint:
            raise ValueError("Expected a TinyMU training checkpoint with a model state dict")
        self.model.load_state_dict(checkpoint["model"], strict=True)
        del checkpoint
        gc.collect()
        self.model.eval().requires_grad_(False).to(args.device)
        # FFT preprocessing stays on CPU; the Transformer runs on the chosen
        # device. This avoids a requirement for complex-valued MPS STFT kernels.
        self.model.audio_encoder.enc.log_mel.cpu()
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        print(f"Loaded {self.parameter_count:,} parameters on {args.device}, float32", flush=True)
        for dataset, prompt in PROMPTS.items():
            token_count = len(self.tokenizer(prompt, add_special_tokens=True)["input_ids"])
            if token_count > self.max_prompt_tokens:
                raise ValueError(f"{dataset} prompt exceeds trained input length")

    def prefix(self, path: str, prompt: str):
        import librosa
        import numpy as np

        # duration=10 is intentional and is recorded in report metadata.
        waveform, _ = librosa.load(path, sr=16000, mono=True, offset=0.0, duration=10.0)
        if not len(waveform) or not np.isfinite(waveform).all():
            raise ValueError(f"Unreadable or nonfinite audio: {path}")
        torch = self.torch
        wave = torch.from_numpy(waveform).float().unsqueeze(0)
        enc = self.model.audio_encoder.enc
        with torch.inference_mode():
            spectrogram = enc.preprocess(wave).to(self.device)
            audio_embeds, _ = enc.forward_precise(spectrogram)
            audio_embeds = torch.nn.functional.avg_pool2d(
                audio_embeds, kernel_size=(self.ds_rate, 1),
            )
            audio_embeds = self.model.projector(audio_embeds)
            embed = self.model.text_decoder.embed_fn
            # Official TinyMU uses token 0 between audio and prompt embeddings.
            separator = embed(torch.tensor([[0]], device=self.device))
            input_ids = self.tokenizer(
                prompt, add_special_tokens=True, return_tensors="pt",
            )["input_ids"].to(self.device)
            prefix = torch.cat([audio_embeds, separator, embed(input_ids)], dim=1)
        return prefix, len(waveform) / 16000, input_ids.shape[1]

    def score_candidates(self, prefix, labels: list[str]):
        """Mean log p(answer tokens + EOS | audio, prompt), all six labels."""
        torch = self.torch
        # Training tokenizes target text separately and appends EOS. Follow
        # that convention rather than inserting undocumented answer templates.
        targets = [
            self.tokenizer(label, add_special_tokens=True)["input_ids"]
            + [self.tokenizer.eos_token_id]
            for label in labels
        ]
        longest = max(map(len, targets))
        token_ids = torch.full(
            (len(labels), longest), self.tokenizer.pad_token_id,
            dtype=torch.long, device=self.device,
        )
        token_mask = torch.zeros_like(token_ids)
        for index, target in enumerate(targets):
            token_ids[index, :len(target)] = torch.tensor(target, device=self.device)
            token_mask[index, :len(target)] = 1
        with torch.inference_mode():
            embeddings = torch.cat([
                prefix.expand(len(labels), -1, -1),
                self.model.text_decoder.embed_fn(token_ids),
            ], dim=1)
            mask = torch.cat([
                torch.ones((len(labels), prefix.shape[1]), dtype=torch.long, device=self.device),
                token_mask,
            ], dim=1)
            outputs = self.model.text_decoder.lm(
                inputs_embeds=embeddings, attention_mask=mask, use_cache=False,
            )
            # Prediction at P-1 produces the first answer token; align each
            # subsequent position with its next token. Padding is not scored.
            relevant = outputs.logits[:, prefix.shape[1] - 1:prefix.shape[1] - 1 + longest]
            token_log_probs = relevant.float().log_softmax(dim=-1).gather(
                -1, token_ids.unsqueeze(-1),
            ).squeeze(-1)
            sums = (token_log_probs * token_mask).sum(dim=-1)
            means = sums / token_mask.sum(dim=-1)
            if not torch.isfinite(means).all():
                raise RuntimeError("Nonfinite candidate likelihood; do not substitute predictions")
            mean_values = means.cpu().tolist()
            sum_values = sums.cpu().tolist()
        # Fixed published class order resolves exact score ties deterministically.
        order = sorted(range(len(labels)), key=lambda i: (-mean_values[i], i))
        details = [{
            "label": label, "mean_token_log_probability": mean_values[i],
            "sequence_log_probability": sum_values[i],
            "token_count_including_eos": len(targets[i]), "token_ids": targets[i],
        } for i, label in enumerate(labels)]
        return [labels[i] for i in order[:3]], details

    def greedy_preview(self, prefix, max_new_tokens: int = 32) -> str:
        """Optional real generated response for inspection, never used to rank."""
        torch = self.torch
        mask = torch.ones(prefix.shape[:2], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            tokens = self.model.text_decoder.lm.generate(
                inputs_embeds=prefix, attention_mask=mask, max_new_tokens=max_new_tokens,
                do_sample=False, num_beams=1, repetition_penalty=1.0,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return self.tokenizer.decode(tokens[0], skip_special_tokens=True)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--records-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=PROJECT / "cache/TinyMU/tinymu.pt")
    parser.add_argument("--config-path", type=Path, default=PROJECT / "cache/TinyMU/tinymu.yaml")
    parser.add_argument("--text-model-path", type=Path, default=PROJECT / "cache/SmolLM2-135M")
    parser.add_argument("--vendor-root", type=Path, default=PROJECT / "vendor/TinyMU")
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-new-samples", type=int)
    parser.add_argument("--generate-preview", action="store_true", help="Also save greedy text for every sample; ranking still uses likelihoods")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.cpu_threads < 1 or (args.max_new_samples is not None and args.max_new_samples < 1):
        parser.error("Thread/sample limits must be positive")
    return args


def main() -> int:
    args = arguments()
    records = load_records(args.records_json, args.data_root, args.split)
    for path in [args.model_path, args.config_path, args.text_model_path / "config.json"]:
        if not path.is_file():
            raise FileNotFoundError(f"Required local model file missing: {path}")
    config = {
        "model": "TinyMU (MATPAC++ + projector + SmolLM2-135M)",
        "model_path": str(args.model_path.resolve()),
        "weights_revision": WEIGHTS_REVISION, "source_commit": SOURCE_COMMIT,
        "text_config_revision": SMOL_REVISION,
        "source_url": "https://github.com/xiquan-li/TinyMU",
        "checkpoint_url": "https://huggingface.co/AndreasXi/TinyMU",
        "split": args.split, "device": args.device, "dtype": "float32",
        "seed": args.seed, "cpu_threads": args.cpu_threads,
        "mode": "constrained_candidate_conditional_likelihood",
        "score_definition": "mean natural-log probability of separately tokenized label plus EOS, conditioned on real audio and prompt",
        "tie_breaking": "fixed published class order",
        "labels": LABELS, "prompts": PROMPTS,
        "prompt_selection_reason": (
            "One short, fixed closed-choice prompt per task was selected before "
            "evaluation to fit TinyMU's 129-token training format and constrain "
            "compute. No examples, labels, prompt search, or fitting on the "
            "evaluation split are used. Release-market wording excludes artist nationality."
        ),
        "audio_preprocessing": "first 10 seconds, mono, 16 kHz; official MATPAC log-mel and fixed normalization; no augmentation",
        "invalid_output_handling": (
            "All six valid class strings are scored; top three are distinct by "
            "construction. No generated text is parsed, so invalid-format "
            "count is zero by design, not a measure of instruction following. "
            "Nonfinite scores or audio/model failures stop the run and are "
            "logged; no fake/fallback predictions. Optional greedy text is "
            "diagnostic and does not affect scores."
        ),
        "fallback_policy": "none; numerical/runtime failures require fixing and resuming",
        "generate_preview": args.generate_preview,
        "records_sha256": hashlib.sha256(args.records_json.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "tiny_config_sha256": hashlib.sha256(args.config_path.read_bytes()).hexdigest(),
        "lm_config_sha256": hashlib.sha256((args.text_model_path / "config.json").read_bytes()).hexdigest(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "alm_run_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Resume configuration changed; use a fresh output directory")
    atomic_json(config_path, config)
    samples_path = args.output_dir / "alm_samples.jsonl"
    completed = load_completed(samples_path)
    keys = {(r["dataset"], r["sample_id"]) for r in records}
    if set(completed) - keys:
        raise ValueError("Saved predictions are not a subset of the input records")
    summary = write_summary(args.output_dir, config, records, completed)
    if args.summarize_only or summary["complete"]:
        print(f"Completed {len(completed)}/{len(records)} records")
        return 0 if summary["complete"] else 2
    model = TinyMUEvaluator(args)
    atomic_json(args.output_dir / "alm_runtime.json", {
        "parameter_count": model.parameter_count, "sampling_rate": 16000,
        "torch_version": model.torch.__version__, "python_version": sys.version,
        "checkpoint_load": "strict=True; all model state keys matched",
    })
    new_count = 0
    for record in records:
        key = (record["dataset"], record["sample_id"])
        if key in completed:
            continue
        if args.max_new_samples is not None and new_count >= args.max_new_samples:
            break
        started = time.perf_counter()
        try:
            prefix, audio_seconds, prompt_tokens = model.prefix(record["path"], PROMPTS[key[0]])
            top3, scores = model.score_candidates(prefix, LABELS[key[0]])
            preview = model.greedy_preview(prefix) if args.generate_preview else None
        except Exception as error:
            append_jsonl(args.output_dir / "alm_errors.jsonl", {
                "dataset": key[0], "sample_id": key[1],
                "error_type": type(error).__name__, "error": str(error),
            })
            write_summary(args.output_dir, config, records, completed)
            raise
        row = {
            "dataset": key[0], "sample_id": key[1], "label": record["label"],
            "top3": top3, "class_scores": scores, "used_fallback": False,
            "audio_seconds": audio_seconds, "prompt_tokens": prompt_tokens,
            "greedy_diagnostic_response": preview,
            "attempts": [{
                "kind": "constrained_candidate_likelihood", "raw_response": None,
                "parse": {"valid": True, "top3": top3,
                          "method": "rank six valid candidates; no text parsing"},
            }],
            "seconds": time.perf_counter() - started,
        }
        append_jsonl(samples_path, row)
        completed[key] = row
        new_count += 1
        summary = write_summary(args.output_dir, config, records, completed)
        print(f"[{len(completed)}/{len(records)}] {key[0]}/{key[1]} {top3} {row['seconds']:.1f}s", flush=True)
        del prefix
        gc.collect()
        if args.device == "mps":
            model.torch.mps.empty_cache()
    print(f"Saved {len(completed)}/{len(records)} records to {args.output_dir / 'alm.json'}")
    return 0 if summary["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
