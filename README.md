# HW1 — Reproducible Music Classification

Student ID: r14942154

This repository contains all project source code and the two trained classifiers
needed to reproduce the submitted Top-3 predictions. The submitted prediction
file is [r14942154.json](r14942154.json); upload it and the PDF report to their
respective NTU COOL submission slots. Intermediate results and comparison
checkpoints are retained locally; they are unnecessary for inference.

## Reproduce the predictions

Use Python 3.10 and run these commands from this repository's root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/infer.py --data-root /absolute/path/to/course_data --model-dir artifacts/models --output r14942154.json --device cpu
```

Expected data layout:

```text
course_data/
  dataset_A/manifest.csv
  dataset_A/audio/A_....wav
  dataset_B/manifest.csv
  dataset_B/audio/B_....wav
```

Inference reads only the `test` rows and ignores the label column. A manifest-free
`dataset_A/test/*.wav` and `dataset_B/test/*.wav` layout is also supported. The
official output contains 132 Dataset A entries and 102 Dataset B entries, with
three distinct valid labels per entry in descending confidence order.

`--data-root` is the parent directory containing both `dataset_A` and `dataset_B`.
The official splits are:

| Dataset | Train | Validation | Test | Allowed labels |
| --- | ---: | ---: | ---: | --- |
| A | 1,026 | 132 | 132 | `1960s`, `1970s`, `1980s`, `1990s`, `2000s`, `2010s` |
| B | 798 | 102 | 102 | `US`, `UK`, `Brazil`, `Spain`, `Germany`, `Italy` |

## Check the submission format

```bash
python src/validate_predictions.py --prediction r14942154.json --data-root /absolute/path/to/course_data
```

The JSON has exactly two top-level keys, `dataset_A` and `dataset_B`. Each contains
all test sample IDs from its manifest, without the `.wav` extension. Each ID maps
to a list of three distinct allowed labels, ranked from highest to lowest model
confidence. Scores, training IDs and validation IDs do not belong in this file.

`prediction_format_example_NOT_ANSWERS.json` illustrates this structure only.
Its label values are not reference answers and must not replace model predictions.
The validator rejects missing or extra IDs, duplicate JSON keys, invalid labels
and repeated Top-3 labels. Inference checks test counts before loading models and
checks the final output before saving it.

The JSON alone cannot verify confidence order. If a local score audit is available,
add `--scores-json /path/to/test_scores.json` to verify each ranking against the
six model probabilities. The included prediction passed this additional check
for all 234 test recordings.

The first run downloads the pinned MERT-v1-95M model from its original publisher.
Use `--mert-model /path/to/MERT-v1-95M` for an existing local model. No course audio
is uploaded to a service. `--device mps` optionally uses an Apple GPU. CPU inference
has been verified on all 234 test recordings: every Top-3 ranking exactly matched
the submitted JSON, with audio features recomputed and no feature cache used.

## Method and selected checkpoints

Both tasks concatenate 310 classical audio statistics with 3,072 frozen MERT
features. Each 30-second recording is divided into six five-second excerpts.
MERT hidden layers 7 and 12 are pooled by mean and population standard deviation.
StandardScaler and per-recording L2 normalization precede the classifier.

- `artifacts/models/A_selected.joblib`: RBF SVC, C=10.
- `artifacts/models/B_selected.joblib`: logistic regression, C=100.

Each checkpoint also includes the fitted preprocessing, fixed class order and
feature version metadata. Models were selected using three-fold cross-validation
only within official train data. Official validation was used for reporting only;
no test labels or additional labeled audio were used.

Validation Top-1 / Top-3: A 43.18% / 86.36%; B 39.22% / 75.49%. These are not test
accuracy claims. The report also contains an independent frozen TinyMU experiment
on all 234 validation clips. Its constrained answer-likelihood ranking produced
A 12.88% / 46.21% and B 16.67% / 49.02%.

## Source code

- `src/infer.py`, `model_utils.py`: checkpoint loading and ranked inference.
- `src/validate_predictions.py`: submission format, test coverage and optional score checks.
- `src/classical_features.py`, `mert_features.py`: reproducible feature extraction.
- `src/prepare_data.py`, `train_models.py`: data audit and train-only CV.
- `src/run_tinymu.py`, `run_alm.py`, `vendor/TinyMU/`: ALM experiment and helpers.
- Remaining scripts: downloads, report generation, packaging and numerical audit.

`requirements.txt` covers inference. `requirements-experiments.txt` adds packages
used by the full experiments and PDF report generator. Report/packaging scripts
require their experiment result JSON inputs, which are not included in this compact
repository; they are not needed to reproduce predictions. Neither the course audio,
feature caches nor downloaded pretrained weights are included.

## Submission link

Repository: https://github.com/alexsu30812-gif/hw1

Place the public artifact URL on the report's first page and in the NTU COOL
`HW1_report` comments. The handout specifies an open-access cloud-drive folder;
if GitHub is not accepted as equivalent storage, mirror this repository's files
to a public cloud-drive folder and update the report URL.

## Pinned public sources and attribution


- MERT model/code: <https://huggingface.co/m-a-p/MERT-v1-95M>, revision
  `12af15fef9d0ac838c3f475bfbbf26d2060dd4f5`.
  Paper: <https://arxiv.org/abs/2306.00107>.
- TinyMU source: <https://github.com/xiquan-li/TinyMU>, commit
  `385133d3df4460b749d3d774c93edd4016bd4bac`.
  Paper: <https://arxiv.org/abs/2604.15849>.
- TinyMU checkpoint/config: <https://huggingface.co/AndreasXi/TinyMU>, revision
  `0735fc50bc8b881d687dedccdd48b742927611b3`, files `tinymu.pt`, `tinymu.yaml`.
- SmolLM2 config/tokenizer: <https://huggingface.co/HuggingFaceTB/SmolLM2-135M>,
  revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`. Required files:
  `config.json`, `generation_config.json`, `tokenizer.json`,
  `tokenizer_config.json`, `special_tokens_map.json`, `vocab.json`, `merges.txt`.
- Libraries: [librosa](https://librosa.org/doc/0.11.0/),
  [scikit-learn](https://scikit-learn.org/stable/),
  [PyTorch/TorchAudio](https://pytorch.org/),
  [Transformers](https://github.com/huggingface/transformers),
  [timm](https://github.com/huggingface/pytorch-image-models),
  [ReportLab](https://www.reportlab.com/).


## AI assistance

Codex assisted with implementation, experiment orchestration, analysis and English
report drafting. All reported scores come from executed experiments. The supplied
course PDFs do not define a specific generative-AI assistance policy; this statement
does not imply separate instructor approval.
