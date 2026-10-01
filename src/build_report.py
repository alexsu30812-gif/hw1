#!/usr/bin/env python3
"""Build the HW1 slide report from measured experiment results.

Usage:
    python src/build_report.py --input results/summary.json --output report.pdf

``build_report(payload, output_path)`` is also available as a Python API.
The input schema is documented by REQUIRED_SCHEMA below. Accuracy values use
fractions in [0, 1]. Classifier and ALM results are mandatory: this builder
never fills missing experiments with invented numbers. A missing student
ID or public artifact URL produces a visibly marked DRAFT. A name is optional.
Only reportlab is required; charts are drawn as PDF vectors.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse
from xml.sax.saxutils import escape


REQUIRED_SCHEMA = {
    "student": {"name": "optional str", "student_id": "str", "cloud_url": "https URL"},
    "experiment": {
        "seed": "int", "device": "str", "feature_method": "str",
        "feature_details": ["str"], "preprocessing": ["str"],
        "classifier": "str", "hyperparameters": {"key": "JSON value"},
        "training_protocol": ["str"], "selection_rule": "str",
        "commands": ["str"], "artifacts": ["str"],
        "versions": {"package": "version"}, "limitations": ["str"],
    },
    "datasets": {
        "dataset_A / dataset_B": {
            "labels": ["six ordered labels"],
            "counts": {"train": "int", "validation": "int", "test": "int"},
            "selected_model": "name in results",
            "results": [{"name": "str", "top1": "fraction", "top3": "fraction",
                         "n_samples": "int", "confusion_matrix": "6x6 integer array"}],
            "errors": [{"sample_id": "str", "true": "label", "predicted": "label",
                        "top3": ["three labels in rank order"]}],
            "analysis": ["str"],
        },
    },
    "alm": {
        "model_id": "str", "split": "validation",
        "prompts": [{"id": "str", "text": "str", "rationale": "str"}],
        "protocol": ["str"], "invalid_handling": "str",
        "results": {"dataset_A / dataset_B": [
            {"prompt_id": "str", "top1": "fraction", "top3": "fraction",
             "n_samples": "int", "confusion_matrix": "6x6 integer array",
             "invalid_outputs": "int", "retries": "int", "fallbacks": "int"}
        ]},
        "notes": ["str"],
    },
    "references": [{"title": "str", "url": "https URL", "used_for": "str"}],
}

LABELS = {
    "dataset_A": ["1960s", "1970s", "1980s", "1990s", "2000s", "2010s"],
    "dataset_B": ["US", "UK", "Brazil", "Spain", "Germany", "Italy"],
}
TASK_NAMES = {"dataset_A": "Release decade", "dataset_B": "Release market"}


def _require(value: Any, condition: bool, location: str, expected: str) -> Any:
    if not condition:
        raise ValueError(f"{location}: expected {expected}; received {value!r}")
    return value


def _text(value: Any, location: str) -> str:
    return _require(value, isinstance(value, str) and bool(value.strip()), location,
                    "a non-empty string")


def _strings(value: Any, location: str, allow_empty: bool = False) -> list[str]:
    _require(value, isinstance(value, list) and (allow_empty or bool(value)), location,
             "a list of strings" + ("" if allow_empty else " with at least one item"))
    for i, item in enumerate(value):
        _text(item, f"{location}[{i}]")
    return value


def _integer(value: Any, location: str, minimum: int = 0) -> int:
    return _require(value, isinstance(value, int) and not isinstance(value, bool)
                    and value >= minimum, location, f"an integer >= {minimum}")


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    return _require(value, isinstance(value, Mapping), location, "an object")


def _valid_url(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _validate_metrics(result: Mapping[str, Any], count: int, where: str) -> None:
    _mapping(result, where)
    n = _integer(result.get("n_samples"), f"{where}.n_samples", 1)
    _require(n, n == count, f"{where}.n_samples", f"all {count} samples in the split")
    for key in ("top1", "top3"):
        v = result.get(key)
        _require(v, isinstance(v, (int, float)) and not isinstance(v, bool)
                 and math.isfinite(v) and 0 <= v <= 1, f"{where}.{key}", "a fraction in [0, 1]")
    _require(result["top3"], result["top3"] >= result["top1"], f"{where}.top3", ">= top1")
    cm = result.get("confusion_matrix")
    _require(cm, isinstance(cm, list) and len(cm) == 6, f"{where}.confusion_matrix", "6 rows")
    for i, row in enumerate(cm):
        _require(row, isinstance(row, list) and len(row) == 6, f"{where}.confusion_matrix[{i}]", "6 counts")
        for j, value in enumerate(row):
            _integer(value, f"{where}.confusion_matrix[{i}][{j}]")
    total = sum(map(sum, cm))
    _require(total, total == n, f"{where}.confusion_matrix", f"counts summing to {n}")
    trace = sum(cm[i][i] for i in range(6))
    # Small tolerance permits input rounded to six decimal places.
    _require(result["top1"], abs(result["top1"] - trace / n) <= 1e-6,
             f"{where}.top1", f"the matrix accuracy {trace / n:.10f}")
    _require(result["top3"], abs(result["top3"] * n - round(result["top3"] * n)) <= n * 1e-6,
             f"{where}.top3", "an accuracy consistent with an integer number of correct samples")


def validate_payload(payload: Mapping[str, Any]) -> list[str]:
    """Validate experimental evidence; return reasons requiring a DRAFT label."""
    _mapping(payload, "root")
    student = _mapping(payload.get("student", {}), "student")
    draft_reasons = []
    for key in ("student_id",):
        if not isinstance(student.get(key), str) or not student[key].strip():
            draft_reasons.append(f"Missing student {key.replace('_', ' ')}")
    if not _valid_url(student.get("cloud_url")):
        draft_reasons.append("Missing public code/model folder URL")
    ex = _mapping(payload.get("experiment"), "experiment")
    _integer(ex.get("seed"), "experiment.seed")
    for key in ("device", "feature_method", "classifier", "selection_rule"):
        _text(ex.get(key), f"experiment.{key}")
    for key in ("feature_details", "preprocessing", "training_protocol", "commands", "artifacts", "limitations"):
        _strings(ex.get(key), f"experiment.{key}")
    for key in ("hyperparameters", "versions"):
        value = _mapping(ex.get(key), f"experiment.{key}")
        _require(value, bool(value), f"experiment.{key}", "a non-empty object")
    datasets = _mapping(payload.get("datasets"), "datasets")
    for key, required_labels in LABELS.items():
        ds = _mapping(datasets.get(key), f"datasets.{key}")
        labels = ds.get("labels")
        _require(labels, labels == required_labels, f"datasets.{key}.labels",
                 f"the matrix label order {required_labels}")
        counts = _mapping(ds.get("counts"), f"datasets.{key}.counts")
        for split in ("train", "validation", "test"):
            _integer(counts.get(split), f"datasets.{key}.counts.{split}", 1)
        results = ds.get("results")
        _require(results, isinstance(results, list) and bool(results), f"datasets.{key}.results", "measured model results")
        names = []
        for i, result in enumerate(results):
            where = f"datasets.{key}.results[{i}]"
            _validate_metrics(result, counts["validation"], where)
            names.append(_text(result.get("name"), f"{where}.name"))
        _require(names, len(set(names)) == len(names), f"datasets.{key}.results", "unique model names")
        _require(ds.get("selected_model"), ds.get("selected_model") in names,
                 f"datasets.{key}.selected_model", "a model name present in results")
        _strings(ds.get("analysis"), f"datasets.{key}.analysis", allow_empty=True)
        errors = ds.get("errors")
        _require(errors, isinstance(errors, list), f"datasets.{key}.errors", "a list (possibly empty)")
        ids = []
        for i, error in enumerate(errors):
            loc = f"datasets.{key}.errors[{i}]"
            _mapping(error, loc)
            ids.append(_text(error.get("sample_id"), f"{loc}.sample_id"))
            for label_key in ("true", "predicted"):
                _require(error.get(label_key), error.get(label_key) in labels, f"{loc}.{label_key}", "a valid label")
            top3 = error.get("top3")
            _require(top3, isinstance(top3, list) and len(top3) == 3
                     and len(set(top3)) == 3 and all(x in labels for x in top3)
                     and top3[0] == error["predicted"], f"{loc}.top3", "three distinct ranked labels, predicted first")
            _require(error["predicted"], error["predicted"] != error["true"], f"{loc}.predicted", "an actual Top-1 error")
        _require(ids, len(ids) == len(set(ids)), f"datasets.{key}.errors", "unique sample IDs")
    alm = _mapping(payload.get("alm"), "alm")
    _text(alm.get("model_id"), "alm.model_id")
    _require(alm.get("split"), alm.get("split") in {"train", "validation", "test"}, "alm.split", "a dataset split")
    for key in ("protocol", "notes"):
        _strings(alm.get(key), f"alm.{key}", allow_empty=(key == "notes"))
    _text(alm.get("invalid_handling"), "alm.invalid_handling")
    prompts = alm.get("prompts")
    _require(prompts, isinstance(prompts, list) and bool(prompts), "alm.prompts", "one or more recorded prompts")
    pids = []
    for i, prompt in enumerate(prompts):
        _mapping(prompt, f"alm.prompts[{i}]")
        for field in ("id", "text", "rationale"):
            _text(prompt.get(field), f"alm.prompts[{i}].{field}")
        pids.append(prompt["id"])
    _require(pids, len(pids) == len(set(pids)), "alm.prompts", "unique prompt IDs")
    all_results = _mapping(alm.get("results"), "alm.results")
    for key in LABELS:
        results = all_results.get(key)
        _require(results, isinstance(results, list) and bool(results), f"alm.results.{key}", "complete measured results")
        result_ids = []
        for i, result in enumerate(results):
            where = f"alm.results.{key}[{i}]"
            _validate_metrics(result, datasets[key]["counts"][alm["split"]], where)
            result_ids.append(result.get("prompt_id"))
            for counter in ("invalid_outputs", "retries", "fallbacks"):
                _integer(result.get(counter), f"{where}.{counter}")
            _require(result["fallbacks"], result["fallbacks"] <= result["n_samples"], f"{where}.fallbacks", "<= n_samples")
        _require(result_ids, sorted(result_ids) == sorted(pids), f"alm.results.{key}", "one full-split result for each prompt")
    refs = payload.get("references")
    _require(refs, isinstance(refs, list) and bool(refs), "references", "references for resources actually used")
    for i, ref in enumerate(refs):
        _mapping(ref, f"references[{i}]")
        for field in ("title", "used_for"):
            _text(ref.get(field), f"references[{i}].{field}")
        _require(ref.get("url"), _valid_url(ref.get("url")), f"references[{i}].url", "an HTTP(S) URL")
    return draft_reasons


def _selected(ds: Mapping[str, Any]) -> Mapping[str, Any]:
    return next(r for r in ds["results"] if r["name"] == ds["selected_model"])


def _pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def _value(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _observations(ds: Mapping[str, Any], dataset_key: str) -> list[str]:
    r = _selected(ds)
    cm, labels = r["confusion_matrix"], ds["labels"]
    mistakes = sum(map(sum, cm)) - sum(cm[i][i] for i in range(6))
    pair_counts = sorted(((cm[i][j] + cm[j][i], i, j)
                         for i in range(6) for j in range(i + 1, 6)), reverse=True)
    out = [f"{mistakes} / {r['n_samples']} clips have an incorrect first choice. "
           f"Top-3 exceeds Top-1 by {100 * (r['top3'] - r['top1']):.2f} percentage points."]
    if pair_counts[0][0]:
        count, i, j = pair_counts[0]
        out.append(f"Largest two-way confusion: {labels[i]} / {labels[j]} "
                   f"({count} clips in total; {cm[i][j]} and {cm[j][i]} by direction).")
    if dataset_key == "dataset_A" and mistakes:
        adjacent = sum(cm[i][j] for i in range(6) for j in range(6) if abs(i - j) == 1)
        out.append(f"Adjacent-decade mistakes: {adjacent} / {mistakes} "
                   f"({_pct(adjacent / mistakes)} of all Top-1 errors).")
    recalls = [(cm[i][i] / sum(cm[i]), i) for i in range(6) if sum(cm[i])]
    if recalls:
        recall, i = min(recalls)
        out.append(f"Lowest class recall: {labels[i]}, {cm[i][i]} / {sum(cm[i])} "
                   f"({_pct(recall)}).")
    return out


class _Slides:
    """A small bounded-layout canvas; text fails clearly instead of being clipped."""

    W, H = 960, 540
    INK = "#162941"
    MUTED = "#556779"
    BLUE = "#146C9E"
    TEAL = "#0C827B"
    BG = "#F3F7FA"
    BORDER = "#DCE5EE"

    def __init__(self, filename: str, draft_reasons: Sequence[str], student: Mapping[str, Any]):
        from reportlab.pdfgen.canvas import Canvas
        from reportlab.lib.colors import HexColor
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        self.c = Canvas(filename, pagesize=(self.W, self.H), pageCompression=1)
        self.color = HexColor
        self.draft_reasons = list(draft_reasons)
        self.student = student
        self.page = 0
        self.c.setTitle("HW1 - Music Release Decade and Market Classification")
        self.c.setAuthor(str(student.get("name") or student.get("student_id") or "Student identity pending"))
        self.c.setSubject("Measured audio-classification and audio-language-model experiments")
        self.c.setCreator("HW1 reproducible report builder; AI assistance disclosed in report")
        # Used only if user-supplied text contains characters outside Helvetica.
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))

    def _font(self, value: str, bold: bool = False) -> str:
        if any(ord(c) > 255 for c in value):
            return "STSong-Light"
        return "Helvetica-Bold" if bold else "Helvetica"

    def text(self, value: str, x: float, top: float, width: float, max_height: float,
             size: float = 16, color: str | None = None, bold: bool = False,
             minimum: float = 10.5, leading: float = 1.30, raw: bool = False) -> float:
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.platypus import Paragraph
        value = str(value)
        markup = value if raw else escape(value).replace("\n", "<br/>")
        current = size
        while current >= minimum - 1e-8:
            style = ParagraphStyle("slide", fontName=self._font(value, bold), fontSize=current,
                                   leading=current * leading, textColor=self.color(color or self.INK),
                                   spaceBefore=0, spaceAfter=0, wordWrap="LTR", splitLongWords=True)
            p = Paragraph(markup, style)
            _, height = p.wrap(width, max_height)
            if height <= max_height + 0.1:
                p.drawOn(self.c, x, self.H - top - height)
                return height
            current -= 0.5
        raise ValueError(f"Slide {self.page} text does not fit ({max_height:g} pt available): {value[:110]}")

    def box(self, x: float, top: float, width: float, height: float,
            fill: str = "#FFFFFF", stroke: str | None = None, radius: float = 12) -> None:
        self.c.setFillColor(self.color(fill))
        self.c.setStrokeColor(self.color(stroke or self.BORDER))
        self.c.setLineWidth(0.7)
        self.c.roundRect(x, self.H - top - height, width, height, radius,
                         stroke=1 if stroke else 0, fill=1)

    def bullets(self, items: Sequence[str], x: float, top: float, width: float,
                height: float, size: float = 15, gap: float = 12) -> None:
        # Paragraph's actual wrapped height makes long technical text predictable.
        if not items:
            return
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.platypus import Paragraph
        current = size
        while current >= 10.5:
            heights = []
            for item in items:
                style = ParagraphStyle("measure", fontName=self._font(item), fontSize=current,
                                       leading=current * 1.32, splitLongWords=True)
                _, h = Paragraph(escape(item).replace("\n", "<br/>"), style).wrap(width - 18, height)
                heights.append(h)
            if sum(heights) + gap * (len(items) - 1) <= height:
                cursor = top
                for item, h in zip(items, heights):
                    self.c.setFillColor(self.color(self.TEAL))
                    self.c.circle(x + 3, self.H - cursor - current * 0.57, 2.5, stroke=0, fill=1)
                    self.text(item, x + 18, cursor, width - 18, h + 1,
                              size=current, minimum=current, leading=1.32)
                    cursor += h + gap
                return
            current -= 0.5
        raise ValueError(f"Slide {self.page}: {len(items)} bullet items exceed available space")

    def start(self, title: str, section: str, subtitle: str = "") -> None:
        if self.page:
            self.c.showPage()
        self.page += 1
        self.c.setFillColor(self.color(self.BG))
        self.c.rect(0, 0, self.W, self.H, stroke=0, fill=1)
        self.c.setFillColor(self.color(self.TEAL))
        self.c.rect(0, self.H - 7, self.W, 7, stroke=0, fill=1)
        self.text(section.upper(), 40, 23, 800, 19, 10.5, self.TEAL, True)
        self.text(title, 40, 52, 880, 48, 28, bold=True, minimum=20)
        if subtitle:
            self.text(subtitle, 40, 103, 880, 35, 12.5, self.MUTED)
        self.c.setStrokeColor(self.color(self.BORDER))
        self.c.line(40, 35, 920, 35)
        label = "HW1  |  Audio-based classification"
        if self.draft_reasons:
            label += "  |  DRAFT: identity / public artifact link pending"
        self.text(label, 40, 510, 835, 18, 9.5, self.MUTED, minimum=9.5)
        self.text(f"{self.page:02d}", 882, 509, 38, 18, 10, self.MUTED, minimum=10)

    def heading(self, value: str, x: float, top: float, width: float) -> None:
        self.text(value, x, top, width, 35, 18, bold=True)

    def text_chunks(self, value: str, width: float, height: float,
                    minimum: float = 10.5) -> list[str]:
        """Split very long verbatim prompts across slides without losing text.

        Ordinary prompts keep their existing single-slide layout. If a prompt
        cannot fit even at the minimum font size, it is split at whitespace;
        concatenating the returned chunks recovers the original prompt.
        """
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.platypus import Paragraph
        style = ParagraphStyle("prompt_size", fontName=self._font(value),
                               fontSize=minimum, leading=minimum * 1.30,
                               splitLongWords=True)

        def fits(part: str) -> bool:
            _, used = Paragraph(escape(part).replace("\n", "<br/>"), style).wrap(width, height)
            return used <= height + 0.1

        parts = []
        rest = value
        while rest:
            if fits(rest):
                parts.append(rest)
                break
            low, high = 1, len(rest)
            while low < high:
                middle = (low + high + 1) // 2
                if fits(rest[:middle]):
                    low = middle
                else:
                    high = middle - 1
            cut = low
            for pos in range(cut - 1, max(0, cut // 2) - 1, -1):
                if rest[pos].isspace():
                    cut = pos + 1
                    break
            if cut < 1 or not fits(rest[:cut]):
                raise ValueError(f"Slide {self.page}: even one prompt character cannot fit")
            parts.append(rest[:cut])
            rest = rest[cut:]
        return parts or [""]

    def table(self, headers: Sequence[str], rows: Sequence[Sequence[str]], x: float, top: float,
              widths: Sequence[float], row_height: float = 38, size: float = 12) -> float:
        total_width = sum(widths)
        self.box(x, top, total_width, row_height, fill=self.INK, radius=6)
        for row_i, row in enumerate([headers] + list(rows)):
            y = top + row_i * row_height
            if row_i:
                self.box(x, y, total_width, row_height - 1,
                         fill="#FFFFFF" if row_i % 2 else "#EAF0F5", radius=0)
            cursor = x
            for value, width in zip(row, widths):
                self.text(str(value), cursor + 10, y + 8, width - 20, row_height - 13,
                          size, color="#FFFFFF" if row_i == 0 else self.INK,
                          bold=(row_i == 0), minimum=9.5)
                cursor += width
        return (len(rows) + 1) * row_height

    def matrix(self, cm: Sequence[Sequence[int]], labels: Sequence[str],
               x: float, top: float, cell: float = 43, title: str = "") -> None:
        from reportlab.lib.colors import Color
        left = 70
        if title:
            self.text(title, x, top, left + 6 * cell, 24, 15, bold=True)
            top += 31
        self.text("PREDICTED", x + left, top, 6 * cell, 18, 9.5, self.MUTED, minimum=9.5)
        y0 = top + 48
        for j, label in enumerate(labels):
            self.text(label, x + left + j * cell, top + 24, cell - 1, 20,
                      9.5, self.MUTED, minimum=8)
        for i, label in enumerate(labels):
            self.text(label, x, y0 + i * cell + cell * 0.33, left - 6, 20,
                      10, self.MUTED, minimum=9)
            support = sum(cm[i])
            for j in range(6):
                fraction = cm[i][j] / support if support else 0
                light = (0.91, 0.95, 0.97)
                dark = (0.06, 0.40, 0.53)
                blend = Color(*(light[k] * (1 - fraction) + dark[k] * fraction for k in range(3)))
                self.c.setFillColor(blend)
                self.c.rect(x + left + j * cell, self.H - y0 - (i + 1) * cell,
                            cell - 1, cell - 1, fill=1, stroke=0)
                self.c.setFont("Helvetica-Bold", 12 if cell >= 39 else 10)
                self.c.setFillColor(self.color("#FFFFFF" if fraction > 0.56 else self.INK))
                self.c.drawCentredString(x + left + (j + 0.5) * cell,
                                        self.H - y0 - (i + 0.62) * cell, str(cm[i][j]))
        self.text("Rows: true label. Cells: clip counts. Color: fraction within each true class.",
                  x, y0 + 6 * cell + 10, left + 6 * cell, 36, 10, self.MUTED, minimum=9)

    def finish(self) -> None:
        self.c.save()


def _render(slide: _Slides, payload: Mapping[str, Any]) -> None:
    student, ex, datasets, alm = (payload.get("student", {}), payload["experiment"],
                                  payload["datasets"], payload["alm"])
    slide.start("Music release decade & market", "Homework 1", "Audio representations, supervised classification, and audio-language-model evaluation")
    slide.text("Two tasks. Six classes each.\nOne reproducible experimental pipeline.",
               40, 155, 530, 106, 26, bold=True)
    identity = f"Student ID: {student.get('student_id') or '[STUDENT ID REQUIRED]'}"
    if student.get("name"):
        identity = f"Student: {student['name']}\n" + identity
    slide.text(identity, 40, 275, 535, 70, 17)
    if _valid_url(student.get("cloud_url")):
        url = escape(student["cloud_url"], {'"': '&quot;'})
        slide.text(f'<b>Public code and trained models</b><br/><link href="{url}" color="#146C9E">{url}</link>',
                   40, 364, 530, 77, 14, minimum=10.5, raw=True)
    else:
        slide.box(40, 359, 530, 103, "#FFF0D9")
        slide.text("DRAFT - NOT READY TO UPLOAD\nA public code/model folder URL must be added here. "
                   "The report can be finalized after the required public link is supplied.",
                   55, 375, 500, 73, 14, bold=True)
    for i, key in enumerate(LABELS):
        r = _selected(datasets[key])
        top = 163 + i * 144
        slide.box(610, top, 310, 123)
        slide.text(TASK_NAMES[key].upper(), 630, top + 17, 270, 21, 11, slide.TEAL, True)
        slide.text(f"{_pct(r['top1'])}  Top-1", 630, top + 47, 270, 38, 25, bold=True)
        slide.text(f"{_pct(r['top3'])} Top-3  |  validation n={r['n_samples']}",
                   630, top + 87, 270, 22, 12, slide.MUTED)

    slide.start("What the models predict", "01 / Data", "Labels describe release decade or release market, using the supplied audio alone.")
    rows = []
    for key in LABELS:
        ds = datasets[key]
        rows.append([key, TASK_NAMES[key], str(ds["counts"]["train"]),
                     str(ds["counts"]["validation"]), str(ds["counts"]["test"])])
    slide.table(["Dataset", "Target", "Train", "Validation", "Test"], rows,
                40, 150, [150, 300, 145, 145, 140], row_height=43, size=14)
    slide.text("30 s clips (midpoint, with two documented active-section replacements); mono 24 kHz PCM16; artist-disjoint splits.",
               40, 282, 880, 16, 10, slide.MUTED, minimum=10)
    slide.box(40, 299, 425, 169)
    slide.box(485, 299, 435, 169)
    slide.heading("A / United States releases", 60, 318, 385)
    slide.bullets(["Six ordered decade classes: " + ", ".join(datasets["dataset_A"]["labels"]),
                   "Question: are mistakes concentrated between neighboring decades?"],
                  60, 360, 385, 95, size=15)
    slide.heading("B / Releases from the 1980s", 505, 318, 395)
    slide.bullets(["Six markets: " + ", ".join(datasets["dataset_B"]["labels"]),
                   "Release market is not equivalent to artist nationality, recording location, or lyric language."],
                  505, 360, 395, 95, size=15)

    slide.start("From audio to ranked predictions", "02 / Method", "A separate supervised classification problem is fitted for each dataset.")
    steps = [("AUDIO", "Supplied music clip"), ("REPRESENTATION", ex["feature_method"]),
             ("CLASSIFIER", ex["classifier"]), ("OUTPUT", "Six scores, ranked Top-3")]
    for i, (title, detail) in enumerate(steps):
        x = 40 + i * 225
        slide.box(x, 156, 205, 126)
        slide.text(title, x + 14, 173, 177, 21, 11, slide.TEAL, True)
        slide.text(detail, x + 14, 207, 177, 61, 16, bold=True)
        if i < 3:
            slide.text(">", x + 207, 205, 18, 35, 23, slide.TEAL, True)
    slide.heading("How the representation helps", 40, 307, 420)
    slide.bullets(ex["feature_details"], 40, 346, 425, 143, size=13, gap=5)
    slide.heading("Preparing comparable vectors", 505, 307, 415)
    slide.bullets(ex["preprocessing"], 505, 346, 415, 143, size=13, gap=5)

    slide.start("Why these operations are used", "02 / Method details", "Frozen audio features describe a clip; small supervised classifiers learn the course labels.")
    slide.box(40, 150, 425, 330)
    slide.heading("Representations and pooling", 60, 169, 385)
    slide.bullets([
        "MFCC and log-mel summarize timbre and spectral energy. Chroma describes pitch-class content; rhythm descriptors describe periodic activity.",
        "MERT provides learned frame-level music representations. Its weights stay frozen; only the downstream classifier is fitted.",
        "Mean pooling describes typical feature activity. Standard-deviation pooling describes variation. Pooling produces a fixed-size vector but discards temporal order.",
    ], 60, 215, 385, 246, size=15)
    slide.box(485, 150, 435, 330)
    slide.heading("Scaling and classification", 505, 169, 395)
    slide.bullets([
        "Standardization: z_j = (x_j - train_mean_j) / train_std_j. This prevents large numerical units from dominating.",
        "L2 normalization: u = z / ||z||_2. Vectors have comparable length; every scaling statistic is learned from the fitting split only.",
        "Logistic regression learns linear class scores. RBF SVC can separate nonlinear feature patterns using similarity exp(-gamma * squared_distance).",
        "C controls regularization. SVC's fitted probability mapping supplies six ranked scores; the highest three form the prediction.",
    ], 505, 215, 395, 246, size=14)

    slide.start("Training and model selection", "03 / Protocol", "Experiment choices, fitting boundaries, and validation use are recorded explicitly.")
    slide.box(40, 150, 540, 329)
    slide.heading("Fitting protocol", 60, 169, 500)
    slide.bullets(ex["training_protocol"], 60, 214, 500, 246, size=15)
    slide.box(600, 150, 320, 150)
    slide.heading("Selection rule", 620, 169, 280)
    slide.text(ex["selection_rule"], 620, 211, 280, 72, 14)
    slide.box(600, 319, 320, 160)
    slide.heading("Execution", 620, 338, 280)
    slide.text(f"Random seed: {ex['seed']}\nDevice: {ex['device']}\n"
               f"Classifier: {ex['classifier']}", 620, 380, 280, 78, 14)

    slide.start("Recorded settings and metrics", "04 / Configuration", "Settings shown below are supplied by the executed experiment, not inferred by the report builder.")
    settings = [f"{key}: {_value(value)}" for key, value in ex["hyperparameters"].items()]
    slide.box(40, 150, 490, 330)
    slide.heading("Hyperparameters", 60, 169, 450)
    slide.bullets(settings, 60, 213, 450, 249, size=14, gap=9)
    slide.heading("Top-1 accuracy", 567, 166, 330)
    slide.text("Fraction of clips whose highest-ranked label equals the reference label.",
               567, 207, 330, 64, 16)
    slide.heading("Top-3 accuracy", 567, 290, 330)
    slide.text("Fraction of clips whose reference label appears among the three distinct, ordered predictions.",
               567, 331, 330, 69, 16)
    slide.text("One reference label per clip; six candidate classes. Confusion matrices use Top-1 predictions.",
               567, 426, 330, 55, 12, slide.MUTED)

    score_rows = []
    for key in LABELS:
        for result in datasets[key]["results"]:
            name = result["name"]
            if name == datasets[key]["selected_model"]:
                name += " [submitted]"
            score_rows.append([key, name, str(result["n_samples"]),
                               _pct(result["top1"]), _pct(result["top3"])])
    for batch_start in range(0, len(score_rows), 7):
        slide.start("Supervised validation results", "05 / Results", "The submitted model is designated by the recorded selection rule. Every row covers the full validation split.")
        batch = score_rows[batch_start:batch_start + 7]
        slide.table(["Dataset", "Model", "Clips", "Top-1", "Top-3"], batch,
                    40, 151, [120, 420, 80, 130, 130], row_height=38, size=12)
        slide.text("These values are validation measurements. Test predictions are submitted without a claimed test accuracy.",
                   40, 465, 880, 25, 12, slide.MUTED)

    for key in LABELS:
        ds, r = datasets[key], _selected(datasets[key])
        slide.start(f"{TASK_NAMES[key]}: where errors occur", "06 / Confusion matrices",
                    f"{key} | {r['name']} | Top-1 {_pct(r['top1'])} | Top-3 {_pct(r['top3'])} | n={r['n_samples']}")
        slide.matrix(r["confusion_matrix"], ds["labels"], 40, 150, cell=43)
        slide.heading("Observed from the matrix", 449, 155, 470)
        slide.bullets(_observations(ds, key), 449, 202, 470, 183, size=15)
        slide.text("The matrix identifies errors, but does not establish which audible cue caused a decision.",
                   449, 418, 470, 61, 14, slide.MUTED)

    # Every prompt is printed in full; extremely long prompts span more slides.
    for prompt in alm["prompts"]:
        chunks = slide.text_chunks(prompt["text"], 510, 247)
        for chunk_i, chunk in enumerate(chunks):
            suffix = f" | part {chunk_i + 1}/{len(chunks)}" if len(chunks) > 1 else ""
            slide.start(f"Audio-language model: {prompt['id']}", "07 / ALM experiment",
                        f"Model: {alm['model_id']} | Evaluated split: {alm['split']}{suffix}")
            slide.box(40, 150, 550, 330)
            slide.heading("Exact prompt / template" + (" (continued)" if chunk_i else ""), 60, 169, 510)
            slide.text(chunk, 60, 213, 510, 247, 14, minimum=10.5)
            slide.heading("Why this prompt", 620, 155, 300)
            slide.text(prompt["rationale"], 620, 197, 300, 200, 16)
            slide.text("The next slide specifies how real audio and the prompt produce the six candidate scores.",
                       620, 412, 300, 68, 14, slide.MUTED)

    slide.start("From music to an ALM ranking", "07 / ALM inference", "The frozen model scores possible answer strings conditioned on both audio and the task prompt.")
    slide.box(40, 150, 540, 330)
    slide.heading("Executed protocol", 60, 169, 500)
    slide.bullets(alm["protocol"], 60, 214, 500, 246, size=15)
    slide.box(600, 150, 320, 330)
    slide.heading("Candidate likelihood", 620, 169, 280)
    slide.bullets([
        "For each valid label y, evaluate log p(y_t | audio, prompt, preceding answer tokens) at every answer token.",
        "Average over answer tokens and EOS to reduce raw sequence-length effects; rank the six averages.",
        "For example, the decade answers 1960s through 2010s are scored separately. No reference answer enters this calculation.",
    ], 620, 214, 280, 246, size=14)

    alm_rows = []
    for key in LABELS:
        for result in alm["results"][key]:
            alm_rows.append([key, result["prompt_id"], str(result["n_samples"]),
                             _pct(result["top1"]), _pct(result["top3"]),
                             f"{result['invalid_outputs']} / {result['retries']} / {result['fallbacks']}"])
    for batch_start in range(0, len(alm_rows), 4):
        slide.start("ALM results and answer validity", "08 / ALM results", "All examples in each selected split are included. No failed answer is removed from the denominator.")
        batch = alm_rows[batch_start:batch_start + 4]
        slide.table(["Dataset", "Prompt", "Clips", "Top-1", "Top-3", "Invalid / retry / fallback"],
                    batch, 40, 150, [118, 127, 70, 100, 100, 265], row_height=39, size=11)
        slide.heading("Output parsing and recovery", 40, 363, 880)
        slide.text(alm["invalid_handling"], 40, 405, 880, 75, 14)

    for prompt in alm["prompts"]:
        slide.start(f"ALM confusion matrices: {prompt['id']}", "09 / ALM error structure", "The exact same label order is used for supervised and ALM matrices; darker cells indicate a larger within-row fraction.")
        for idx, key in enumerate(LABELS):
            result = next(r for r in alm["results"][key] if r["prompt_id"] == prompt["id"])
            slide.matrix(result["confusion_matrix"], datasets[key]["labels"],
                         40 + idx * 455, 148, cell=38,
                         title=f"{key}: Top-1 {_pct(result['top1'])}; Top-3 {_pct(result['top3'])}")

    for key in LABELS:
        ds = datasets[key]
        slide.start(f"Interpreting {TASK_NAMES[key].lower()}", "10 / Analysis", "Examples use anonymous sample IDs. Cue-level explanations remain hypotheses unless independently tested.")
        slide.heading("Observed errors and learned separation", 40, 151, 535)
        observations = ds["analysis"] or ["The matrix quantifies confusion patterns; this run does not establish a causal audio explanation."]
        slide.bullets(observations, 40, 194, 535, 286, size=14, gap=10)
        slide.heading("What the representations show", 615, 151, 305)
        slide.text(ds.get('representation_comparison', ''), 615, 195, 305, 113, size=13)
        slide.text(ds.get('learned_evidence', ''), 615, 323, 305, 102, size=12)
        errors = ds["errors"][:2]
        if errors:
            e = errors[0]
            slide.text(f"Example {e['sample_id']}\nTrue: {e['true']} | Top-3: {', '.join(e['top3'])}",
                       615, 443, 305, 46, size=11)

    slide.start("Limits of the evidence", "11 / Limitations", "Interpret measured performance within this dataset and protocol.")
    slide.box(40, 150, 540, 330)
    slide.heading("Classification and generalization", 60, 169, 500)
    slide.bullets(ex["limitations"], 60, 216, 500, 245, size=15)
    slide.box(600, 150, 320, 330)
    slide.heading("ALM observations", 620, 169, 280)
    notes = alm["notes"] or ["ALM scores describe this model, prompt, parsing rule, and split only; they do not establish general superiority."]
    slide.bullets(notes, 620, 216, 280, 245, size=14)

    slide.start("Reproduce the submission", "12 / Reproducibility", "The public folder must contain code, trained artifacts, and an environment specification.")
    slide.heading("Commands", 40, 152, 540)
    slide.box(40, 196, 540, 177, fill="#E7EEF5")
    slide.text("\n\n".join(ex["commands"]), 57, 213, 506, 142, 12)
    slide.heading("Saved inference artifacts", 615, 152, 305)
    slide.bullets(ex["artifacts"], 615, 195, 305, 176, size=13, gap=9)
    version_text = " | ".join(f"{k} {v}" for k, v in ex["versions"].items())
    slide.text("Environment: " + version_text, 40, 399, 880, 78, 12, slide.MUTED)

    slide.start("AI assistance", "13 / Disclosure", "This disclosure describes assistance used to prepare the deliverables.")
    slide.box(40, 150, 880, 156)
    slide.text("AI assisted code, experiment orchestration, analysis drafting and English report drafting by Codex.",
               64, 180, 832, 96, 23, bold=True)
    slide.bullets([
        "Reported scores are supplied by executed experiments. The report builder rejects missing results, incomplete split counts, and inconsistent confusion matrices.",
        "Public models, source code, and libraries used in the work are attributed in the references. Pretraining is distinguished from fitting on the supplied labeled data.",
        "Predictions are computed from the supplied audio and saved model artifacts. No test labels, external labeled audio, or manually assigned test predictions are used.",
    ], 40, 333, 880, 147, size=14, gap=11)

    refs = payload["references"]
    for start in range(0, len(refs), 5):
        slide.start("References and resource attribution", "14 / References", "Only resources used in the submitted work should appear here; URLs are clickable in the PDF.")
        for i, ref in enumerate(refs[start:start + 5]):
            top = 148 + i * 67
            number = start + i + 1
            url = escape(ref["url"], {'"': '&quot;'})
            slide.text(f"[{number}] {ref['title']}", 40, top, 880, 24, 14, bold=True)
            slide.text(ref["used_for"], 63, top + 26, 857, 19, 10.5, slide.MUTED)
            slide.text(f'<link href="{url}" color="#146C9E">{url}</link>',
                       63, top + 46, 857, 20, 10.5, raw=True)


def build_report(payload: Mapping[str, Any], output_path: str | Path) -> dict[str, Any]:
    """Validate actual results and atomically build a 16:9 PDF.

    Returns ``{"path": ..., "pages": ..., "draft": bool, "draft_reasons": [...]}``.
    Missing result evidence raises ValueError. Identity/link omissions generate
    a DRAFT, never a silently complete-looking report.
    """
    reasons = validate_payload(payload)
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    handle, temp = tempfile.mkstemp(prefix=".hw1-report-", suffix=".pdf", dir=output.parent)
    os.close(handle)
    try:
        slide = _Slides(temp, reasons, payload.get("student", {}))
        _render(slide, payload)
        slide.finish()
        os.replace(temp, output)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise
    return {"path": str(output), "pages": slide.page, "draft": bool(reasons), "draft_reasons": reasons}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, help="JSON summary containing actual experiment results")
    parser.add_argument("--output", type=Path, help="Destination PDF")
    parser.add_argument("--schema", action="store_true", help="Print the input schema and exit")
    parser.add_argument("--require-final", action="store_true", help="Fail if student identity or public link is missing")
    args = parser.parse_args(argv)
    if args.schema:
        print(json.dumps(REQUIRED_SCHEMA, indent=2, ensure_ascii=False))
        return 0
    if args.input is None or args.output is None:
        parser.error("--input and --output are required unless --schema is used")
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if args.require_final:
        reasons = validate_payload(payload)
        if reasons:
            parser.error("Final report blocked: " + "; ".join(reasons))
    result = build_report(payload, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
