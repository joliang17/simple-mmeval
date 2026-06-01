#!/usr/bin/env python3
"""Extract simple-mmeval result.json files into an updatable CSV summary."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


FIELDNAMES = [
    "run",
    "dataset",
    "samples",
    "answered",
    "correct",
    "accuracy",
    "scoring",
    "result_path",
    "updated_at",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan simple-mmeval result.json files and update eval_summary.csv."
    )
    parser.add_argument(
        "--root",
        default="work_dirs",
        help="Directory containing eval runs. Defaults to work_dirs.",
    )
    parser.add_argument(
        "--out",
        default="eval_summary.csv",
        help="CSV path to create/update. Defaults to eval_summary.csv.",
    )
    parser.add_argument(
        "--include-examples",
        action="store_true",
        help="Include work_dirs/examples results.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional run names to include, for example ckpt200_qwen25vl3b.",
    )
    return parser.parse_args()


def load_existing(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not path.exists():
        return {}

    rows: dict[tuple[str, str], dict[str, str]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run = row.get("run", "")
            dataset = row.get("dataset", "")
            if run and dataset:
                rows[(run, dataset)] = {field: row.get(field, "") for field in FIELDNAMES}
    return rows


def discover_results(root: Path, include_examples: bool, only: set[str] | None) -> list[Path]:
    paths = sorted(root.rglob("result.json"))
    selected = []
    for path in paths:
        rel = path.relative_to(root)
        parts = rel.parts
        if not include_examples and parts and parts[0] == "examples":
            continue
        run, _ = run_and_dataset(root, path)
        if only is not None and run not in only:
            continue
        selected.append(path)
    return selected


def run_and_dataset(root: Path, result_path: Path) -> tuple[str, str]:
    rel = result_path.relative_to(root)
    parts = rel.parts
    if len(parts) >= 3 and parts[-1] == "result.json":
        return parts[0], parts[-2]
    if len(parts) == 2 and parts[-1] == "result.json":
        return parts[0], parts[0]
    return result_path.parent.name, result_path.parent.name


def response_text(sample: dict[str, Any]) -> str:
    if "response" in sample:
        value = sample["response"]
    else:
        messages = sample.get("messages") or []
        value = messages[-1].get("response", "") if messages else ""

    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    return str(value)


def valid_option_letters(sample: dict[str, Any]) -> list[str]:
    letters = []
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        if letter in sample and sample[letter] not in (None, ""):
            letters.append(letter)
    return letters


def ground_truth(sample: dict[str, Any]) -> tuple[str | None, str]:
    if sample.get("answer_option") not in (None, ""):
        return str(sample["answer_option"]).strip().upper(), "choice"

    answer = sample.get("answer")
    if answer is None:
        return None, "unknown"

    answer_text = str(answer).strip()
    letters = valid_option_letters(sample)
    if len(answer_text) == 1 and answer_text.upper() in letters:
        return answer_text.upper(), "choice"

    normalized_answer = normalize_text(answer_text)
    for letter in letters:
        if normalize_text(str(sample.get(letter, ""))) == normalized_answer:
            return letter, "choice"

    return answer_text, "free_form"


def predicted_choice(text: str, sample: dict[str, Any]) -> str | None:
    letters = valid_option_letters(sample)
    if not letters:
        return None
    letter_class = "".join(re.escape(letter) for letter in letters)

    patterns = [
        rf"(?i)(?:final\s+answer|answer|option|choice)\s*(?:is|:|-)?\s*\(?([{letter_class}])\)?\b",
        rf"\(([{letter_class}])\)",
        rf"\b([{letter_class}])\s*[\.\)]",
        rf"\b([{letter_class}])\b",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].upper()

    normalized_response = normalize_text(text)
    for letter in letters:
        option_text = normalize_text(str(sample.get(letter, "")))
        if option_text and option_text in normalized_response:
            return letter
    return None


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9.+-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def numeric_value(text: str) -> float | None:
    matches = re.findall(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?", text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def numeric_tolerance(answer: str) -> float:
    answer = answer.strip()
    if "." in answer:
        decimals = len(answer.split(".", 1)[1])
        return 0.5 * (10 ** -decimals) + 1e-12
    return 1e-12


def is_correct(sample: dict[str, Any]) -> tuple[bool | None, bool, str]:
    text = response_text(sample)
    answered = bool(text.strip())
    gt, kind = ground_truth(sample)
    if gt is None:
        return None, answered, "unknown"

    if kind == "choice":
        pred = predicted_choice(text, sample)
        if pred is None:
            return False, answered, "choice"
        return pred == gt, answered, "choice"

    gt_num = numeric_value(str(gt))
    pred_num = numeric_value(text)
    if gt_num is not None and pred_num is not None:
        return math.isclose(pred_num, gt_num, rel_tol=0.0, abs_tol=numeric_tolerance(str(gt))), answered, "numeric"

    return normalize_text(str(gt)) in normalize_text(text), answered, "free_form"


def summarize_result(root: Path, result_path: Path) -> dict[str, str]:
    with result_path.open() as f:
        samples = json.load(f)

    run, dataset = run_and_dataset(root, result_path)
    answered = 0
    correct = 0
    scorable = 0
    scoring_modes: set[str] = set()

    for sample in samples:
        ok, has_answer, mode = is_correct(sample)
        if has_answer:
            answered += 1
        scoring_modes.add(mode)
        if ok is None:
            continue
        scorable += 1
        if ok:
            correct += 1

    accuracy = "" if scorable == 0 else f"{correct / scorable:.6f}"
    return {
        "run": run,
        "dataset": dataset,
        "samples": str(len(samples)),
        "answered": str(answered),
        "correct": str(correct) if scorable else "",
        "accuracy": accuracy,
        "scoring": "+".join(sorted(scoring_modes)),
        "result_path": str(result_path),
        "updated_at": str(int(result_path.stat().st_mtime)),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out = Path(args.out)
    only = set(args.only) if args.only else None

    rows = load_existing(out)
    result_paths = discover_results(root, args.include_examples, only)
    for result_path in result_paths:
        row = summarize_result(root, result_path)
        rows[(row["run"], row["dataset"])] = row

    ordered = sorted(rows.values(), key=lambda row: (row["run"], row["dataset"]))
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(ordered)

    print(f"Updated {out} with {len(result_paths)} result file(s); total rows: {len(ordered)}")


if __name__ == "__main__":
    main()
