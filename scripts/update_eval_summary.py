#!/usr/bin/env python3
"""Extract simple-mmeval result.json files into an updatable CSV summary."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


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
        "--delimiter",
        default="\t",
        help="Output delimiter. Defaults to a tab character.",
    )
    parser.add_argument(
        "--include-examples",
        action="store_true",
        help="Include work_dirs/examples results.",
    )
    parser.add_argument(
        "--score-output-name",
        default="score.json",
        help="Score JSON file name next to each result.json. Defaults to score.json.",
    )
    parser.add_argument(
        "--strict-scores",
        action="store_true",
        help="Fail if any discovered result.json is missing the requested score file.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional run names to include, for example ckpt200_qwen25vl3b.",
    )
    parser.add_argument(
        "--html",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help="Also write a self-contained HTML report with CSV data embedded (no server needed). "
             "Omit PATH to default to eval_summary_static.html next to the CSV.",
    )
    return parser.parse_args()


def load_existing(path: Path, delimiter: str) -> dict[tuple[str, str], dict[str, str]]:
    if not path.exists():
        return {}

    rows: dict[tuple[str, str], dict[str, str]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
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
        return "/".join(parts[:-2]), parts[-2]
    if len(parts) == 2 and parts[-1] == "result.json":
        return parts[0], parts[0]
    return result_path.parent.name, result_path.parent.name


def format_accuracy(value: object) -> str:
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def format_count(value: object) -> str:
    if value in (None, ""):
        return ""
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def summarize_score(root: Path, result_path: Path, score_output_name: str) -> dict[str, str]:
    score_path = result_path.with_name(score_output_name)
    with score_path.open() as f:
        payload = json.load(f)

    summary = payload.get("summary") or {}
    config = payload.get("config") or {}
    total = summary.get("total", "")
    invalid = summary.get("invalid", 0)
    answered = ""
    if total not in (None, ""):
        try:
            answered = str(int(total) - int(invalid or 0))
        except (TypeError, ValueError):
            answered = format_count(total)

    run, dataset = run_and_dataset(root, result_path)
    return {
        "run": run,
        "dataset": dataset,
        "samples": format_count(total),
        "answered": answered,
        "correct": format_count(summary.get("correct", "")),
        "accuracy": format_accuracy(summary.get("accuracy", "")),
        "scoring": str(config.get("matching_order", "")),
        "result_path": str(result_path),
        "updated_at": str(int(score_path.stat().st_mtime)),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out = Path(args.out)
    only = set(args.only) if args.only else None

    rows = load_existing(out, args.delimiter)
    result_paths = discover_results(root, args.include_examples, only)
    missing_scores = []
    updated_count = 0
    for result_path in result_paths:
        score_path = result_path.with_name(args.score_output_name)
        if not score_path.exists():
            missing_scores.append(score_path)
            print(f"WARNING: missing score file, skipping: {score_path}", file=sys.stderr)
            continue

        row = summarize_score(root, result_path, args.score_output_name)
        rows[(row["run"], row["dataset"])] = row
        updated_count += 1

    if args.strict_scores and missing_scores:
        print(
            f"ERROR: {len(missing_scores)} score file(s) missing for --score-output-name "
            f"{args.score_output_name}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    ordered = sorted(rows.values(), key=lambda row: (row["run"], row["dataset"]))
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter=args.delimiter)
        writer.writeheader()
        writer.writerows(ordered)

    print(f"Updated {out} with {updated_count} score file(s); total rows: {len(ordered)}")


if __name__ == "__main__":
    main()
