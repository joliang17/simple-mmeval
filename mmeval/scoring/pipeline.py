import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from mmeval.scoring.match import MATCHER_REGISTRY
from mmeval.scoring.report import build_summary
from mmeval.scoring.schema import (
    extract_options,
    get_field,
    get_question,
    infer_question_type,
)

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - fallback when tqdm is unavailable
    tqdm = None


def discover_result_files(out_dir: str, pattern: str = "**/result.json", recursive: bool = True) -> List[str]:
    if not out_dir:
        return []
    if recursive:
        search_pattern = os.path.join(out_dir, pattern)
        return sorted([x for x in glob.glob(search_pattern, recursive=True) if os.path.isfile(x)])
    direct = os.path.join(out_dir, "result.json")
    return [direct] if os.path.exists(direct) else []


def _build_option_text_map(sample: Dict[str, Any], options: List[str]) -> Dict[str, str]:
    messages = sample.get("messages") or []
    if messages and isinstance(messages[0], dict):
        msg_options = messages[0].get("options")
        if isinstance(msg_options, dict):
            return {str(k).strip().upper(): str(v) for k, v in msg_options.items()}

    option_text_map = {}
    for label in options:
        if label in sample:
            option_text_map[label] = str(sample[label])
    return option_text_map


def build_matchers(args):
    matcher_names = [x.strip().lower() for x in args.matching_order.split(",") if x.strip()]
    matchers = []
    for name in matcher_names:
        if name not in MATCHER_REGISTRY:
            raise ValueError(f"Unknown matcher `{name}` in --matching_order")
        matcher_cls = MATCHER_REGISTRY[name]
        if name == "llm":
            matchers.append(matcher_cls(args))
        else:
            matchers.append(matcher_cls())
    return matchers


def _normalize_eval_id(value: Any) -> str:
    return str(value)


def _load_cached_results(out_path: str, resume_enabled: bool) -> Dict[str, Dict[str, Any]]:
    if not resume_enabled:
        return {}

    cache_by_id: Dict[str, Dict[str, Any]] = {}
    tmp_path = f"{out_path}.tmp"
    candidates = [tmp_path, out_path]
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r") as f:
                payload = json.load(f)
            rows = payload.get("samples") if isinstance(payload, dict) else payload
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                if "eval-id" not in row:
                    continue
                cache_by_id[_normalize_eval_id(row["eval-id"])] = row
            # Prefer tmp (fresher) when available.
            if path.endswith(".tmp"):
                return cache_by_id
        except Exception:
            continue
    return cache_by_id


def _flush_resume_tmp(tmp_path: str, samples: List[Dict[str, Any]]) -> None:
    with open(tmp_path, "w") as f:
        json.dump(samples, f, indent=2)


def _score_single_sample(sample: Dict[str, Any], args, matchers) -> Dict[str, Any]:
    gt = get_field(sample, args.score_gt_field, None)
    pred = get_field(sample, args.score_pred_field, None)
    question = get_question(sample)
    question_type = infer_question_type(
        sample=sample,
        score_force_question_type=args.score_force_question_type,
        score_type_field=args.score_type_field,
    )
    options = extract_options(sample) if question_type == "mcq" else []
    option_text_map = _build_option_text_map(sample, options)

    eval_id = sample.get("eval-id", sample.get("id"))
    output = {
        "eval-id": eval_id,
        "question_type": question_type,
        "gt": gt,
        "pred": pred,
        "matcher_used": None,
        "matched": None,
        "is_correct": 0,
        "status": "ok",
    }

    if gt is None or pred is None:
        output["status"] = "invalid"
        output["reason"] = "missing_gt_or_pred"
        return output

    context = {
        "question_type": question_type,
        "question": question,
        "pred": pred,
        "gt": gt,
        "options": options,
        "option_text_map": option_text_map,
    }

    trace = []
    for matcher in matchers:
        result = matcher.match(sample, context)
        trace.append(
            {
                "matcher": matcher.name,
                "is_match": result.is_match,
                "matched": result.matched,
                "reason": result.reason,
                "meta": result.meta,
            }
        )
        if result.is_match:
            output["matcher_used"] = matcher.name
            output["matched"] = result.matched
            output["is_correct"] = 1
            if result.reason is not None and args.judge_include_reason:
                output["reason"] = result.reason
            break
        if matcher.name == "llm" and result.reason is not None and args.judge_include_reason:
            output["reason"] = result.reason
        if args.matching_stop_on_first and result.is_match:
            break

    if args.score_debug:
        output["trace"] = trace
    return output


def _score_single_sample_with_index(idx: int, sample: Dict[str, Any], args, matchers):
    scored = _score_single_sample(sample, args, matchers)
    if "eval-id" not in scored or scored["eval-id"] is None:
        scored["eval-id"] = sample.get("eval-id", sample.get("id", idx))
    return idx, scored


def score_result_file(result_file: str, args) -> Dict[str, Any]:
    with open(result_file, "r") as f:
        samples = json.load(f)
    if not isinstance(samples, list):
        raise ValueError(f"{result_file} must be a JSON list.")

    out_path = os.path.join(os.path.dirname(result_file), args.score_output_name)
    tmp_path = f"{out_path}.tmp"
    cached_by_id = _load_cached_results(out_path=out_path, resume_enabled=args.score_resume)
    matchers = build_matchers(args)

    scored_samples: List[Dict[str, Any]] = [None] * len(samples)
    resumed_count = 0
    pbar = None
    if args.score_progress_bar and tqdm is not None:
        pbar = tqdm(
            total=len(samples),
            desc=f"Scoring {os.path.basename(os.path.dirname(result_file))}",
            leave=False,
        )

    uncached_samples = []
    for idx, sample in enumerate(samples):
        eval_id = _normalize_eval_id(sample.get("eval-id", sample.get("id", idx)))
        cached = cached_by_id.get(eval_id)
        if cached is not None:
            scored_samples[idx] = cached
            resumed_count += 1
            if pbar is not None:
                pbar.update(1)
            continue
        uncached_samples.append((idx, sample))

    sample_workers = max(1, int(getattr(args, "parallel_per_task", 1)))
    processed_new = 0

    if sample_workers <= 1 or len(uncached_samples) <= 1:
        for idx, sample in uncached_samples:
            i, scored = _score_single_sample_with_index(idx, sample, args, matchers)
            scored_samples[i] = scored
            processed_new += 1
            if pbar is not None:
                pbar.update(1)
            if args.score_resume and processed_new % max(1, int(args.score_save_freq)) == 0:
                _flush_resume_tmp(tmp_path, [x for x in scored_samples if x is not None])
    else:
        with ThreadPoolExecutor(max_workers=sample_workers) as executor:
            futures = [
                executor.submit(_score_single_sample_with_index, idx, sample, args, matchers)
                for idx, sample in uncached_samples
            ]
            for future in as_completed(futures):
                i, scored = future.result()
                scored_samples[i] = scored
                processed_new += 1
                if pbar is not None:
                    pbar.update(1)
                if args.score_resume and processed_new % max(1, int(args.score_save_freq)) == 0:
                    _flush_resume_tmp(tmp_path, [x for x in scored_samples if x is not None])

    if pbar is not None:
        pbar.close()

    if any(x is None for x in scored_samples):
        raise RuntimeError(f"Incomplete scoring results found for {result_file}")

    summary = build_summary(scored_samples)

    payload = {
        "summary": summary,
        "config": {
            "matching_order": args.matching_order,
            "matching_stop_on_first": args.matching_stop_on_first,
            "score_gt_field": args.score_gt_field,
            "score_pred_field": args.score_pred_field,
            "score_type_field": args.score_type_field,
            "score_force_question_type": args.score_force_question_type,
            "judge_include_reason": args.judge_include_reason,
            "score_progress_bar": args.score_progress_bar,
            "score_resume": args.score_resume,
            "score_save_freq": args.score_save_freq,
            "parallel_per_task": sample_workers,
            "resumed_count": resumed_count,
        },
        "samples": scored_samples,
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    if args.score_resume and os.path.exists(tmp_path):
        os.remove(tmp_path)

    if args.score_dump_failures:
        fail_path = os.path.join(os.path.dirname(result_file), "score_failures.json")
        failed = [x for x in scored_samples if x.get("is_correct") != 1]
        with open(fail_path, "w") as f:
            json.dump(failed, f, indent=2)

    return {
        "result_file": result_file,
        "score_file": out_path,
        "summary": summary,
        "resumed_count": resumed_count,
    }

