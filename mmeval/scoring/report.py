from collections import defaultdict
from typing import Any, Dict, List


def build_summary(sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(sample_results)
    correct = sum(1 for x in sample_results if x.get("is_correct") == 1)
    accuracy = (correct / total) if total else 0.0

    by_matcher = defaultdict(lambda: {"total": 0, "correct": 0})
    by_question_type = defaultdict(lambda: {"total": 0, "correct": 0})
    invalid = 0
    for item in sample_results:
        matcher = item.get("matcher_used") or "unmatched"
        qtype = item.get("question_type") or "unknown"
        by_matcher[matcher]["total"] += 1
        by_matcher[matcher]["correct"] += int(item.get("is_correct") == 1)
        by_question_type[qtype]["total"] += 1
        by_question_type[qtype]["correct"] += int(item.get("is_correct") == 1)
        if item.get("status") == "invalid":
            invalid += 1

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "invalid": invalid,
        "by_matcher": dict(by_matcher),
        "by_question_type": dict(by_question_type),
    }

