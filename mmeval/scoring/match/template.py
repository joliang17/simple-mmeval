import re

from mmeval.scoring.match.base import BaseMatcher, MatchResult
from mmeval.scoring.schema import (
    extract_mcq_option,
    normalize_for_exact,
    normalize_open_answer,
    normalize_text,
)


class TemplateMatcher(BaseMatcher):
    name = "template"

    def match(self, sample, context):
        question_type = context["question_type"]
        pred_raw = context["pred"]
        gt_raw = context["gt"]
        if gt_raw is None or pred_raw is None:
            return MatchResult(is_match=False)

        if question_type == "mcq":
            return self._match_mcq(context)
        return self._match_open(context)

    def _match_mcq(self, context):
        pred_raw = context["pred"]
        gt_raw = context["gt"]
        candidates = context["options"]

        option = extract_mcq_option(pred_raw, candidates)
        gt_option = extract_mcq_option(gt_raw, candidates) or normalize_for_exact(gt_raw).upper()
        if option and gt_option and option == gt_option:
            return MatchResult(is_match=True, matched=gt_option)

        # fallback: match option text in prediction body
        option_text_map = context.get("option_text_map", {})
        pred_norm = normalize_for_exact(pred_raw)
        for label, text in option_text_map.items():
            if not text:
                continue
            text_norm = normalize_for_exact(text)
            if text_norm and text_norm in pred_norm:
                if label == gt_option:
                    return MatchResult(is_match=True, matched=gt_option)
                return MatchResult(is_match=False)
        return MatchResult(is_match=False)

    def _match_open(self, context):
        pred_raw = normalize_text(context["pred"])
        gt_raw = context["gt"]
        gt_norm = normalize_open_answer(gt_raw)
        pred_norm = normalize_open_answer(pred_raw)

        # yes/no templates
        if gt_norm in {"yes", "no"}:
            m = re.search(r"\b(yes|no)\b", pred_norm)
            if m and m.group(1) == gt_norm:
                return MatchResult(is_match=True, matched=str(gt_raw))
            return MatchResult(is_match=False)

        # number templates
        if re.fullmatch(r"-?\d+(\.\d+)?", gt_norm or ""):
            nums = re.findall(r"-?\d+(?:\.\d+)?", pred_norm)
            if nums and nums[0] == gt_norm:
                return MatchResult(is_match=True, matched=str(gt_raw))
            return MatchResult(is_match=False)

        # generic "final answer is ..."
        patterns = [
            r"(?:final answer|answer is|the answer is)\s*[:\-]?\s*(.+)$",
            r"^\s*(.+?)\s*$",
        ]
        for pattern in patterns:
            m = re.search(pattern, pred_raw, flags=re.IGNORECASE)
            if not m:
                continue
            candidate = normalize_open_answer(m.group(1))
            if candidate == gt_norm and candidate:
                return MatchResult(is_match=True, matched=str(gt_raw))
        return MatchResult(is_match=False)

