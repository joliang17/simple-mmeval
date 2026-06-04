from mmeval.scoring.match.base import BaseMatcher, MatchResult
from mmeval.scoring.schema import extract_mcq_option, normalize_for_exact, normalize_open_answer


class ExactMatcher(BaseMatcher):
    name = "exact"

    def match(self, sample, context):
        question_type = context["question_type"]
        pred_raw = context["pred"]
        gt_raw = context["gt"]

        if gt_raw is None or pred_raw is None:
            return MatchResult(is_match=False)

        if question_type == "mcq":
            candidates = context["options"]
            pred_option = extract_mcq_option(pred_raw, candidates)
            gt_option = extract_mcq_option(gt_raw, candidates) or normalize_for_exact(gt_raw).upper()
            if pred_option and gt_option and pred_option == gt_option:
                return MatchResult(is_match=True, matched=gt_option)
            return MatchResult(is_match=False)

        pred_norm = normalize_open_answer(pred_raw)
        gt_norm = normalize_open_answer(gt_raw)
        if pred_norm and gt_norm and pred_norm == gt_norm:
            return MatchResult(is_match=True, matched=str(gt_raw))
        return MatchResult(is_match=False)

