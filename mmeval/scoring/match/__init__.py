from mmeval.scoring.match.base import MatchResult
from mmeval.scoring.match.exact import ExactMatcher
from mmeval.scoring.match.template import TemplateMatcher
from mmeval.scoring.match.llm import LLMJudgeMatcher

MATCHER_REGISTRY = {
    "exact": ExactMatcher,
    "template": TemplateMatcher,
    "llm": LLMJudgeMatcher,
}

__all__ = [
    "MatchResult",
    "ExactMatcher",
    "TemplateMatcher",
    "LLMJudgeMatcher",
    "MATCHER_REGISTRY",
]
