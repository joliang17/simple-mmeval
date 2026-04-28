from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class MatchResult:
    is_match: bool
    matched: Optional[str] = None
    reason: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


class BaseMatcher:
    name = "base"

    def match(self, sample: Dict[str, Any], context: Dict[str, Any]) -> MatchResult:
        raise NotImplementedError

