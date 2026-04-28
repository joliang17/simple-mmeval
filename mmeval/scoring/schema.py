import re
import unicodedata
from typing import Any, Dict, List, Optional


MCQ_OPTIONS_DEFAULT = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _to_text(value: Any) -> str:
    if isinstance(value, list):
        if not value:
            return ""
        value = value[0]
    if value is None:
        return ""
    return str(value)


def _extract_boxed_content(text: str) -> str:
    marker = "\\boxed{"
    start = text.find(marker)
    if start < 0:
        return text
    i = start + len(marker)
    depth = 1
    content = []
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(content).strip()
        content.append(ch)
        i += 1
    return text


def normalize_text(
    value: Any,
    lower: bool = False,
    extract_boxed: bool = True,
    strip_latex_commands: bool = True,
) -> str:
    text = _to_text(value)

    # Keep normalization mostly syntax-level and semantics-preserving.
    text = unicodedata.normalize("NFKC", text)
    if extract_boxed:
        text = _extract_boxed_content(text)

    # Remove CoT / answer wrappers if models output tagged sections.
    text = re.sub(r"</?(?:think|answer)\s*>", " ", text, flags=re.IGNORECASE)

    # Remove markdown code fences while preserving body text.
    text = text.replace("```", " ")

    # Common inference artifacts.
    text = text.replace("<|end_of_sentence|>", "")
    text = text.replace("<|end▁of▁sentence|>", "")
    text = text.replace("</s>", "")
    text = text.replace("<CONCLUSION>", "")
    text = text.replace("</CONCLUSION>", "")
    text = text.replace("Falcon: ", "")

    # Light LaTeX cleanup inspired by math-verify style normalization.
    if strip_latex_commands:
        text = re.sub(r"\\(?:left|right|displaystyle|mathrm|textbf|textit|text)\b", " ", text)
        text = re.sub(r"\\(?:,|;|!|\:)", " ", text)  # spacing commands
        text = text.replace("\\%", "%")

    # Normalize common unicode/operator variants.
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = text.replace("×", "*").replace("·", "*")
    text = text.replace("÷", "/")

    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    if lower:
        text = text.lower()
    return text


def normalize_for_exact(value: Any) -> str:
    return normalize_text(value, lower=True).strip()


def normalize_open_answer(value: Any) -> str:
    text = normalize_for_exact(value)
    text = re.sub(r"^[\"'`]+|[\"'`]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def get_field(sample: Dict[str, Any], field: str, default: Any = None) -> Any:
    if not field:
        return default
    if field == "messages[-1].response":
        messages = sample.get("messages") or []
        if not messages:
            return default
        last = messages[-1]
        if not isinstance(last, dict):
            return default
        return last.get("response", default)
    return sample.get(field, default)


def get_question(sample: Dict[str, Any]) -> str:
    messages = sample.get("messages") or []
    if messages and isinstance(messages[0], dict):
        q = messages[0].get("question")
        if q is not None:
            return str(q)
    return str(sample.get("question", ""))


def extract_options(sample: Dict[str, Any]) -> List[str]:
    messages = sample.get("messages") or []
    if messages and isinstance(messages[0], dict):
        msg = messages[0]
        choices = msg.get("choices")
        if isinstance(choices, list) and choices:
            return [str(c).strip().upper() for c in choices]
        options = msg.get("options")
        if isinstance(options, dict) and options:
            return [str(k).strip().upper() for k in options.keys()]

    if isinstance(sample.get("choices"), list) and sample["choices"]:
        return [str(c).strip().upper() for c in sample["choices"]]
    if isinstance(sample.get("options"), dict) and sample["options"]:
        return [str(k).strip().upper() for k in sample["options"].keys()]

    dynamic = [x for x in MCQ_OPTIONS_DEFAULT if x in sample]
    return dynamic or MCQ_OPTIONS_DEFAULT[:6]


def infer_question_type(
    sample: Dict[str, Any],
    score_force_question_type: str = "auto",
    score_type_field: Optional[str] = None,
) -> str:
    forced = (score_force_question_type or "auto").strip().lower()
    if forced in {"mcq", "open"}:
        return forced

    if score_type_field:
        raw = str(sample.get(score_type_field, "")).strip().lower()
        if raw in {"mcq", "mc", "multi-choice", "multiple_choice"}:
            return "mcq"
        if raw in {"open", "open-ended", "open_ended", "free"}:
            return "open"

    options = extract_options(sample)
    if options:
        return "mcq"
    return "open"


def extract_mcq_option(text: Any, candidates: List[str]) -> Optional[str]:
    clean = normalize_text(text)
    if not clean:
        return None

    uppercase = clean.strip().upper()
    if uppercase in candidates:
        return uppercase
    if len(uppercase) >= 2 and uppercase[0] in candidates and uppercase[1] in {".", ")", ":"}:
        return uppercase[0]

    patterns = [
        r"^(?:answer|option|choice)?\s*[:\-]?\s*([A-Z])(?:[\s\.\),:;]|$)",
        r"(?:final answer|the answer is|answer is|choice is|option is)\s*[:\-]?\s*([A-Z])(?:[\s\.\),:;]|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, uppercase, re.IGNORECASE)
        if match:
            candidate = match.group(1).upper()
            if candidate in candidates:
                return candidate
    return None

