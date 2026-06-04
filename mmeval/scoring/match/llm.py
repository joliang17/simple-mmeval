import json
import os
import re
import time
from typing import Any, Dict

from mmeval.scoring.match.base import BaseMatcher, MatchResult
from mmeval.scoring.schema import normalize_text


class LLMJudgeMatcher(BaseMatcher):
    name = "llm"

    def __init__(self, args):
        self.provider = (args.judge_provider or "openai").strip().lower()
        self.model = args.judge_model
        self.max_retry = max(1, int(args.judge_max_retry))
        self.temperature = float(args.judge_temperature)
        self.include_reason = bool(args.judge_include_reason)
        self.client = self._build_client()
        if not self.model:
            raise ValueError("`--judge_model` is required when llm matcher is enabled.")

    def _build_client(self):
        try:
            from openai import AzureOpenAI, OpenAI
        except Exception as exc:
            raise RuntimeError("openai package is required for llm-as-judge matcher.") from exc

        if self.provider == "azure_openai":
            key = os.getenv("AZURE_OPENAI_KEY")
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            if not key or not endpoint:
                raise ValueError("AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT are required for azure_openai judge.")
            self.model = os.getenv("AZURE_OPENAI_DEPLOYNAME", self.model)
            return AzureOpenAI(
                api_key=key,
                azure_endpoint=endpoint,
                api_version="2024-02-15-preview",
            )

        key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        if not key:
            raise ValueError("OPENAI_API_KEY is required for openai judge.")
        kwargs = {"api_key": key}
        if base_url:
            kwargs["base_url"] = base_url
        return OpenAI(**kwargs)

    def match(self, sample: Dict[str, Any], context: Dict[str, Any]) -> MatchResult:
        # Keep more source detail for judge prompts: do light cleanup only.
        pred = normalize_text(context["pred"], extract_boxed=False, strip_latex_commands=False)
        gt = normalize_text(context["gt"], extract_boxed=False, strip_latex_commands=False)
        question = normalize_text(context["question"], extract_boxed=False, strip_latex_commands=False)
        question_type = context["question_type"]

        prompt = self._build_prompt(question_type, question, pred, gt, context)
        messages = [
            {"role": "system", "content": "You are a strict evaluator for VLM outputs."},
            {"role": "user", "content": prompt},
        ]

        parsed = self._call_with_retry(messages)
        if parsed is None:
            return MatchResult(is_match=False, reason="llm_judge_failed")

        is_correct = int(parsed.get("is_correct", 0)) == 1
        reason = parsed.get("reason") if self.include_reason else None
        if is_correct:
            return MatchResult(is_match=True, matched=str(context["gt"]), reason=reason)
        return MatchResult(is_match=False, matched=None, reason=reason)

    def _build_prompt(self, question_type, question, pred, gt, context):
        options = context.get("option_text_map", {})
        options_text = ""
        if question_type == "mcq" and options:
            rendered = [f"{k}. {v}" for k, v in options.items()]
            options_text = "\nOptions:\n" + "\n".join(rendered)

        reason_instruction = "Include a concise reason." if self.include_reason else "Set reason to an empty string."
        return (
            "Judge whether MODEL_RESPONSE is correct given QUESTION and GROUND_TRUTH.\n"
            f"QuestionType: {question_type}\n"
            f"QUESTION: {question}\n"
            f"{options_text}\n"
            f"MODEL_RESPONSE: {pred}\n"
            f"GROUND_TRUTH: {gt}\n\n"
            "Return JSON only with schema:\n"
            '{"is_correct": 0_or_1, "reason": "string"}\n'
            f"{reason_instruction}\n"
            "No markdown, no extra fields."
        )

    def _call_with_retry(self, messages):
        for _ in range(self.max_retry):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                )
                text = resp.choices[0].message.content or ""
                parsed = self._parse_response(text)
                if parsed is not None:
                    return parsed
            except Exception:
                time.sleep(1.5)
        return None

    def _parse_response(self, text):
        text = text.strip()
        if not text:
            return None
        try:
            obj = json.loads(text)
            return self._normalize_judge_obj(obj)
        except Exception:
            pass

        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group(0))
                return self._normalize_judge_obj(obj)
            except Exception:
                return None
        return None

    @staticmethod
    def _normalize_judge_obj(obj):
        if not isinstance(obj, dict):
            return None
        raw = obj.get("is_correct", 0)
        if isinstance(raw, bool):
            value = 1 if raw else 0
        else:
            try:
                value = int(raw)
            except Exception:
                value = 0
        reason = obj.get("reason", "")
        return {"is_correct": 1 if value == 1 else 0, "reason": str(reason)}

