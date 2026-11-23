"""
Extracts discrete, checkable factual claims from raw input text, and
filters out opinion/commentary that can't be fact-checked.
"""
import json
import re

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from app.config import settings


EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You extract discrete, checkable factual claims from text.

A checkable factual claim is a specific, verifiable statement about the
world (a statistic, an event, a quote, a cause-effect relationship).

NOT a checkable claim:
- Opinions ("this policy is bad")
- Predictions about the future ("this will fail")
- Vague statements with no specific fact to verify ("things are getting worse")

Return ONLY a JSON array of objects, each with:
  - "claim": the claim rewritten as a single, self-contained, checkable statement
  - "original_span": the exact text span from the input this claim came from

If there are no checkable claims, return an empty array [].
Do not include any text outside the JSON array."""),
    ("user", "{text}")
])

EXTRACTION_PROMPT_TEXT = """You extract discrete, checkable factual claims from text.

A checkable factual claim is a specific, verifiable statement about the
world (a statistic, an event, a quote, a cause-effect relationship).

NOT a checkable claim:
- Opinions ("this policy is bad")
- Predictions about the future ("this will fail")
- Vague statements with no specific fact to verify ("things are getting worse")

Return ONLY a JSON array of objects, each with:
  - "claim": the claim rewritten as a single, self-contained, checkable statement
  - "original_span": the exact text span from the input this claim came from

If there are no checkable claims, return an empty array [].
Do not include any text outside the JSON array."""


class LocalTextGenerator:
    def __init__(self, model_name: str, max_new_tokens: int = 256):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=-1,
        )
        self.max_new_tokens = max_new_tokens

    def generate_text(self, prompt: str) -> str:
        outputs = self.generator(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return outputs[0]["generated_text"]


def get_llm():
    if settings.USE_LOCAL_LLM:
        print(f"Using local LLM model: {settings.LOCAL_LLM_MODEL}")
        return LocalTextGenerator(
            model_name=settings.LOCAL_LLM_MODEL,
            max_new_tokens=settings.LOCAL_LLM_MAX_NEW_TOKENS,
        )

    return ChatOpenAI(
        model=settings.LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
    )


def _safe_json_array_from_text(raw_text: str):
    if raw_text is None:
        return None

    cleaned = raw_text.strip()
    if not cleaned:
        return None

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start : end + 1]

    try:
        value = json.loads(cleaned)
        if isinstance(value, list):
            return value
    except Exception:
        pass

    return None


def _fallback_claims(text: str) -> list[dict]:
    plain = (text or "").strip()
    if not plain:
        return []

    normalized = plain.rstrip("?")
    if normalized:
        if re.match(r"^(what|who|when|where|which|why|how|is|are|do|does|did|can|could|should|would|tell|explain|summarize)\b", normalized, flags=re.I):
            claim_text = normalized.strip()
            if claim_text.endswith(":"):
                claim_text = claim_text[:-1]
            return [{"claim": claim_text, "original_span": text}]

    # Fallback for non-question statements: keep the first meaningful sentence.
    sentence = re.split(r"(?<=[.!?])\s+", plain)[0].strip()
    if sentence:
        return [{"claim": sentence, "original_span": text}]
    return [{"claim": plain, "original_span": text}]


def extract_claims(text: str) -> list[dict]:
    llm = get_llm()
    parser = JsonOutputParser()

    if settings.USE_LOCAL_LLM:
        prompt = f"{EXTRACTION_PROMPT_TEXT}\n\nInput:\n{text}"
        try:
            raw_output = llm.generate_text(prompt)
            result = _safe_json_array_from_text(raw_output)
            if isinstance(result, list):
                if result:
                    return result
                if text and text.strip():
                    return _fallback_claims(text)
                return []
            return _fallback_claims(text)
        except Exception as e:
            print(f"Claim extraction failed: {e}")
            return _fallback_claims(text)

    chain = EXTRACTION_PROMPT | llm | parser
    try:
        result = chain.invoke({"text": text})
        if not isinstance(result, list):
            return _fallback_claims(text)
        if result:
            return result
        if text and text.strip():
            return _fallback_claims(text)
        return []
    except Exception as e:
        print(f"Claim extraction failed: {e}")
        return _fallback_claims(text)


if __name__ == "__main__":
    test_input = (
        "The new policy cut national unemployment in half within a year, "
        "which honestly makes it one of the best decisions this administration "
        "has made. Meanwhile, TechCorp laid off 12,000 workers last quarter."
    )

    claims = extract_claims(test_input)

    print(f"Input:\n{test_input}\n")
    print(f"Extracted {len(claims)} claim(s):\n")
    for i, c in enumerate(claims, 1):
        print(f"{i}. {c.get('claim')}")
        print(f"   (from: \"{c.get('original_span')}\")\n")