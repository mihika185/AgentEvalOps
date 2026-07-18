import random
import time
from typing import Any, Literal, Optional

from backend.app.config import settings
from backend.app.logging_config import get_logger
from backend.app.rag.answer_service import (
    AnswerGenerator,
    SimpleExtractiveAnswerGenerator,
)
from backend.app.retrieval.retrieval_service import RetrievedChunk

logger = get_logger(__name__)

SAFE_FALLBACK_ANSWER = (
    "I could not find enough reliable evidence in the provided documents "
    "to answer this confidently."
)

class LLMAnswerGenerationError(Exception):
    pass

AnswerGeneratorProvider = Literal["extractive", "groq"]

def normalize_answer_provider(provider_name: str) -> AnswerGeneratorProvider:
    provider = str(provider_name or "").strip().lower()

    if provider in {"mock", "simple", "extractive"}:
        return "extractive"

    if provider == "groq":
        return "groq"

    raise LLMAnswerGenerationError(
        f"Unsupported LLM provider: {provider_name}"
    )

def resolve_answer_model(
    provider_name: str,
    model_name: str,
) -> str:
    provider = normalize_answer_provider(provider_name)

    if provider == "extractive":
        return SimpleExtractiveAnswerGenerator.generator_name

    cleaned_model_name = str(model_name or "").strip()

    if not cleaned_model_name:
        raise LLMAnswerGenerationError(
            "DEFAULT_LLM_MODEL must be set when using a hosted LLM provider"
        )

    return cleaned_model_name

def get_configured_answer_provider() -> AnswerGeneratorProvider:
    return normalize_answer_provider(settings.default_llm_provider)

def get_configured_answer_model() -> str:
    return resolve_answer_model(
        provider_name=settings.default_llm_provider,
        model_name=settings.default_llm_model,
    )

def create_answer_generator(
    provider_name: str,
    model_name: str,
    api_key: Optional[str] = None,
) -> AnswerGenerator:
    provider = normalize_answer_provider(provider_name)
    resolved_model_name = resolve_answer_model(
        provider_name=provider,
        model_name=model_name,
    )

    if provider == "extractive":
        return SimpleExtractiveAnswerGenerator()

    return GroqGroundedAnswerGenerator(
        model_name=resolved_model_name,
        api_key=api_key,
    )

class GroqGroundedAnswerGenerator:
    def __init__(
        self,
        model_name: str = "llama-3.3-70b-versatile",
        api_key: Optional[str] = None,
    ):
        self.model_name = model_name
        self.generator_name = f"groq:{model_name}"
        self.last_usage: Optional[dict[str, int]] = None

        self.last_retry_count = 0
        self.last_retry_delays_seconds: list[float] = []
        self.last_retry_reasons: list[str] = []

        self.max_retries = settings.groq_max_retries
        self.retry_base_seconds = settings.groq_retry_base_seconds
        self.retry_max_seconds = settings.groq_retry_max_seconds
        self.retry_jitter_seconds = settings.groq_retry_jitter_seconds

        try:
            from groq import Groq
        except ImportError as exc:
            raise LLMAnswerGenerationError(
                "groq is not installed. Install it with: "
                "python3 -m pip install \"groq\""
            ) from exc

        resolved_api_key = api_key or settings.groq_api_key

        if not resolved_api_key:
            raise LLMAnswerGenerationError(
                "GROQ_API_KEY is not set. Add it to your .env file or use "
                "DEFAULT_LLM_PROVIDER=mock for local testing."
            )

        self.client = Groq(
            api_key=resolved_api_key,
            max_retries=0,
        )

    def generate_answer(self, query: str, chunks: list[RetrievedChunk]) -> str:
        self.last_usage = None
        self.last_retry_count = 0
        self.last_retry_delays_seconds = []
        self.last_retry_reasons = []

        if not chunks:
            return SAFE_FALLBACK_ANSWER

        context = build_context(chunks)

        messages = [
            {
                "role": "system",
                "content": build_system_prompt(),
            },
            {
                "role": "user",
                "content": build_user_prompt(
                    query=query,
                    context=context,
                ),
            },
        ]

        completion = self.create_completion_with_retries(messages)

        self.last_usage = extract_completion_usage(completion)

        answer = completion.choices[0].message.content

        if not answer:
            return SAFE_FALLBACK_ANSWER

        return clean_answer(answer)

    def create_completion_with_retries(
        self,
        messages: list[dict[str, str]],
    ) -> Any:
        retry_number = 0

        while True:
            try:
                return self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0,
                    max_completion_tokens=220,
                )
            except Exception as exc:
                if (
                    retry_number >= self.max_retries
                    or not is_retryable_groq_error(exc)
                ):
                    attempt_count = retry_number + 1

                    raise LLMAnswerGenerationError(
                        "Groq answer generation failed after "
                        f"{attempt_count} attempt(s): {exc}"
                    ) from exc

                delay_seconds = calculate_retry_delay_seconds(
                    exc=exc,
                    retry_number=retry_number,
                    base_seconds=self.retry_base_seconds,
                    max_seconds=self.retry_max_seconds,
                    jitter_seconds=self.retry_jitter_seconds,
                )

                retry_number += 1
                reason = retry_reason(exc)

                self.last_retry_count = retry_number
                self.last_retry_delays_seconds.append(delay_seconds)
                self.last_retry_reasons.append(reason)

                logger.warning(
                    "Retrying Groq answer generation after %ss "
                    "(%s, retry %s/%s)",
                    delay_seconds,
                    reason,
                    retry_number,
                    self.max_retries,
                )

                time.sleep(delay_seconds)

def is_retryable_groq_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)

    if status_code in {408, 409, 429}:
        return True

    if isinstance(status_code, int) and status_code >= 500:
        return True

    return exc.__class__.__name__ in {
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "RateLimitError",
    }


def calculate_retry_delay_seconds(
    exc: Exception,
    retry_number: int,
    base_seconds: float,
    max_seconds: float,
    jitter_seconds: float,
) -> float:
    retry_after_seconds = read_retry_after_seconds(exc)

    if retry_after_seconds is not None:
        base_delay = retry_after_seconds
    else:
        base_delay = base_seconds * (2 ** retry_number)

    jitter = (
        random.uniform(0.0, jitter_seconds)
        if jitter_seconds > 0
        else 0.0
    )

    return round(
        min(base_delay + jitter, max_seconds),
        3,
    )


def read_retry_after_seconds(
    exc: Exception,
) -> Optional[float]:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)

    if not headers:
        return None

    retry_after = headers.get("retry-after")

    if retry_after is None:
        return None

    try:
        return max(float(retry_after), 0.0)
    except (TypeError, ValueError):
        return None


def retry_reason(exc: Exception) -> str:
    status_code = getattr(exc, "status_code", None)

    if status_code is not None:
        return f"http_{status_code}"

    return exc.__class__.__name__


def get_default_answer_generator() -> AnswerGenerator:
    return create_answer_generator(
        provider_name=get_configured_answer_provider(),
        model_name=get_configured_answer_model(),
    )

def extract_completion_usage(completion: Any) -> Optional[dict[str, int]]:
    usage = getattr(completion, "usage", None)

    if usage is None:
        return None

    prompt_tokens = read_usage_value(usage, "prompt_tokens")
    completion_tokens = read_usage_value(usage, "completion_tokens")
    total_tokens = read_usage_value(usage, "total_tokens")

    if prompt_tokens is None or completion_tokens is None:
        return None

    if total_tokens is None:
        total_tokens = prompt_tokens + completion_tokens

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }

def read_usage_value(usage: Any, key: str) -> Optional[int]:
    if isinstance(usage, dict):
        value = usage.get(key)
    else:
        value = getattr(usage, key, None)

    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None

def build_system_prompt() -> str:
    return (
        "You are a grounded RAG answer generator. "
        "Use only the provided context to answer the question. "
        "Do not use outside knowledge. "
        "If the context does not contain enough evidence, return exactly: "
        f"{SAFE_FALLBACK_ANSWER} "
        "Prefer the same wording used in the context when possible. "
        "Keep the answer concise and factual."
    )

def build_user_prompt(query: str, context: str) -> str:
    return (
        "Question:\n"
        f"{query}\n\n"
        "Retrieved context:\n"
        f"{context}\n\n"
        "Instructions:\n"
        "- Answer using only the retrieved context.\n"
        "- Do not invent facts.\n"
        "- Do not mention unsupported details.\n"
        "- Do not include source IDs in the answer; sources are handled separately.\n"
        "- If the answer is not clearly supported, use the exact fallback sentence."
    )

def build_context(chunks: list[RetrievedChunk]) -> str:
    context_blocks: list[str] = []

    for index, chunk in enumerate(chunks, start=1):
        context_blocks.append(
            "[Source {source_number}]\n"
            "chunk_id: {chunk_id}\n"
            "document_id: {document_id}\n"
            "retrieval_score: {score:.4f}\n"
            "text:\n{text}".format(
                source_number=index,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                score=chunk.score,
                text=chunk.text.strip(),
            )
        )

    return "\n\n".join(context_blocks)

def clean_answer(answer: str) -> str:
    cleaned = " ".join(answer.strip().split())

    if not cleaned:
        return SAFE_FALLBACK_ANSWER

    return cleaned