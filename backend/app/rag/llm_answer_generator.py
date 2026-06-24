from typing import Optional

from backend.app.config import settings
from backend.app.rag.answer_service import (
    AnswerGenerator,
    SimpleExtractiveAnswerGenerator,
)
from backend.app.retrieval.retrieval_service import RetrievedChunk


SAFE_FALLBACK_ANSWER = (
    "I could not find enough reliable evidence in the provided documents "
    "to answer this confidently."
)


class LLMAnswerGenerationError(Exception):
    pass

class GroqGroundedAnswerGenerator:
    def __init__(
        self,
        model_name: str = "llama-3.3-70b-versatile",
        api_key: Optional[str] = None
    ):
        self.model_name = model_name
        self.generator_name = f"groq:{model_name}"

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

        self.client = Groq(api_key=resolved_api_key)

    def generate_answer(self, query: str, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return SAFE_FALLBACK_ANSWER

        context = build_context(chunks)

        messages = [
            {
                "role": "system",
                "content": build_system_prompt()
            },
            {
                "role": "user",
                "content": build_user_prompt(
                    query=query,
                    context=context
                )
            }
        ]

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0,
                max_completion_tokens=220
            )
        except Exception as exc:
            raise LLMAnswerGenerationError(
                f"Groq answer generation failed: {exc}"
            ) from exc

        answer = completion.choices[0].message.content

        if not answer:
            return SAFE_FALLBACK_ANSWER

        return clean_answer(answer)


def get_default_answer_generator() -> AnswerGenerator:
    provider = settings.default_llm_provider.strip().lower()

    if provider in {"mock", "extractive", "simple"}:
        return SimpleExtractiveAnswerGenerator()

    if provider == "groq":
        return GroqGroundedAnswerGenerator(
            model_name=settings.default_llm_model
        )

    raise LLMAnswerGenerationError(
        f"Unsupported LLM provider: {settings.default_llm_provider}"
    )


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
    return(
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
                text=chunk.text.strip()
            )
        )

    return "\n\n".join(context_blocks)

def clean_answer(answer: str) -> str:
    cleaned = " ".join(answer.strip().split())

    if not cleaned:
        return SAFE_FALLBACK_ANSWER

    return cleaned