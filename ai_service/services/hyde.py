import logging

from config import settings
from services.openrouter import complete

logger = logging.getLogger(__name__)

_HYDE_PROMPT = (
    'Write a short passage (2-4 sentences) that would appear in a document and directly '
    'answer the following question. Use formal, document-like language. '
    'Output only the passage, no preamble or explanation.'
)


def _get_models() -> list[str]:
    return [m.strip() for m in settings.openrouter_chat_models.split(',') if m.strip()]


async def generate_hypothetical(question: str) -> str:
    """Generate a hypothetical document passage that would answer the question.

    The passage uses corpus-like vocabulary, improving ANN and BM25 recall
    for indirect or paraphrased queries (HyDE technique).
    Tries each model in openrouter_chat_models in order, falls back to raw question
    if all models fail so retrieval still runs.
    """
    prompt = f'{_HYDE_PROMPT}\n\nQuestion: {question}'

    for model in _get_models():
        try:
            text = (await complete(
                model,
                [{'role': 'user', 'content': prompt}],
                max_tokens=150,
                temperature=0.1,
            )).strip()
            return text if text else question
        except Exception as exc:
            logger.warning('HyDE: model %s failed, trying next: %s', model, exc)
            continue

    logger.warning('HyDE: all models exhausted, falling back to raw question')
    return question
