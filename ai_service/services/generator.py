import json
import logging
from typing import Any, AsyncIterator

from config import settings
from services.openrouter import stream

logger = logging.getLogger(__name__)

_LOW_GROUNDING_THRESHOLD = 0.15
_MAX_HISTORY_TURNS = 10

_SYSTEM_PROMPT = (
    'You are a helpful document assistant. '
    'Answer the user\'s question using only the source excerpts provided in the message. '
    'You may synthesize information across multiple excerpts to form a complete answer. '
    'If the excerpts genuinely do not contain enough information to answer the question, '
    'say so briefly and specifically — explain what is missing rather than giving a generic refusal. '
    'Do not use outside knowledge or invent facts not present in the excerpts.'
)

_SUMMARY_SYSTEM_PROMPT = (
    'You are a helpful document assistant. '
    'Based on the source excerpts provided, answer the user\'s question about the document. '
    'If the question asks for a summary, overview, or general description: cover the main themes, '
    'purpose, key points, and notable details. Use bullet points or clear sections for readability. '
    'If the question asks about the document\'s purpose, goal, objective, or scope: synthesize an answer '
    'from the content — you may infer purpose from what the document discusses even if not stated verbatim. '
    'Note at the end if the excerpts are a sample and the full document may contain additional information. '
    'Do not use outside knowledge unrelated to the document content.'
)

_EXTRACTION_SYSTEM_PROMPT = (
    'You are a professional document analysis assistant. '
    'Extract and list every instance of what the user is asking about from the source excerpts. '
    'Format the output as a structured, numbered or bulleted list. '
    'Include the source reference (document name and page) for each item. '
    'Do not infer, group, or add information not explicitly present in the excerpts.'
)

_COMPARISON_SYSTEM_PROMPT = (
    'You are a professional document analysis assistant. '
    'Compare the items, sections, or concepts the user is asking about using only the source excerpts. '
    'Present the comparison in a structured format — a table or side-by-side bullet points. '
    'Clearly highlight similarities and differences. '
    'Do not use outside knowledge or invent details not present in the excerpts.'
)

_BOOLEAN_SYSTEM_PROMPT = (
    'You are a professional document analysis assistant. '
    'Answer the user\'s question with "Yes" or "No" as the first word, '
    'followed by a single sentence citing the specific part of the document that supports the answer. '
    'Do not elaborate beyond what the excerpts directly support. '
    'Do not use outside knowledge.'
)

_DEFINITION_SYSTEM_PROMPT = (
    'You are a professional document analysis assistant. '
    'Explain the term or concept the user is asking about strictly as it is defined or used in the source excerpts. '
    'Do not use dictionary definitions or outside knowledge. '
    'If the document does not explicitly define the term but uses it in context, describe how it is used. '
    'Keep the response concise and precise.'
)

_PROCEDURAL_SYSTEM_PROMPT = (
    'You are a professional document analysis assistant. '
    'The user is asking how to perform a process or follow a procedure described in the document. '
    'Extract the relevant steps from the source excerpts and present them as a clear, numbered list. '
    'Preserve the sequence exactly as described in the document. '
    'If prerequisites or conditions are mentioned, list them before the steps. '
    'Do not add steps, infer missing steps, or use outside knowledge.'
)

_ANALYTICAL_SYSTEM_PROMPT = (
    'You are a professional document analysis assistant. '
    'The user is asking an analytical question that requires reasoning across the source excerpts. '
    'Synthesize the relevant evidence from the excerpts to form a reasoned answer. '
    'Clearly distinguish between what the document explicitly states and what can be reasonably inferred. '
    'Label inferences with phrases like "This suggests..." or "Based on the document, it appears...". '
    'Do not use outside knowledge or introduce facts not present in the excerpts.'
)

_TROUBLESHOOTING_SYSTEM_PROMPT = (
    'You are a professional document analysis assistant. '
    'The user is describing a problem and looking for resolution steps or causes described in the document. '
    'Structure your response as: (1) Likely cause(s) based on the excerpts, '
    '(2) Resolution steps in numbered order, (3) Any conditions or warnings mentioned. '
    'Only include causes and steps that are explicitly or clearly implied by the source excerpts. '
    'Do not invent steps or use outside knowledge.'
)

_RECOMMENDATION_SYSTEM_PROMPT = (
    'You are a professional document analysis assistant. '
    'The user is asking what the document recommends or suggests for their situation. '
    'Extract the relevant guidance, recommendation, or suggested approach from the source excerpts. '
    'State the recommendation clearly, then cite the specific part of the document it comes from. '
    'If the document presents multiple options, list them with any conditions or trade-offs stated. '
    'Do not recommend beyond what the document explicitly states. Do not use outside knowledge.'
)

_NO_CONTEXT_SYSTEM_PROMPT = (
    'You are a professional document analysis assistant. '
    'The retrieved excerpts do not appear to contain information directly relevant to the user\'s question. '
    'Respond with a single concise sentence acknowledging this, and suggest the user try rephrasing '
    'or ask about specific sections or topics that may be covered. '
    'Use professional language. Do not use outside knowledge.'
)


def _get_models() -> list[str]:
    return [m.strip() for m in settings.openrouter_chat_models.split(',') if m.strip()]


def _build_context(chunks: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for i, c in enumerate(chunks, start=1):
        parts.append(
            f'[Source {i}] {c["document_name"]}, page {c["page_number"]}\n{c["content"]}'
        )
    return '\n\n---\n\n'.join(parts)


_REFUSAL_MARKER = 'do not contain sufficient information'


def _build_contents(
    question: str,
    context: str,
    history: list[dict[str, Any]],
) -> list[dict[str, str]]:
    contents: list[dict[str, str]] = []

    # Strip assistant refusals: they bias the next model response even when the
    # current turn has valid excerpts. Keep the user message so conversation
    # thread stays intact for follow-up resolution.
    filtered: list[dict[str, Any]] = []
    for msg in history[-_MAX_HISTORY_TURNS:]:
        if msg['role'] == 'assistant' and _REFUSAL_MARKER in msg['content']:
            pass  # drop only the refusal; preceding user message is retained
        else:
            filtered.append(msg)

    for msg in filtered:
        role = 'user' if msg['role'] == 'user' else 'assistant'
        contents.append({'role': role, 'content': msg['content']})

    contents.append({'role': 'user', 'content': f'Source excerpts:\n\n{context}\n\nQuestion: {question}'})
    return contents



async def _stream_with_fallback(
    contents: list[dict[str, str]],
    answer_parts: list[str],
    system_prompt: str = _SYSTEM_PROMPT,
) -> AsyncIterator[str]:
    models = _get_models()
    last_exc: Exception | None = None

    for model in models:
        try:
            messages = [{'role': 'system', 'content': system_prompt}, *contents]
            async for token in stream(model, messages):
                answer_parts.append(token)
                yield f'data: {json.dumps({"token": token})}\n\n'
            yield f'__model__:{model}'
            return
        except Exception as exc:
            logger.warning('Model %s failed (%s), trying next', model, exc)
            last_exc = exc
            continue

    raise RuntimeError('All OpenRouter models exhausted') from last_exc


async def generate(
    question: str,
    chunks: list[dict[str, Any]],
    history: list[dict[str, Any]] | None = None,
    intent: str = 'factual',
) -> AsyncIterator[str]:
    raw_scores = [float(c.get('rerank_score', c.get('similarity_score', 0.0))) for c in chunks]
    top_sim = max(raw_scores, default=0.0)

    _stratified_intents = {'summary', 'extraction'}
    if intent in _stratified_intents:
        # Stratified chunks have flat scores — only refuse if empty
        low_confidence = len(chunks) == 0
        grounding_score = round(sum(raw_scores) / len(raw_scores), 4) if raw_scores else 0.0
    else:
        low_confidence = top_sim < _LOW_GROUNDING_THRESHOLD
        grounding_score = round(top_sim, 4)

    _prompt_map = {
        'summary': _SUMMARY_SYSTEM_PROMPT,
        'extraction': _EXTRACTION_SYSTEM_PROMPT,
        'comparison': _COMPARISON_SYSTEM_PROMPT,
        'boolean': _BOOLEAN_SYSTEM_PROMPT,
        'definition': _DEFINITION_SYSTEM_PROMPT,
        'procedural': _PROCEDURAL_SYSTEM_PROMPT,
        'analytical': _ANALYTICAL_SYSTEM_PROMPT,
        'troubleshooting': _TROUBLESHOOTING_SYSTEM_PROMPT,
        'recommendation': _RECOMMENDATION_SYSTEM_PROMPT,
    }
    system_prompt = _prompt_map.get(intent, _SYSTEM_PROMPT)
    context = _build_context(chunks)
    contents = _build_contents(question, context, history or [])

    citations = [
        {
            'chunk_id': c['chunk_id'],
            'document_name': c['document_name'],
            'page_number': c['page_number'],
            'similarity_score': round(float(c.get('rerank_score', c.get('similarity_score', 0.0))), 4),
            'content': c.get('content', ''),
        }
        for c in chunks
    ]

    answer_parts: list[str] = []
    model_used = _get_models()[0]

    if low_confidence:
        no_context_contents = [{'role': 'user', 'content': f'Question: {question}'}]
        try:
            async for event in _stream_with_fallback(no_context_contents, answer_parts, _NO_CONTEXT_SYSTEM_PROMPT):
                if event.startswith('__model__:'):
                    model_used = event[len('__model__:'):]
                else:
                    yield event
        except Exception as exc:
            logger.exception('Low-confidence fallback failed: %s', exc)
            answer = 'The provided documents do not contain information to answer this question.'
            yield f'data: {json.dumps({"token": answer})}\n\n'
            answer_parts.append(answer)
    else:
        try:
            async for event in _stream_with_fallback(contents, answer_parts, system_prompt):
                if event.startswith('__model__:'):
                    model_used = event[len('__model__:'):]
                else:
                    yield event
        except Exception as exc:
            logger.exception('All models failed: %s', exc)
            error_msg = 'Service temporarily unavailable. Please try again in a moment.'
            yield f'data: {json.dumps({"token": error_msg})}\n\n'
            answer_parts.append(error_msg)

    full_answer = ''.join(answer_parts)
    final = {
        'answer': full_answer,
        'citations': citations,
        'grounding_score': grounding_score,
        'top_similarity_score': round(top_sim, 4),
        'model_used': model_used,
    }
    yield f'data: {json.dumps({"done": True, **final})}\n\n'
