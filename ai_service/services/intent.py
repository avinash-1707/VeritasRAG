import json
import logging
from typing import NamedTuple

from config import settings
from services.openrouter import complete

logger = logging.getLogger(__name__)

_INTENT_PROMPT = (
    'Classify the question and return a JSON object with three fields.\n\n'
    'FIELD 1 — "intent": one of:\n'
    '  "summary"      — ANY question asking about the document as a whole: its purpose, goal, objective,\n'
    '                   scope, audience, overview, main themes, or general description.\n'
    '                   This includes: summarize, overview, explain, describe the document, what is this about,\n'
    '                   what does this document do/cover/contain, what is the purpose/goal/objective/aim/intent\n'
    '                   of this document, what problem does this solve, who is this for, what is the scope.\n'
    '                   Examples: "What is this doc about?", "Summarize this", "What does this document cover?",\n'
    '                             "Give me an overview", "What is the main topic?",\n'
    '                             "What is the purpose of this document?",\n'
    '                             "What does this document aim to achieve?",\n'
    '                             "What are the goals of this document?",\n'
    '                             "What is the scope of this document?",\n'
    '                             "Who is the intended audience?",\n'
    '                             "What problem does this document address?",\n'
    '                             "What is the objective here?",\n'
    '                             "Can you explain what this document is about?",\n'
    '                             "Give me a brief description of this document"\n'
    '  "extraction"   — list every instance of something (dates, names, clauses, amounts, parties)\n'
    '                   Examples: "List all dates", "Extract all party names", "What obligations are mentioned?"\n'
    '  "comparison"   — compare or contrast two or more items from the document\n'
    '                   Examples: "Compare section A and B", "Differences between X and Y"\n'
    '  "boolean"      — yes/no question about whether something exists or is true in the document\n'
    '                   Examples: "Is X mentioned?", "Does the document contain Y?", "Was Z discussed?"\n'
    '  "definition"   — explain a term or concept as used in the document\n'
    '                   Examples: "What does X mean here?", "Define Y as used in the document"\n'
    '  "procedural"   — how-to or step-by-step process described in the document\n'
    '                   Examples: "How do I apply for X?", "What are the steps to complete Y?",\n'
    '                             "Walk me through the process of Z", "What is the procedure for X?",\n'
    '                             "How should I submit a claim?", "What do I need to do to renew?"\n'
    '  "analytical"   — why/what-does-it-mean questions requiring inference or reasoning across the document\n'
    '                   Examples: "Why does the document require X?", "What are the implications of clause Y?",\n'
    '                             "What does this policy suggest about Z?", "What patterns can you identify?",\n'
    '                             "What is the significance of X?", "What can we infer from section Y?"\n'
    '  "troubleshooting" — diagnosing a problem or finding resolution steps described in the document\n'
    '                   Examples: "Why is X not working?", "What should I do if Y fails?",\n'
    '                             "How do I resolve error Z?", "What causes X issue?",\n'
    '                             "What are common problems with Y?", "How do I fix X?"\n'
    '  "recommendation" — asking for a suggested course of action based on document content\n'
    '                   Examples: "What approach does the document recommend for X?",\n'
    '                             "Which option should I choose according to this?",\n'
    '                             "What does the document suggest I do about Y?",\n'
    '                             "What is the recommended way to handle Z?"\n'
    '  "factual"      — specific fact, name, date, clause, value, or detail from the document\n'
    '                   (NOT document-level questions — those are "summary")\n'
    '                   Examples: "What is the deadline?", "Who signed?", "What does clause 3.2 say?",\n'
    '                             "What is the penalty for late payment?", "When was this signed?"\n'
    '  "chitchat"     — greeting, thanks, or small talk with no document question\n'
    '                   Examples: "Hello", "Thanks!", "Great job", "How are you?"\n'
    '  "out_of_scope" — general world knowledge with no connection to any uploaded document\n'
    '                   This includes: geography, history, science, math, current events, coding help,\n'
    '                   definitions of common words, recipes, sports, entertainment — anything a search\n'
    '                   engine would answer without needing a specific document.\n'
    '                   Examples: "What is the capital of France?", "Tell me a joke",\n'
    '                             "Who is the president of the US?", "What is machine learning?",\n'
    '                             "How do I cook pasta?", "What year did WW2 end?"\n\n'
    'DISAMBIGUATION RULES:\n'
    '1. If the question could be answered from general knowledge without any specific document → "out_of_scope".\n'
    '2. If the question asks about the document itself at a high level (purpose, goal, objective, scope,\n'
    '   audience, what it covers, what it is about) → always "summary".\n'
    '3. Only use "factual" for specific named items, clauses, values, dates, or parties within the document.\n\n'
    'FIELD 2 — "hypothetical": ONLY if intent is "factual", write a 2-4 sentence formal document-style\n'
    'passage that would directly answer the question (HyDE technique for better retrieval).\n'
    'Empty string for all other intents.\n\n'
    'FIELD 3 — "standalone_query": if the question uses pronouns or references that require conversation\n'
    'history to understand (e.g., "tell me more about that", "what about section 3?",\n'
    '"expand on the previous point", "and what about X?"), rewrite it as a fully self-contained question.\n'
    'Empty string if the question is already standalone.\n\n'
    'Return JSON only.'
)

_RESPONSE_SCHEMA = {
    'type': 'object',
    'properties': {
        'intent': {
            'type': 'string',
            'enum': [
                'summary', 'extraction', 'comparison', 'boolean',
                'definition', 'procedural', 'analytical', 'troubleshooting',
                'recommendation', 'factual', 'chitchat', 'out_of_scope',
            ],
        },
        'hypothetical': {'type': 'string'},
        'standalone_query': {'type': 'string'},
    },
    'required': ['intent', 'hypothetical', 'standalone_query'],
    'additionalProperties': False,
}

_VALID_INTENTS = {
    'summary', 'extraction', 'comparison', 'boolean',
    'definition', 'procedural', 'analytical', 'troubleshooting',
    'recommendation', 'factual', 'chitchat', 'out_of_scope',
}


class IntentResult(NamedTuple):
    intent: str
    hypothetical: str
    standalone_query: str


def _get_models() -> list[str]:
    return [m.strip() for m in settings.openrouter_chat_models.split(',') if m.strip()]


def _build_history_snippet(history: list[dict]) -> str:
    if not history:
        return ''
    recent = history[-4:]
    lines = []
    for msg in recent:
        role = 'User' if msg['role'] == 'user' else 'Assistant'
        lines.append(f'{role}: {msg["content"][:200]}')
    return '\n\nRecent conversation:\n' + '\n'.join(lines)


async def classify_intent(question: str, history: list[dict] | None = None) -> IntentResult:
    """Classify intent; for factual queries also generates a HyDE passage in one call.

    Saves one round-trip vs calling classify + generate_hypothetical separately.
    Falls back to (factual, '', '') if all models fail.
    """
    history_snippet = _build_history_snippet(history or [])
    messages = [{'role': 'user', 'content': f'{_INTENT_PROMPT}{history_snippet}\n\nQuestion: {question}'}]
    response_format = {
        'type': 'json_schema',
        'json_schema': {'name': 'intent_result', 'strict': True, 'schema': _RESPONSE_SCHEMA},
    }

    for model in _get_models():
        try:
            response = await complete(
                model,
                messages,
                response_format=response_format,
                provider={'require_parameters': True},
                max_tokens=300,
                temperature=0.0,
            )
            data = json.loads(response)
            intent = data.get('intent', '')
            if intent in _VALID_INTENTS:
                return IntentResult(
                    intent=intent,
                    hypothetical=data.get('hypothetical', '') or '',
                    standalone_query=data.get('standalone_query', '') or '',
                )
            logger.warning('Intent classifier: unexpected value "%s" from %s', intent, model)
        except Exception as exc:
            logger.warning('Intent classifier: %s failed, trying next: %s', model, exc)
            continue

    logger.warning('Intent classifier: all models exhausted, defaulting to factual')
    return IntentResult(intent='factual', hypothetical='', standalone_query='')
