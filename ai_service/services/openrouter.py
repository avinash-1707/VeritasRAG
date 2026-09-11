import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from config import settings

_CHAT_COMPLETIONS_URL = 'https://openrouter.ai/api/v1/chat/completions'
_STREAM_TIMEOUT = httpx.Timeout(60.0, read=None)


def _headers() -> dict[str, str]:
    return {
        'Authorization': f'Bearer {settings.openrouter_api_key}',
        'Content-Type': 'application/json',
    }


async def complete(
    model: str,
    messages: list[dict[str, str]],
    **options: Any,
) -> str:
    payload = {'model': model, 'messages': messages, **options}
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(_CHAT_COMPLETIONS_URL, headers=_headers(), json=payload)
        response.raise_for_status()

    data = response.json()
    content = data['choices'][0]['message']['content']
    if not isinstance(content, str):
        raise RuntimeError(f'OpenRouter returned non-text content for {model}')
    return content


async def stream(
    model: str,
    messages: list[dict[str, str]],
    **options: Any,
) -> AsyncIterator[str]:
    payload = {'model': model, 'messages': messages, 'stream': True, **options}
    async with httpx.AsyncClient(timeout=_STREAM_TIMEOUT) as client:
        async with client.stream('POST', _CHAT_COMPLETIONS_URL, headers=_headers(), json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith('data:'):
                    continue

                event = line[5:].strip()
                if event == '[DONE]':
                    return

                data = json.loads(event)
                if error := data.get('error'):
                    raise RuntimeError(f'OpenRouter stream failed for {model}: {error.get("message", error)}')

                choices = data.get('choices', [])
                if choices:
                    content = choices[0].get('delta', {}).get('content')
                    if content:
                        yield content
