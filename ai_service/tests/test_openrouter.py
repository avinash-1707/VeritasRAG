import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault('DATABASE_URL', 'postgresql://test:test@localhost:5432/test')
os.environ.setdefault('INTERNAL_API_KEY', 'test-key')
os.environ.setdefault('GOOGLE_API_KEY', 'test-key')
os.environ.setdefault('OPENROUTER_API_KEY', 'openrouter-test-key')

import httpx

from services import openrouter


def test_complete_sends_openrouter_request(monkeypatch):
    request_data = {}

    def handler(request: httpx.Request) -> httpx.Response:
        request_data['authorization'] = request.headers['Authorization']
        request_data['payload'] = json.loads(request.content)
        return httpx.Response(200, json={'choices': [{'message': {'content': 'Answer'}}]})

    original_async_client = httpx.AsyncClient
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        openrouter.httpx,
        'AsyncClient',
        lambda **kwargs: original_async_client(transport=transport, **kwargs),
    )

    answer = asyncio.run(openrouter.complete('google/gemini-3.1-flash-lite', [{'role': 'user', 'content': 'Hi'}]))

    assert answer == 'Answer'
    assert request_data == {
        'authorization': f'Bearer {openrouter.settings.openrouter_api_key}',
        'payload': {
            'model': 'google/gemini-3.1-flash-lite',
            'messages': [{'role': 'user', 'content': 'Hi'}],
        },
    }


def test_stream_parses_openrouter_sse(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)['stream'] is True
        return httpx.Response(
            200,
            content=(
                'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
                'data: {"choices":[{"delta":{"content":" world"}}]}\n\n'
                'data: [DONE]\n\n'
            ),
        )

    original_async_client = httpx.AsyncClient
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        openrouter.httpx,
        'AsyncClient',
        lambda **kwargs: original_async_client(transport=transport, **kwargs),
    )

    async def collect_tokens() -> list[str]:
        return [token async for token in openrouter.stream('model', [{'role': 'user', 'content': 'Hi'}])]

    assert asyncio.run(collect_tokens()) == ['Hello', ' world']
