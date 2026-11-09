import asyncio
import json
import unittest
from unittest.mock import patch

import httpx

import router


class CapturingClient:
    """记录真正发给上游的请求体，并回一个最简正常流。"""

    sent = []
    closed = 0

    def __init__(self, *args, **kwargs):
        pass

    def build_request(self, method, url, **kwargs):
        type(self).sent.append(kwargs.get("json"))
        return httpx.Request(method, url, **kwargs)

    async def send(self, request, stream=False):
        return httpx.Response(
            200,
            request=request,
            headers={"content-type": "text/event-stream"},
            stream=httpx.ByteStream(
                b'data: {"choices":[{"index":0,"delta":{"content":"ok"}}]}\n\n'
                b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
                b'"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}\n\n'
                b"data: [DONE]\n\n"
            ),
        )

    async def aclose(self):
        type(self).closed += 1


class FakeRequest:
    headers = {}

    def __init__(self, body):
        self.body = body

    async def json(self):
        return self.body


class RelayRepairIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def _relay(self, body, incoming_wire, upstream_wire):
        supplier = {
            "name": "pixel",
            "base_url": "https://api.pixel.test",
            "api_key": "test-key",
            "wire_api": upstream_wire,
        }

        async def resolve(_cfg, _model):
            return dict(supplier), "deepseek-v4.1-flash"

        CapturingClient.sent = []
        CapturingClient.closed = 0
        records = []
        with (
            patch.object(router, "load_config", return_value={}),
            patch.object(router, "resolve_model_target", new=resolve),
            patch.object(router.httpx, "AsyncClient", CapturingClient),
            patch.object(router.stats, "record", side_effect=records.append),
        ):
            response = await router.relay(FakeRequest(body), incoming_wire)
            b"".join([chunk async for chunk in response.body_iterator])
        return records

    async def test_chat_history_with_dangling_call_is_repaired_before_send(self):
        body = {
            "model": "pixel/deepseek-v4.1-flash",
            "stream": True,
            "messages": [
                {"role": "user", "content": "跑一下"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_orphan",
                        "type": "function",
                        "function": {"name": "pwsh", "arguments": "{}"},
                    }],
                },
                {"role": "user", "content": "继续"},
            ],
        }
        records = await self._relay(body, "chat", "chat")

        sent = CapturingClient.sent[0]
        roles = [m["role"] for m in sent["messages"]]
        self.assertEqual(roles, ["user", "assistant", "tool", "user"])
        self.assertEqual(sent["messages"][2]["tool_call_id"], "call_orphan")
        self.assertEqual(records[0]["repaired_tool_calls"], 1)

    async def test_responses_history_with_dangling_call_is_repaired(self):
        body = {
            "model": "pixel/deepseek-v4.1-flash",
            "stream": True,
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
                {"type": "function_call", "call_id": "call_orphan", "name": "pwsh", "arguments": "{}"},
            ],
        }
        records = await self._relay(body, "responses", "responses")

        sent = CapturingClient.sent[0]
        types = [item.get("type") for item in sent["input"]]
        self.assertIn("function_call_output", types)
        self.assertEqual(records[0]["repaired_tool_calls"], 1)

    async def test_clean_history_reports_zero_repairs(self):
        body = {
            "model": "pixel/deepseek-v4.1-flash",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        }
        records = await self._relay(body, "chat", "chat")

        self.assertEqual(records[0]["repaired_tool_calls"], 0)
        self.assertEqual(
            [m["role"] for m in CapturingClient.sent[0]["messages"]], ["user"],
        )

    async def test_responses_to_chat_conversion_is_repaired_on_chat_side(self):
        """responses 入口 + chat 上游：补齐要发生在转换后的 chat 历史上。"""
        body = {
            "model": "pixel/deepseek-v4.1-flash",
            "stream": True,
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
                {"type": "function_call", "call_id": "call_orphan", "name": "pwsh", "arguments": "{}"},
            ],
        }
        records = await self._relay(body, "responses", "chat")

        sent = CapturingClient.sent[0]
        roles = [m["role"] for m in sent["messages"]]
        self.assertIn("tool", roles)
        self.assertEqual(records[0]["repaired_tool_calls"], 1)


if __name__ == "__main__":
    unittest.main()
