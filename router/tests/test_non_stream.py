import json
import unittest
from unittest.mock import patch

import httpx

import router


class FakeRequest:
    headers = {}

    def __init__(self, body):
        self.body = body

    async def json(self):
        return self.body


class StaticJsonClient:
    raw_body = b"{}"
    content_type = "application/json"
    requests = []
    closed = False

    def __init__(self, *args, **kwargs):
        type(self).closed = False

    def build_request(self, method, url, **kwargs):
        request = httpx.Request(method, url, **kwargs)
        type(self).requests.append(request)
        return request

    async def send(self, request, stream=False):
        return httpx.Response(
            200,
            request=request,
            headers={"content-type": type(self).content_type},
            content=type(self).raw_body,
        )

    async def aclose(self):
        type(self).closed = True


class NonStreamRelayTests(unittest.IsolatedAsyncioTestCase):
    async def _relay(self, incoming_wire, upstream_wire, request_body, raw_body, content_type="application/json"):
        StaticJsonClient.raw_body = raw_body
        StaticJsonClient.content_type = content_type
        StaticJsonClient.requests = []
        records = []

        async def find_supplier(_cfg, _model):
            return {
                "name": "provider",
                "base_url": "https://provider.example/v1",
                "api_key": "test-key",
                "wire_api": upstream_wire,
            }

        with (
            patch.object(router, "load_config", return_value={}),
            patch.object(router, "find_supplier", new=find_supplier),
            patch.object(router.httpx, "AsyncClient", StaticJsonClient),
            patch.object(router.stats, "record", side_effect=records.append),
        ):
            response = await router.relay(FakeRequest(request_body), incoming_wire)
        return response, records

    async def test_same_wire_preserves_json_bytes_and_content_type(self):
        raw_body = (
            b'{\n  "id": "resp_same", "object": "response", "status": "completed",\n'
            b'  "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5}\n}\n'
        )
        content_type = "application/vnd.openai+json; charset=utf-8"

        response, records = await self._relay(
            "responses",
            "responses",
            {"model": "same-model", "input": "hello", "stream": False},
            raw_body,
            content_type,
        )

        self.assertNotIsInstance(response, router.StreamingResponse)
        self.assertEqual(response.body, raw_body)
        self.assertEqual(response.headers["content-type"], content_type)
        self.assertTrue(StaticJsonClient.closed)
        self.assertFalse(json.loads(StaticJsonClient.requests[0].content)["stream"])
        self.assertEqual(records[0]["total_tokens"], 5)

    async def test_same_wire_responses_sanitizes_agent_message_for_third_party(self):
        raw_body = b'{"object":"response","status":"completed"}'
        request_body = {
            "model": "same-model",
            "stream": False,
            "input": [{
                "type": "agent_message",
                "author": "subagent",
                "content": [
                    {"type": "input_text", "text": "Message Type: NEW_TASK ... Payload:\n"},
                    {"type": "encrypted_content", "encrypted_content": "opaque-ciphertext"},
                ],
            }],
        }

        response, _records = await self._relay(
            "responses",
            "responses",
            request_body,
            raw_body,
        )

        sent = json.loads(StaticJsonClient.requests[0].content)
        self.assertNotIn("encrypted_content", json.dumps(sent))
        self.assertNotIn("opaque-ciphertext", json.dumps(sent))
        item = sent["input"][0]
        self.assertEqual(item["type"], "message")
        self.assertEqual(item["role"], "user")
        self.assertIn(router._OPENAI_ENCRYPTED_PLACEHOLDER, item["content"][0]["text"])
        self.assertEqual(response.body, raw_body)

    async def test_chat_json_is_converted_to_non_stream_responses_json(self):
        upstream = {
            "id": "chatcmpl_upstream",
            "object": "chat.completion",
            "created": 123,
            "model": "deepseek-upstream",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "reasoning_content": "Checked the repository.",
                    "content": "Done.",
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 4,
                "total_tokens": 11,
            },
        }

        response, records = await self._relay(
            "responses",
            "chat",
            {
                "model": "deepseek",
                "input": "inspect",
                "reasoning": {"effort": "high"},
                "stream": False,
            },
            json.dumps(upstream).encode("utf-8"),
        )
        result = json.loads(response.body)

        self.assertNotIsInstance(response, router.StreamingResponse)
        self.assertEqual(response.headers["content-type"], "application/json")
        self.assertEqual(result["object"], "response")
        self.assertEqual(result["status"], "completed")
        self.assertEqual([item["type"] for item in result["output"]], ["reasoning", "message"])
        self.assertEqual(result["output"][1]["content"][0]["text"], "Done.")
        self.assertEqual(result["usage"]["total_tokens"], 11)
        sent = json.loads(StaticJsonClient.requests[0].content)
        self.assertFalse(sent["stream"])
        self.assertEqual(records[0]["input_tokens"], 7)
        self.assertEqual(records[0]["reasoning_effort"], "high")

    async def test_responses_json_is_converted_to_non_stream_chat_json(self):
        upstream = {
            "id": "resp_upstream",
            "object": "response",
            "created_at": 456,
            "status": "completed",
            "model": "gpt-upstream",
            "output": [
                {
                    "id": "rs_1",
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "Use the tool."}],
                },
                {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Checking."}],
                },
                {
                    "id": "fc_1",
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "read_file",
                    "arguments": '{"path":"README.md"}',
                },
            ],
            "usage": {
                "input_tokens": 5,
                "input_tokens_details": {"cached_tokens": 2},
                "output_tokens": 6,
                "output_tokens_details": {"reasoning_tokens": 3},
                "total_tokens": 11,
            },
        }

        response, records = await self._relay(
            "chat",
            "responses",
            {
                "model": "gpt-compatible",
                "messages": [{"role": "user", "content": "inspect"}],
                "reasoning_effort": "xhigh",
                "stream": False,
            },
            json.dumps(upstream).encode("utf-8"),
        )
        result = json.loads(response.body)
        choice = result["choices"][0]
        message = choice["message"]

        self.assertNotIsInstance(response, router.StreamingResponse)
        self.assertEqual(result["object"], "chat.completion")
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual(message["content"], "Checking.")
        self.assertEqual(message["reasoning_content"], "Use the tool.")
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "read_file")
        self.assertEqual(message["tool_calls"][0]["function"]["arguments"], '{"path":"README.md"}')
        self.assertEqual(result["usage"]["prompt_tokens_details"]["cached_tokens"], 2)
        sent = json.loads(StaticJsonClient.requests[0].content)
        self.assertFalse(sent["stream"])
        self.assertEqual(records[0]["reasoning_tokens"], 3)
        self.assertEqual(records[0]["reasoning_effort"], "xhigh")

    async def test_cross_wire_invalid_json_returns_json_error_not_sse(self):
        response, records = await self._relay(
            "responses",
            "chat",
            {"model": "deepseek", "input": "hello", "stream": False},
            b"not-json",
            "text/plain",
        )
        result = json.loads(response.body)

        self.assertEqual(response.status_code, 502)
        self.assertNotIsInstance(response, router.StreamingResponse)
        self.assertEqual(response.headers["content-type"], "application/json")
        self.assertIn("invalid JSON", result["error"]["message"])
        self.assertEqual(records[0]["status"], "error")
        self.assertTrue(StaticJsonClient.closed)


if __name__ == "__main__":
    unittest.main()
