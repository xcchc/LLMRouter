# -*- coding: utf-8 -*-
import json
import unittest
from unittest.mock import patch

import httpx

import router


class SequenceJsonClient:
    requests = []
    responses = []
    closed = False

    def __init__(self, *args, **kwargs):
        type(self).closed = False

    def build_request(self, method, url, **kwargs):
        request = httpx.Request(method, url, **kwargs)
        type(self).requests.append(request)
        return request

    async def send(self, request, stream=False):
        status, content, content_type = type(self).responses.pop(0)
        return httpx.Response(
            status,
            request=request,
            headers={"content-type": content_type},
            content=content,
        )

    async def aclose(self):
        type(self).closed = True


class FakeRequest:
    headers = {}

    def __init__(self, body):
        self.body = body

    async def json(self):
        return self.body


class MiniMaxSanitizerTests(unittest.TestCase):
    def test_converts_codex_tool_items_to_standard_function_items(self):
        original = {
            "model": "MiniMax-M3",
            "input": [
                {"type": "tool_search_call", "call_id": "call_search", "arguments": {"query": "files", "limit": 10}},
                {
                    "type": "tool_search_output",
                    "call_id": "call_search",
                    "tools": [{
                        "type": "function",
                        "name": "read_file",
                        "description": "Read a file.",
                        "parameters": {"type": "object", "properties": {}},
                    }],
                },
                {
                    "type": "custom_tool_call",
                    "call_id": "call_patch",
                    "name": "apply_patch",
                    "input": "apply patch",
                },
                {"type": "custom_tool_call_output", "call_id": "call_patch", "output": "ok"},
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                },
            ],
            "tools": [
                {
                    "type": "tool_search",
                    "name": "tool_search",
                    "parameters": {"type": "object", "properties": {}},
                },
                {"type": "custom", "name": "apply_patch", "description": "Edit files."},
            ],
            "tool_choice": {"type": "custom", "name": "apply_patch"},
        }

        sanitized = router._sanitize_responses_for_minimax(original)

        self.assertEqual(
            [item["type"] for item in sanitized["input"]],
            ["function_call", "function_call_output", "function_call", "function_call_output", "message"],
        )
        tool_search_call = sanitized["input"][0]
        self.assertEqual(tool_search_call["name"], "tool_search")
        self.assertIsInstance(tool_search_call["arguments"], str)
        self.assertIn("files", json.loads(tool_search_call["arguments"])["query"])
        self.assertIn("Loaded tools: read_file", sanitized["input"][1]["output"])
        custom_call = sanitized["input"][2]
        self.assertEqual(custom_call["name"], "apply_patch")
        self.assertEqual(json.loads(custom_call["arguments"]), {"input": "apply patch"})
        self.assertTrue(all(tool["type"] == "function" for tool in sanitized["tools"]))
        self.assertEqual(sanitized["tool_choice"], {"type": "function", "name": "apply_patch"})
        self.assertEqual(original["input"][0]["type"], "tool_search_call")
        self.assertIsInstance(original["input"][0]["arguments"], dict)
        self.assertEqual(original["tool_choice"], {"type": "custom", "name": "apply_patch"})

    def test_agent_message_becomes_user_message_and_removes_private_fields(self):
        original = {
            "input": [{
                "type": "agent_message",
                "author": "subagent",
                "recipient": "/root",
                "content": [
                    {"type": "input_text", "text": "Task:"},
                    {"type": "encrypted_content", "encrypted_content": "PAYLOAD-1234"},
                ],
            }],
        }

        sanitized = router._sanitize_responses_for_minimax(original)
        item = sanitized["input"][0]

        self.assertEqual(item["type"], "message")
        self.assertEqual(item["role"], "user")
        self.assertNotIn("author", item)
        self.assertNotIn("recipient", item)
        self.assertIn("PAYLOAD-1234", json.dumps(sanitized, ensure_ascii=False))
        self.assertNotIn("encrypted_content", json.dumps(sanitized, ensure_ascii=False))

    def test_namespace_tools_flatten_to_standard_function_tools(self):
        original = {
            "tools": [{
                "type": "namespace",
                "name": "fs",
                "tools": [{
                    "type": "function",
                    "name": "read_file",
                    "description": "Read a file.",
                    "parameters": {"type": "object", "properties": {}},
                }],
            }],
        }

        sanitized = router._sanitize_responses_for_minimax(original)

        self.assertEqual(sanitized["tools"], [{
            "type": "function",
            "name": "fs__read_file",
            "description": "Read a file.",
            "parameters": {"type": "object", "properties": {}},
        }])

    def test_orphan_and_unknown_items_become_user_text(self):
        original = {
            "input": [
                {
                    "type": "tool_search_output",
                    "tools": [{"type": "function", "name": "read_file"}],
                },
                {"type": "web_search_call", "input": {"query": "search"}},
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "think"}]},
            ],
        }

        sanitized = router._sanitize_responses_for_minimax(original)

        self.assertEqual(sanitized["input"][0]["type"], "message")
        self.assertIn("Loaded tools: read_file", sanitized["input"][0]["content"][0]["text"])
        self.assertEqual(sanitized["input"][1]["type"], "message")
        self.assertIn("query", json.dumps(sanitized["input"][1], ensure_ascii=False))
        self.assertEqual(sanitized["input"][2]["type"], "reasoning")

    def test_validate_supplier_accepts_only_empty_or_minimax(self):
        base = {"name": "pengm3", "base_url": "https://api.minimaxi.com/v1"}
        self.assertIsNone(router.validate_supplier({**base, "responses_compat": "minimax"}))
        self.assertIsNone(router.validate_supplier({**base, "responses_compat": ""}))
        self.assertIsNotNone(router.validate_supplier({**base, "responses_compat": "openai"}))
        self.assertIsNotNone(router.validate_supplier({**base, "responses_compat": 1}))

    def test_sensitive_image_error_detection(self):
        self.assertTrue(router._minimax_sensitive_image_error(
            "input new_sensitive, messages[208]'s content[1] image is sensitive, please check your input (1026)"
        ))
        self.assertFalse(router._minimax_sensitive_image_error("unsupported field"))

    def test_sensitive_image_candidate_replaces_nested_image_only(self):
        original = {
            "input": [{
                "type": "function_call_output",
                "call_id": "call_view",
                "output": [
                    {"type": "input_text", "text": "Screenshot:"},
                    {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
                ],
            }],
        }

        sanitized = router._strip_sensitive_image_candidate(original)

        self.assertIn("input_image", json.dumps(original, ensure_ascii=False))
        self.assertNotIn("input_image", json.dumps(sanitized, ensure_ascii=False))
        self.assertIn(router.MINIMAX_SENSITIVE_IMAGE_PLACEHOLDER, json.dumps(sanitized, ensure_ascii=False))
        self.assertIn("Screenshot:", json.dumps(sanitized, ensure_ascii=False))


class MiniMaxRelayTests(unittest.IsolatedAsyncioTestCase):
    async def _relay(self, body, supplier, responses):
        SequenceJsonClient.requests = []
        SequenceJsonClient.responses = list(responses)
        records = []

        async def find_supplier(_cfg, _model):
            return supplier

        with (
            patch.object(router, "load_config", return_value={}),
            patch.object(router, "find_supplier", new=find_supplier),
            patch.object(router.httpx, "AsyncClient", SequenceJsonClient),
            patch.object(router.stats, "record", side_effect=records.append),
        ):
            response = await router.relay(FakeRequest(body), "responses")
        return response, records

    async def test_minimax_retries_sensitive_image_once_with_placeholder(self):
        success = (
            b'{"id":"resp_ok","object":"response","status":"completed",'
            b'"usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}}'
        )
        sensitive_error = (
            b'{"error":{"message":"input new_sensitive, messages[208]\'s content[1] image '
            b'is sensitive, please check your input (1026)"}}'
        )
        body = {
            "model": "MiniMax-M3",
            "stream": False,
            "input": [{
                "type": "message",
                "role": "user",
                "content": [{"type": "input_image", "image_url": "data:image/png;base64,AAAA"}],
            }],
        }
        supplier = {
            "name": "pengm3",
            "base_url": "https://api.minimaxi.com/v1",
            "api_key": "test-key",
            "wire_api": "responses",
            "openai_sanitize": False,
            "responses_compat": "minimax",
        }

        response, records = await self._relay(
            body,
            supplier,
            [(422, sensitive_error, "application/json"), (200, success, "application/json")],
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(SequenceJsonClient.requests), 2)
        first = json.loads(SequenceJsonClient.requests[0].content)
        second = json.loads(SequenceJsonClient.requests[1].content)
        self.assertIn("input_image", json.dumps(first, ensure_ascii=False))
        self.assertNotIn("input_image", json.dumps(second, ensure_ascii=False))
        self.assertIn(router.MINIMAX_SENSITIVE_IMAGE_PLACEHOLDER, json.dumps(second, ensure_ascii=False))
        self.assertEqual(records[0]["retry_count"], 1)
        self.assertTrue(SequenceJsonClient.closed)

    async def test_non_minimax_responses_supplier_preserves_tool_search_body(self):
        success = (
            b'{"id":"resp_ok","object":"response","status":"completed",'
            b'"usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}}'
        )
        body = {
            "model": "same-model",
            "stream": False,
            "input": [{
                "type": "tool_search_call",
                "call_id": "call_search",
                "arguments": {"query": "files"},
            }],
        }
        supplier = {
            "name": "responses-provider",
            "base_url": "https://provider.example/v1",
            "api_key": "test-key",
            "wire_api": "responses",
            "openai_sanitize": False,
        }

        response, _records = await self._relay(
            body,
            supplier,
            [(200, success, "application/json")],
        )

        sent = json.loads(SequenceJsonClient.requests[0].content)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(sent["input"][0]["type"], "tool_search_call")
        self.assertIsInstance(sent["input"][0]["arguments"], dict)


if __name__ == "__main__":
    unittest.main()
