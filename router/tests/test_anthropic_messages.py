import asyncio
import json
import unittest
from unittest.mock import patch

import httpx

import router
from suppliers.anthropic_messages import (
    ChatToAnthropic,
    anthropic_error_json,
    anthropic_to_chat,
    chat_json_to_anthropic,
    estimate_input_tokens,
)


class FakeRequest:
    headers = {}

    def __init__(self, body):
        self.body = body

    async def json(self):
        return self.body


class HeaderRequest:
    def __init__(self, headers):
        self.headers = httpx.Headers(headers)


class StaticStream(httpx.AsyncByteStream):
    def __init__(self, *chunks):
        self.chunks = chunks

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk


class StaticJsonClient:
    raw_body = b"{}"
    content_type = "application/json"
    requests = []

    def __init__(self, *args, **kwargs):
        pass

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
        pass


class StaticSseClient:
    raw = b""
    requests = []

    def __init__(self, *args, **kwargs):
        pass

    def build_request(self, method, url, **kwargs):
        request = httpx.Request(method, url, **kwargs)
        type(self).requests.append(request)
        return request

    async def send(self, request, stream=False):
        return httpx.Response(
            200,
            request=request,
            headers={"content-type": "text/event-stream"},
            stream=StaticStream(type(self).raw),
        )

    async def aclose(self):
        pass


def sse_events(raw: bytes) -> list:
    events = []
    for line in raw.decode("utf-8").splitlines():
        line = line.strip()
        if line.startswith("data:") and line[5:].strip():
            events.append(json.loads(line[5:].strip()))
    return events


class AnthropicToChatTests(unittest.TestCase):
    def test_system_and_text_messages(self):
        out = anthropic_to_chat({
            "model": "glm",
            "max_tokens": 128,
            "system": "be brief",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
                {"role": "user", "content": [{"type": "text", "text": "ok"}]},
            ],
        })
        self.assertEqual(out["model"], "glm")
        self.assertFalse(out["stream"])
        self.assertEqual(out["max_tokens"], 128)
        self.assertEqual(
            out["messages"],
            [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": "ok"},
            ],
        )

    def test_tool_roundtrip(self):
        out = anthropic_to_chat({
            "model": "glm",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "checking"},
                        {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "SF"}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "sunny"},
                        {"type": "text", "text": "thanks"},
                    ],
                },
            ],
        })
        assistant = out["messages"][0]
        self.assertEqual(assistant["role"], "assistant")
        self.assertEqual(assistant["content"], "checking")
        self.assertEqual(assistant["tool_calls"][0]["id"], "toolu_1")
        self.assertEqual(assistant["tool_calls"][0]["function"]["name"], "get_weather")
        self.assertEqual(json.loads(assistant["tool_calls"][0]["function"]["arguments"]), {"city": "SF"})
        tool = out["messages"][1]
        self.assertEqual(tool, {"role": "tool", "tool_call_id": "toolu_1", "content": "sunny"})
        self.assertEqual(out["messages"][2], {"role": "user", "content": "thanks"})

    def test_tools_and_tool_choice(self):
        out = anthropic_to_chat({
            "model": "glm",
            "tools": [
                {"name": "f1", "description": "d1", "input_schema": {"type": "object", "properties": {}}},
                {"type": "web_search_20250305", "name": "web"},  # 托管工具应被丢弃
            ],
            "tool_choice": {"type": "tool", "name": "f1", "disable_parallel_tool_use": True},
        })
        self.assertEqual(len(out["tools"]), 1)
        self.assertEqual(out["tools"][0]["type"], "function")
        self.assertEqual(out["tools"][0]["function"]["name"], "f1")
        self.assertEqual(out["tool_choice"], {"type": "function", "function": {"name": "f1"}})
        self.assertFalse(out["parallel_tool_calls"])

    def test_tool_choice_variants(self):
        self.assertEqual(anthropic_to_chat({"tool_choice": {"type": "auto"}})["tool_choice"], "auto")
        self.assertEqual(anthropic_to_chat({"tool_choice": {"type": "any"}})["tool_choice"], "required")

    def test_thinking_maps_to_reasoning_effort(self):
        out = anthropic_to_chat({
            "model": "glm",
            "thinking": {"type": "enabled", "budget_tokens": 32000},
            "messages": [{"role": "user", "content": "hi"}],
        })
        self.assertEqual(out["reasoning_effort"], "max")
        out = anthropic_to_chat({
            "model": "glm",
            "thinking": {"type": "enabled", "budget_tokens": 2048},
            "messages": [{"role": "user", "content": "hi"}],
        })
        self.assertEqual(out["reasoning_effort"], "high")

    def test_image_base64_becomes_data_url(self):
        out = anthropic_to_chat({
            "model": "glm",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"}},
                ],
            }],
        })
        content = out["messages"][0]["content"]
        self.assertTrue(content[0]["text"] == "look")
        self.assertEqual(content[1]["type"], "image_url")
        self.assertEqual(content[1]["image_url"]["url"], "data:image/png;base64,QUJD")

    def test_stop_sequences_and_stream(self):
        out = anthropic_to_chat({
            "model": "glm",
            "stream": True,
            "stop_sequences": ["END"],
            "messages": [{"role": "user", "content": "hi"}],
        })
        self.assertTrue(out["stream"])
        self.assertEqual(out["stop"], ["END"])


class ChatToAnthropicJsonTests(unittest.TestCase):
    def test_text_and_usage(self):
        out = chat_json_to_anthropic({
            "id": "chatcmpl-1",
            "model": "glm",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        }, model="glm")
        self.assertEqual(out["type"], "message")
        self.assertEqual(out["role"], "assistant")
        self.assertEqual(out["stop_reason"], "end_turn")
        self.assertEqual(out["content"], [{"type": "text", "text": "hello"}])
        self.assertEqual(out["usage"]["input_tokens"], 4)
        self.assertEqual(out["usage"]["output_tokens"], 2)

    def test_reasoning_then_tool_use(self):
        out = chat_json_to_anthropic({
            "model": "glm",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "reasoning_content": "thinking...",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{\"a\":1}"},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
        })
        self.assertEqual(out["content"][0]["type"], "thinking")
        self.assertEqual(out["content"][1]["type"], "tool_use")
        self.assertEqual(out["content"][1]["input"], {"a": 1})
        self.assertEqual(out["stop_reason"], "tool_use")


class ChatToAnthropicStreamTests(unittest.TestCase):
    def _collect(self, chunks):
        conv = ChatToAnthropic("glm")
        events = []
        for chunk in chunks:
            events.extend(conv.feed(chunk))
        events.extend(conv.final_events())
        return events, conv

    def test_text_flow_indices_and_terminator(self):
        events, conv = self._collect([
            {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "he"}, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {"content": "llo"}, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 3, "completion_tokens": 4}},
        ])
        types = [e["type"] for e in events]
        self.assertEqual(types[0], "message_start")
        self.assertEqual(types[1], "content_block_start")
        self.assertEqual(types[1] and events[1]["content_block"]["type"], "text")
        self.assertEqual([e["type"] for e in events[2:4]], ["content_block_delta", "content_block_delta"])
        self.assertEqual(events[2]["delta"], {"type": "text_delta", "text": "he"})
        self.assertEqual(types[4], "content_block_stop")
        self.assertEqual(types[-2], "message_delta")
        self.assertEqual(events[-2]["delta"]["stop_reason"], "end_turn")
        self.assertEqual(events[-2]["usage"], {"output_tokens": 4})
        self.assertEqual(types[-1], "message_stop")
        self.assertEqual(conv.usage["prompt_tokens"], 3)
        self.assertEqual(conv.status, "completed")

    def test_tool_call_stream(self):
        events, _ = self._collect([
            {"choices": [{"index": 0, "delta": {
                "tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "f", "arguments": ""}}],
            }, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {
                "tool_calls": [{"index": 0, "function": {"arguments": "{\"a\":1}"}}],
            }, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        ])
        starts = [e for e in events if e["type"] == "content_block_start"]
        self.assertEqual(len(starts), 1)
        self.assertEqual(starts[0]["content_block"]["type"], "tool_use")
        self.assertEqual(starts[0]["content_block"]["id"], "call_1")
        self.assertEqual(starts[0]["content_block"]["name"], "f")
        deltas = [e for e in events if e["type"] == "content_block_delta"]
        self.assertEqual(deltas[0]["delta"]["type"], "input_json_delta")
        self.assertEqual(deltas[0]["delta"]["partial_json"], "{\"a\":1}")
        self.assertEqual(events[-2]["delta"]["stop_reason"], "tool_use")

    def test_interleaved_tools_flushed_in_order(self):
        events, _ = self._collect([
            {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "a", "arguments": "1"}}]}}]},
            {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 1, "id": "c2", "function": {"name": "b", "arguments": "2"}}]}}]},
            {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "3"}}]}}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        ])
        # Anthropic 事件流不允许交错回写已关闭的块：工具按首次出现顺序，
        # 在 finish 时一次性输出，每个工具恰好一个完整块。
        blocks = [e["content_block"] for e in events if e["type"] == "content_block_start"]
        self.assertEqual([b["id"] for b in blocks], ["c1", "c2"])
        json_deltas = [e for e in events if e["type"] == "content_block_delta" and e["delta"]["type"] == "input_json_delta"]
        self.assertEqual([(d["index"], d["delta"]["partial_json"]) for d in json_deltas], [(0, "13"), (1, "2")])
        stops = [e["index"] for e in events if e["type"] == "content_block_stop"]
        self.assertEqual(stops, [0, 1])
        self.assertEqual(events[-2]["delta"]["stop_reason"], "tool_use")

    def test_error_chunk_marks_failure(self):
        events, conv = self._collect([
            {"error": {"type": "server_error", "message": "boom"}},
        ])
        self.assertEqual(events[0]["type"], "error")
        self.assertEqual(events[0]["error"]["message"], "boom")
        self.assertEqual(conv.status, "failed")


class DetectAgentTests(unittest.TestCase):
    def test_wire_based_detection(self):
        self.assertEqual(router.detect_agent(HeaderRequest({}), "anthropic"), "claude-code")
        self.assertEqual(router.detect_agent(HeaderRequest({}), "responses"), "codex")

    def test_ua_based_detection_for_chat(self):
        self.assertEqual(router.detect_agent(HeaderRequest({"user-agent": "codex_cli_rs/1.0"}), "chat"), "codex")
        self.assertEqual(router.detect_agent(HeaderRequest({"user-agent": "claude-cli/2.1"}), "chat"), "claude-code")
        self.assertEqual(router.detect_agent(HeaderRequest({"user-agent": "node; undici"}), "chat"), "dsh")
        self.assertEqual(router.detect_agent(HeaderRequest({}), "chat"), "")


class AgentProfileTests(unittest.TestCase):
    def test_default_model_only_fills_missing_model(self):
        cfg = {"agent_profiles": {"claude-code": {"default_model": "glm"}}}
        body = {"model": "claude-sonnet-4-5"}
        router._apply_agent_profile(cfg, "claude-code", body)
        self.assertEqual(body["model"], "claude-sonnet-4-5")

        body = {"max_tokens": 8}
        router._apply_agent_profile(cfg, "claude-code", body)
        self.assertEqual(body["model"], "glm")

        legacy = {"agent_profiles": {"claude-code": {"model_map": {"*": "glm"}}}}
        body = {}
        router._apply_agent_profile(legacy, "claude-code", body)
        self.assertEqual(body["model"], "glm")

    def test_unknown_agent_is_noop(self):
        body = {"model": "m"}
        router._apply_agent_profile({}, "codex", body)
        self.assertEqual(body["model"], "m")

    def test_agent_reasoning_effort_validation(self):
        cfg = {"agent_profiles": {"dsh": {"reasoning_effort": "max"}}}
        self.assertEqual(router._agent_reasoning_effort(cfg, "dsh"), "max")
        cfg_bad = {"agent_profiles": {"dsh": {"reasoning_effort": "bogus"}}}
        self.assertIsNone(router._agent_reasoning_effort(cfg_bad, "dsh"))


class RelayAnthropicTests(unittest.IsolatedAsyncioTestCase):
    async def _relay(self, incoming_wire, upstream_wire, request_body, client_cls, cfg=None):
        records = []

        async def find_supplier(_cfg, _model):
            return {
                "name": "provider",
                "base_url": "https://provider.example/v1",
                "api_key": "test-key",
                "wire_api": upstream_wire,
            }

        with (
            patch.object(router, "load_config", return_value=cfg or {}),
            patch.object(router, "find_supplier", new=find_supplier),
            patch.object(router.httpx, "AsyncClient", client_cls),
            patch.object(router.stats, "record", side_effect=records.append),
        ):
            response = await router.relay(FakeRequest(request_body), incoming_wire)
            if hasattr(response, "body_iterator"):
                # 流式统计在生成器 finally 里落账，必须在 patch 作用域内消费完。
                raw = b""
                async for chunk in response.body_iterator:
                    raw += chunk if isinstance(chunk, bytes) else str(chunk).encode()
                response._raw_stream = raw
        return response, records

    async def test_non_stream_via_chat_upstream(self):
        StaticJsonClient.raw_body = json.dumps({
            "id": "chatcmpl-1",
            "model": "glm",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        }).encode()
        StaticJsonClient.requests = []
        response, records = await self._relay(
            "anthropic",
            "chat",
            {"model": "glm", "max_tokens": 16, "messages": [{"role": "user", "content": "ping"}]},
            StaticJsonClient,
        )
        self.assertEqual(response.status_code, 200)
        body = json.loads(response.body)
        self.assertEqual(body["type"], "message")
        self.assertEqual(body["stop_reason"], "end_turn")
        self.assertEqual(body["content"][0], {"type": "text", "text": "hi"})
        # 转发出去的是 chat 请求
        sent = json.loads(StaticJsonClient.requests[0].content)
        self.assertEqual(sent["model"], "glm")
        self.assertFalse(sent["stream"])
        # 统计来自转换前的 chat usage
        self.assertEqual(records[0]["input_tokens"], 2)
        self.assertEqual(records[0]["output_tokens"], 1)
        self.assertEqual(records[0]["agent"], "claude-code")
        self.assertEqual(records[0]["incoming_wire"], "anthropic")

    async def test_non_stream_via_responses_upstream(self):
        StaticJsonClient.raw_body = json.dumps({
            "id": "resp_1",
            "object": "response",
            "status": "completed",
            "model": "glm",
            "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hey"}]}],
            "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
        }).encode()
        StaticJsonClient.requests = []
        response, _ = await self._relay(
            "anthropic",
            "responses",
            {"model": "glm", "max_tokens": 16, "messages": [{"role": "user", "content": "ping"}]},
            StaticJsonClient,
        )
        self.assertEqual(response.status_code, 200)
        body = json.loads(response.body)
        self.assertEqual(body["type"], "message")
        texts = [b.get("text") for b in body["content"] if b.get("type") == "text"]
        self.assertEqual("".join(texts), "hey")
        self.assertEqual(body["usage"]["output_tokens"], 3)

    async def test_stream_via_chat_upstream(self):
        StaticSseClient.raw = (
            b'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"hi"},'
            b'"finish_reason":null}]}\n\n'
            b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
            b'"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}\n\n'
            b'data: [DONE]\n\n'
        )
        StaticSseClient.requests = []
        response, records = await self._relay(
            "anthropic",
            "chat",
            {"model": "glm", "stream": True, "max_tokens": 16, "messages": [{"role": "user", "content": "ping"}]},
            StaticSseClient,
        )
        self.assertEqual(response.status_code, 200)
        raw = response._raw_stream
        events = sse_events(raw)
        self.assertEqual(events[0]["type"], "message_start")
        self.assertEqual(events[-1]["type"], "message_stop")
        self.assertNotIn("data: [DONE]", raw.decode())
        types = [e["type"] for e in events]
        self.assertIn("content_block_delta", types)
        self.assertEqual(events[-2]["delta"]["stop_reason"], "end_turn")
        self.assertEqual(records[0]["input_tokens"], 2)
        self.assertEqual(records[0]["output_tokens"], 1)

    async def test_agent_default_model_used_only_when_missing(self):
        StaticJsonClient.raw_body = b"{}"
        StaticJsonClient.requests = []
        cfg = {"agent_profiles": {"claude-code": {"default_model": "glm"}}}
        await self._relay(
            "anthropic",
            "chat",
            {"max_tokens": 8, "messages": [{"role": "user", "content": "ping"}]},
            StaticJsonClient,
            cfg=cfg,
        )
        sent = json.loads(StaticJsonClient.requests[0].content)
        self.assertEqual(sent["model"], "glm")

        StaticJsonClient.requests = []
        await self._relay(
            "anthropic",
            "chat",
            {"model": "claude-sonnet-4-5", "max_tokens": 8, "messages": [{"role": "user", "content": "ping"}]},
            StaticJsonClient,
            cfg=cfg,
        )
        sent = json.loads(StaticJsonClient.requests[0].content)
        self.assertEqual(sent["model"], "claude-sonnet-4-5")

    async def test_upstream_error_wraps_anthropic_shape(self):
        class ErrorClient(StaticJsonClient):
            async def send(self, request, stream=False):
                return httpx.Response(400, request=request, json={"error": {"message": "bad request"}})

        response, _ = await self._relay(
            "anthropic",
            "chat",
            {"model": "glm", "max_tokens": 8, "messages": [{"role": "user", "content": "ping"}]},
            ErrorClient,
        )
        self.assertEqual(response.status_code, 400)
        body = json.loads(response.body)
        self.assertEqual(body["type"], "error")
        self.assertEqual(body["error"]["message"], "bad request")


class MiscAnthropicTests(unittest.TestCase):
    def test_upstream_stream_failure_event(self):
        raw = router.upstream_stream_failure_event("anthropic", "glm", "boom")
        evt = json.loads(raw.decode().removeprefix("data: ").strip())
        self.assertEqual(evt["type"], "error")
        self.assertEqual(evt["error"]["message"], "boom")

    def test_error_json_mapping(self):
        self.assertEqual(anthropic_error_json("x", status=429)["error"]["type"], "rate_limit_error")
        self.assertEqual(anthropic_error_json("x", status=503)["error"]["type"], "overloaded_error")
        self.assertEqual(anthropic_error_json("x")["error"]["type"], "api_error")

    def test_estimate_input_tokens(self):
        body = {"messages": [{"role": "user", "content": "a" * 40}]}
        # "user"(4) + 40 个 a = 44 字符 → 44 // 4 = 11
        self.assertEqual(estimate_input_tokens(body), 11)
        self.assertGreaterEqual(estimate_input_tokens({}), 1)


if __name__ == "__main__":
    unittest.main()
