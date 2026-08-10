import gzip
import json
import unittest
from unittest.mock import patch

import httpx

import router


class GzipStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        payload = b'data: {"type":"response.completed"}\n\ndata: [DONE]\n\n'
        yield gzip.compress(payload)


class StaticStream(httpx.AsyncByteStream):
    def __init__(self, *chunks):
        self.chunks = chunks

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk


class FakeRequest:
    headers = {}

    def __init__(self, body):
        self.body = body

    async def json(self):
        return self.body


class RetryThenSuccessClient:
    send_count = 0

    def __init__(self, *args, **kwargs):
        pass

    def build_request(self, method, url, **kwargs):
        return httpx.Request(method, url, **kwargs)

    async def send(self, request, stream=False):
        type(self).send_count += 1
        if type(self).send_count < 3:
            return httpx.Response(
                500,
                request=request,
                json={"type": "Router.Unavailable"},
            )
        payload = (
            b'data: {"choices":[{"index":0,"delta":{"content":"ok"},'
            b'"finish_reason":"stop"}],"usage":{"prompt_tokens":1,'
            b'"completion_tokens":1,"total_tokens":2}}\n\n'
            b'data: [DONE]\n\n'
        )
        return httpx.Response(
            200,
            request=request,
            headers={"content-type": "text/event-stream"},
            stream=StaticStream(payload),
        )

    async def aclose(self):
        pass


class RetryThenSuccess429Client:
    send_count = 0

    def __init__(self, *args, **kwargs):
        pass

    def build_request(self, method, url, **kwargs):
        return httpx.Request(method, url, **kwargs)

    async def send(self, request, stream=False):
        type(self).send_count += 1
        if type(self).send_count < 3:
            return httpx.Response(
                429,
                request=request,
                json={"error": {"message": "rate limited"}},
            )
        payload = (
            b'data: {"choices":[{"index":0,"delta":{"content":"ok"},'
            b'"finish_reason":"stop"}],"usage":{"prompt_tokens":1,'
            b'"completion_tokens":1,"total_tokens":2}}\n\n'
            b'data: [DONE]\n\n'
        )
        return httpx.Response(
            200,
            request=request,
            headers={"content-type": "text/event-stream"},
            stream=StaticStream(payload),
        )

    async def aclose(self):
        pass


class IncompleteResponseClient:
    def __init__(self, *args, **kwargs):
        pass

    def build_request(self, method, url, **kwargs):
        return httpx.Request(method, url, **kwargs)

    async def send(self, request, stream=False):
        payload = (
            b'data: {"type":"response.incomplete","response":{'
            b'"status":"incomplete","usage":{"input_tokens":2,'
            b'"output_tokens":1,"total_tokens":3}}}\n\n'
            b'data: [DONE]\n\n'
        )
        return httpx.Response(
            200,
            request=request,
            headers={"content-type": "text/event-stream"},
            stream=StaticStream(payload),
        )

    async def aclose(self):
        pass


class RouterStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_relay_retries_two_upstream_500s_before_streaming_success(self):
        RetryThenSuccessClient.send_count = 0
        records = []

        async def find_supplier(_cfg, _model):
            return {
                "name": "chat-provider",
                "base_url": "https://provider.example/v1",
                "api_key": "test-key",
                "wire_api": "chat",
            }

        async def no_sleep(_seconds):
            pass

        request = FakeRequest({
            "model": "deepseek-v4-flash",
            "input": "Say ok.",
            "stream": True,
        })
        with (
            patch.object(router, "load_config", return_value={}),
            patch.object(router, "find_supplier", new=find_supplier),
            patch.object(router.httpx, "AsyncClient", RetryThenSuccessClient),
            patch.object(router.asyncio, "sleep", new=no_sleep),
            patch.object(router.stats, "record", side_effect=records.append),
        ):
            response = await router.relay(request, "responses")
            body = b"".join([chunk async for chunk in response.body_iterator])

        self.assertEqual(RetryThenSuccessClient.send_count, 3)
        self.assertIn(b'"type": "response.completed"', body)
        self.assertNotIn(b'"type": "response.failed"', body)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["retry_count"], 2)
        self.assertEqual(records[0]["http_status"], 200)

    async def test_relay_retries_two_upstream_429s_before_streaming_success(self):
        RetryThenSuccess429Client.send_count = 0
        records = []

        async def find_supplier(_cfg, _model):
            return {
                "name": "chat-provider",
                "base_url": "https://provider.example/v1",
                "api_key": "test-key",
                "wire_api": "chat",
            }

        async def no_sleep(_seconds):
            pass

        request = FakeRequest({
            "model": "responses-model",
            "input": "Say ok.",
            "stream": True,
        })
        with (
            patch.object(router, "load_config", return_value={}),
            patch.object(router, "find_supplier", new=find_supplier),
            patch.object(router.httpx, "AsyncClient", RetryThenSuccess429Client),
            patch.object(router.asyncio, "sleep", new=no_sleep),
            patch.object(router.stats, "record", side_effect=records.append),
        ):
            response = await router.relay(request, "responses")
            body = b"".join([chunk async for chunk in response.body_iterator])

        self.assertEqual(RetryThenSuccess429Client.send_count, 3)
        self.assertIn(b'"type": "response.completed"', body)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["retry_count"], 2)
        self.assertEqual(records[0]["http_status"], 200)

    async def test_relay_records_incomplete_for_same_wire_sse(self):
        records = []

        async def find_supplier(_cfg, _model):
            return {
                "name": "responses-provider",
                "base_url": "https://provider.example/v1",
                "api_key": "test-key",
                "wire_api": "responses",
            }

        request = FakeRequest({
            "model": "responses-model",
            "input": "Say partial.",
            "stream": True,
        })
        with (
            patch.object(router, "load_config", return_value={}),
            patch.object(router, "find_supplier", new=find_supplier),
            patch.object(router.httpx, "AsyncClient", IncompleteResponseClient),
            patch.object(router.stats, "record", side_effect=records.append),
        ):
            response = await router.relay(request, "responses")
            body = b"".join([chunk async for chunk in response.body_iterator])

        self.assertIn(b'"type":"response.incomplete"', body)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["status"], "incomplete")

    async def test_gzip_sse_is_decoded_before_parsing(self):
        response = httpx.Response(
            200,
            headers={
                "content-type": "text/event-stream",
                "content-encoding": "gzip",
            },
            stream=GzipStream(),
        )

        events = []
        async for event in router.iter_upstream_events(response):
            events.append(event)

        self.assertEqual(events[0]["type"], "response.completed")
        self.assertIs(events[1], router.DONE)

    async def test_chat_done_or_eof_without_finish_reason_becomes_failed(self):
        partial = (
            b'data: {"choices":[{"index":0,"delta":{"content":"partial"},'
            b'"finish_reason":null}]}\n\n'
        )
        for ending in (b"data: [DONE]\n\n", b""):
            converter = router.ChatToResponses("deepseek-v4-flash")
            response = httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=StaticStream(partial, ending),
            )

            events = [event async for event in converter.convert(response)]

            self.assertEqual(events[-1]["type"], "response.failed")
            self.assertEqual(
                events[-1]["response"]["error"]["code"],
                "upstream_stream_terminated",
            )
            self.assertNotIn("response.completed", [event["type"] for event in events])


class RouterRequestConversionTests(unittest.TestCase):
    def test_standalone_reasoning_is_attached_to_tool_call_message(self):
        chat = router.responses_to_chat({
            "model": "deepseek-v4-flash",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Create a file."}],
                },
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "I should use a tool."}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "apply_patch",
                    "arguments": {"patch": "demo"},
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "done",
                },
            ],
        })

        self.assertEqual(chat["messages"][1]["role"], "assistant")
        self.assertEqual(chat["messages"][1]["reasoning_content"], "I should use a tool.")
        self.assertEqual(chat["messages"][1]["tool_calls"][0]["id"], "call_1")
        self.assertEqual(chat["messages"][2]["role"], "tool")

    def test_parallel_tool_calls_share_one_assistant_message(self):
        chat = router.responses_to_chat({
            "model": "deepseek-v4-flash",
            "input": [
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "Run both checks."}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Checking now."}],
                },
                {"type": "function_call", "call_id": "call_1", "name": "first", "arguments": {}},
                {"type": "function_call", "call_id": "call_2", "name": "second", "arguments": {}},
                {"type": "function_call_output", "call_id": "call_1", "output": "one"},
                {"type": "function_call_output", "call_id": "call_2", "output": "two"},
            ],
        })

        self.assertEqual(len(chat["messages"]), 3)
        assistant = chat["messages"][0]
        self.assertEqual(assistant["content"], "Checking now.")
        self.assertEqual(assistant["reasoning_content"], "Run both checks.")
        self.assertEqual([call["id"] for call in assistant["tool_calls"]], ["call_1", "call_2"])
        self.assertEqual([message["tool_call_id"] for message in chat["messages"][1:]], ["call_1", "call_2"])

    def test_orphan_reasoning_before_user_message_is_not_sent_as_empty_assistant(self):
        chat = router.responses_to_chat({
            "model": "deepseek-v4-flash",
            "input": [
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "Previous turn."}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "New turn."}],
                },
            ],
        })

        self.assertEqual(chat["messages"], [{"role": "user", "content": "New turn."}])
        self.assertTrue(all(
            message.get("role") != "assistant"
            or message.get("content")
            or message.get("tool_calls")
            for message in chat["messages"]
        ))

    def test_encrypted_reasoning_is_not_replayed_as_plaintext(self):
        chat = router.responses_to_chat({
            "model": "deepseek-v4-flash",
            "input": [
                {
                    "type": "reasoning",
                    "summary": [],
                    "encrypted_content": "opaque-ciphertext",
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "inspect",
                    "arguments": {},
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "done",
                },
            ],
        })

        assistant = chat["messages"][0]
        self.assertNotIn("reasoning_content", assistant)
        self.assertNotIn("opaque-ciphertext", json.dumps(chat, ensure_ascii=False))

    def test_chat_tool_response_round_trips_into_a_valid_follow_up(self):
        converter = router.ChatToResponses("deepseek-v4-flash")
        converter.on_chunk({
            "choices": [{
                "delta": {
                    "reasoning_content": "Inspect both targets.",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "function": {"name": "first", "arguments": "{}"},
                        },
                        {
                            "index": 1,
                            "id": "call_2",
                            "function": {"name": "second", "arguments": "{}"},
                        },
                    ],
                },
                "finish_reason": "tool_calls",
            }],
        })
        converter.final_events()

        chat = router.responses_to_chat({
            "model": "deepseek-v4-flash",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Inspect them."}],
                },
                *converter.output,
                {"type": "function_call_output", "call_id": "call_1", "output": "one"},
                {"type": "function_call_output", "call_id": "call_2", "output": "two"},
            ],
        })

        assistant = chat["messages"][1]
        self.assertEqual(assistant["reasoning_content"], "Inspect both targets.")
        self.assertEqual([call["id"] for call in assistant["tool_calls"]], ["call_1", "call_2"])
        self.assertEqual([message["role"] for message in chat["messages"]], ["user", "assistant", "tool", "tool"])

    def test_custom_tool_definition_and_history_are_bridged_to_chat(self):
        chat = router.responses_to_chat({
            "model": "deepseek-v4-flash",
            "input": [
                {
                    "type": "custom_tool_call",
                    "call_id": "call_patch",
                    "name": "apply_patch",
                    "input": "*** Begin Patch\n*** End Patch",
                },
                {
                    "type": "custom_tool_call_output",
                    "call_id": "call_patch",
                    "output": [{"type": "input_text", "text": "Success"}],
                },
            ],
            "tools": [
                {
                    "type": "custom",
                    "name": "apply_patch",
                    "description": "Apply a patch.",
                    "format": {"type": "text"},
                },
            ],
        })

        tool = chat["tools"][0]["function"]
        self.assertEqual(tool["name"], "apply_patch")
        self.assertEqual(tool["parameters"]["required"], ["input"])
        arguments = json.loads(chat["messages"][0]["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(arguments["input"], "*** Begin Patch\n*** End Patch")
        self.assertEqual(chat["messages"][1]["content"], "Success")

    def test_function_tool_strict_is_forwarded_to_chat_definition(self):
        chat = router.responses_to_chat({
            "model": "deepseek-v4-flash",
            "tools": [{
                "type": "function",
                "name": "read_file",
                "description": "Read a file.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "additionalProperties": False,
                },
            }],
        })

        self.assertIs(chat["tools"][0]["function"]["strict"], True)

    def test_union_function_schema_gets_explicit_object_root(self):
        chat = router.responses_to_chat({
            "model": "deepseek-v4-flash",
            "tools": [{
                "type": "function",
                "name": "codex_app__automation_update",
                "description": "Manage an automation.",
                "parameters": {
                    "anyOf": [
                        {
                            "type": "object",
                            "properties": {"mode": {"const": "view"}},
                            "required": ["mode"],
                        },
                        {
                            "type": "object",
                            "properties": {
                                "mode": {"const": "delete"},
                                "id": {"type": "string"},
                            },
                            "required": ["mode", "id"],
                        },
                    ],
                },
            }],
        })

        parameters = chat["tools"][0]["function"]["parameters"]
        self.assertEqual(parameters["type"], "object")
        self.assertEqual(len(parameters["anyOf"]), 2)
        self.assertEqual(parameters["anyOf"][1]["required"], ["mode", "id"])

    def test_custom_tool_format_is_preserved_in_chat_description(self):
        definition = 'start: "BEGIN" /.+/ "END"'
        chat = router.responses_to_chat({
            "model": "deepseek-v4-flash",
            "tools": [{
                "type": "custom",
                "name": "apply_patch",
                "description": "Apply a patch.",
                "format": {
                    "type": "grammar",
                    "syntax": "lark",
                    "definition": definition,
                },
            }],
        })

        description = chat["tools"][0]["function"]["description"]
        self.assertIn("Chat Completions has no native custom grammar field", description)
        self.assertIn("format.syntax: lark", description)
        self.assertIn(f"format.definition:\n{definition}", description)

    def test_image_input_and_tool_image_output_are_preserved(self):
        image_url = "data:image/png;base64,AAAA"
        chat = router.responses_to_chat({
            "model": "deepseek-v4-flash",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Describe this image."},
                        {"type": "input_image", "image_url": image_url, "detail": "original"},
                    ],
                },
                {
                    "type": "function_call",
                    "call_id": "call_view",
                    "name": "view_image",
                    "arguments": {"path": "demo.png"},
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_view",
                    "output": [
                        {"type": "input_text", "text": "Loaded image."},
                        {"type": "input_image", "image_url": image_url, "detail": "high"},
                    ],
                },
            ],
            "tools": [{
                "type": "function",
                "name": "view_image",
                "parameters": {"type": "object", "properties": {}},
            }],
        })

        user_content = chat["messages"][0]["content"]
        self.assertEqual(user_content[1]["type"], "image_url")
        self.assertEqual(user_content[1]["image_url"]["detail"], "high")
        self.assertEqual(chat["messages"][2]["role"], "tool")
        self.assertEqual(chat["messages"][3]["role"], "user")
        self.assertEqual(chat["messages"][3]["content"][1]["image_url"]["url"], image_url)

    def test_parallel_tool_images_are_appended_after_all_tool_outputs(self):
        image_url = "data:image/png;base64,AAAA"
        chat = router.responses_to_chat({
            "model": "deepseek-v4-flash",
            "input": [
                {"type": "function_call", "call_id": "call_1", "name": "first", "arguments": {}},
                {"type": "function_call", "call_id": "call_2", "name": "second", "arguments": {}},
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": [{"type": "input_image", "image_url": image_url}],
                },
                {"type": "function_call_output", "call_id": "call_2", "output": "two"},
            ],
        })

        self.assertEqual(
            [message["role"] for message in chat["messages"]],
            ["assistant", "tool", "tool", "user"],
        )
        self.assertEqual(
            [message["tool_call_id"] for message in chat["messages"][1:3]],
            ["call_1", "call_2"],
        )
        self.assertEqual(chat["messages"][3]["content"][1]["image_url"]["url"], image_url)

    def test_namespace_tool_is_flattened_and_history_uses_flat_name(self):
        body = {
            "model": "deepseek-v4-flash",
            "input": [
                {
                    "type": "function_call",
                    "call_id": "call_thread",
                    "namespace": "codex_app",
                    "name": "read_thread",
                    "arguments": {"threadId": "thread_1"},
                },
                {"type": "function_call_output", "call_id": "call_thread", "output": "ok"},
            ],
            "tools": [{
                "type": "namespace",
                "name": "codex_app",
                "description": "Codex app tools.",
                "tools": [{
                    "type": "function",
                    "name": "read_thread",
                    "description": "Read a thread.",
                    "parameters": {
                        "type": "object",
                        "properties": {"threadId": {"type": "string", "encrypted": True}},
                        "required": ["threadId"],
                    },
                }],
            }],
        }
        bridge = router._build_tool_bridge(body)
        chat = router.responses_to_chat(body, tool_bridge=bridge)

        self.assertEqual(chat["tools"][0]["function"]["name"], "codex_app__read_thread")
        self.assertNotIn("encrypted", chat["tools"][0]["function"]["parameters"]["properties"]["threadId"])
        self.assertEqual(
            chat["messages"][0]["tool_calls"][0]["function"]["name"],
            "codex_app__read_thread",
        )

    def test_additional_tools_are_loaded_into_the_chat_tool_bridge(self):
        body = {
            "model": "deepseek-v4-flash",
            "input": [{
                "type": "additional_tools",
                "role": "assistant",
                "tools": [{
                    "type": "namespace",
                    "name": "collaboration",
                    "description": "Sub-agent tools.",
                    "tools": [{
                        "type": "function",
                        "name": "spawn_agent",
                        "description": "Spawn a sub-agent.",
                        "parameters": {
                            "type": "object",
                            "properties": {"task_name": {"type": "string"}},
                            "required": ["task_name"],
                        },
                    }],
                }],
            }],
        }

        bridge = router._build_tool_bridge(body)
        chat = router.responses_to_chat(body, tool_bridge=bridge)

        self.assertIn("collaboration__spawn_agent", bridge["by_chat"])
        self.assertEqual(chat["tools"][0]["function"]["name"], "collaboration__spawn_agent")
        self.assertEqual(chat["messages"], [])

    def test_tool_search_gets_codex_query_schema(self):
        chat = router.responses_to_chat({
            "model": "deepseek-v4-flash",
            "input": "Find a collaboration tool.",
            "tools": [{
                "type": "tool_search",
                "execution": "client",
                "description": "Discover deferred client tools.",
            }],
        })

        function = chat["tools"][0]["function"]
        self.assertEqual(function["name"], "tool_search")
        self.assertEqual(function["parameters"]["required"], ["query"])
        self.assertIn("limit", function["parameters"]["properties"])

    def test_json_schema_text_format_maps_to_chat_response_format(self):
        chat = router.responses_to_chat({
            "model": "deepseek-v4-flash",
            "input": "Return a title.",
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "title_schema",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"title": {"type": "string"}},
                        "required": ["title"],
                    },
                },
            },
        })

        self.assertEqual(chat["response_format"]["type"], "json_schema")
        self.assertEqual(chat["response_format"]["json_schema"]["name"], "title_schema")
        self.assertIn("Return only JSON matching this schema", chat["messages"][0]["content"])

    def test_text_verbosity_becomes_a_chat_system_instruction(self):
        chat = router.responses_to_chat({
            "model": "deepseek-v4-flash",
            "input": "Explain the result.",
            "text": {"verbosity": "low"},
        })

        self.assertIn("concise", chat["messages"][0]["content"])
        self.assertEqual(chat["messages"][1]["role"], "user")

    def test_server_hosted_tools_are_reported_instead_of_silently_dropped(self):
        chat = router.responses_to_chat({
            "model": "deepseek-v4-flash",
            "input": "Search the web.",
            "tools": [{"type": "web_search", "external_web_access": True}],
        })

        self.assertNotIn("tools", chat)
        self.assertIn("web_search", chat["messages"][0]["content"])

    def test_chat_history_function_arguments_remain_a_json_string(self):
        responses = router.chat_to_responses({
            "model": "gpt-5.6",
            "messages": [{
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_read",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path":"README.md"}',
                    },
                }],
            }],
        })

        arguments = responses["input"][0]["arguments"]
        self.assertIsInstance(arguments, str)
        self.assertEqual(arguments, '{"path":"README.md"}')

    def test_chat_reasoning_content_becomes_a_standalone_reasoning_item(self):
        responses = router.chat_to_responses({
            "model": "gpt-5.6",
            "messages": [{
                "role": "assistant",
                "reasoning_content": "Inspect the repository first.",
                "content": "I will inspect it.",
            }],
        })

        self.assertEqual([item["type"] for item in responses["input"]], ["reasoning", "message"])
        reasoning, message = responses["input"]
        self.assertEqual(
            reasoning["summary"],
            [{"type": "summary_text", "text": "Inspect the repository first."}],
        )
        self.assertNotIn("reasoning", message)

    def test_sanitizer_converts_agent_message_and_unwraps_encrypted_content(self):
        original = {
            "model": "responses-model",
            "input": [
                {
                    "type": "agent_message",
                    "author": "subagent",
                    "content": [
                        {"type": "input_text", "text": "Message Type: NEW_TASK ... Payload:\n"},
                        {"type": "encrypted_content", "encrypted_content": "REPRO-TOKEN-1234"},
                    ],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Original user."}],
                },
            ],
        }

        sanitized = router._sanitize_responses_for_non_openai(original)

        self.assertEqual(original["input"][0]["type"], "agent_message")
        self.assertIn("REPRO-TOKEN-1234", json.dumps(sanitized, ensure_ascii=False))
        self.assertNotIn("encrypted_content", json.dumps(sanitized, ensure_ascii=False))
        first = sanitized["input"][0]
        self.assertEqual(first["type"], "message")
        self.assertEqual(first["role"], "user")
        self.assertIn("NEW_TASK", first["content"][0]["text"])
        self.assertEqual(first["content"][1]["text"], "REPRO-TOKEN-1234")
        self.assertEqual(sanitized["input"][1], original["input"][1])

    def test_sanitizer_replaces_encrypted_content_parts_in_standard_messages(self):
        original = {
            "input": [{
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Hello"},
                    {"type": "encrypted_content", "encrypted_content": "REPRO-TOKEN-5678"},
                ],
            }],
        }

        sanitized = router._sanitize_responses_for_non_openai(original)
        content = sanitized["input"][0]["content"]

        self.assertEqual(
            content,
            [
                {"type": "input_text", "text": "Hello"},
                {"type": "input_text", "text": "REPRO-TOKEN-5678"},
            ],
        )

    def test_sanitizer_uses_placeholder_for_non_string_encrypted_content(self):
        original = {
            "input": [{
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "encrypted_content", "encrypted_content": None},
                ],
            }],
        }

        sanitized = router._sanitize_responses_for_non_openai(original)
        content = sanitized["input"][0]["content"]

        self.assertEqual(
            content,
            [{"type": "input_text", "text": router._OPENAI_ENCRYPTED_PLACEHOLDER}],
        )

    def test_openai_sanitize_default_and_override(self):
        self.assertTrue(router._should_sanitize_openai_fields({
            "wire_api": "responses",
            "base_url": "https://provider.example/v1",
        }))
        self.assertFalse(router._should_sanitize_openai_fields({
            "wire_api": "responses",
            "base_url": "https://api.openai.com/v1",
        }))
        self.assertFalse(router._should_sanitize_openai_fields({
            "wire_api": "responses",
            "base_url": "https://provider.example/v1",
            "openai_sanitize": False,
        }))
        self.assertTrue(router._should_sanitize_openai_fields({
            "wire_api": "responses",
            "base_url": "https://api.openai.com/v1",
            "openai_sanitize": True,
        }))


class ChatCompatibilityFallbackTests(unittest.TestCase):
    def test_thinking_tool_choice_falls_back_to_auto_with_system_requirement(self):
        body = {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "system", "content": "Keep existing instructions."}],
            "tool_choice": {
                "type": "function",
                "function": {"name": "apply_patch"},
            },
        }

        fallback, name = router._next_chat_compat_fallback(
            body,
            400,
            '{"error":{"message":"Thinking mode does not support this tool_choice"}}',
        )

        self.assertEqual(name, "tool_choice")
        self.assertEqual(fallback["tool_choice"], "auto")
        self.assertIn("apply_patch", fallback["messages"][0]["content"])
        self.assertEqual(fallback["messages"][1], body["messages"][0])
        self.assertIsInstance(body["tool_choice"], dict)
        self.assertEqual(len(body["messages"]), 1)

        repeated = router._next_chat_compat_fallback(
            body,
            400,
            "Thinking mode does not support this tool_choice",
            {"tool_choice"},
        )
        self.assertEqual(repeated, (None, None))

    def test_text_only_chat_upstream_retries_images_as_text_placeholders(self):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is shown here?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,AAAA", "detail": "high"},
                        },
                    ],
                },
            ],
        }

        fallback, name = router._next_chat_compat_fallback(
            body,
            400,
            "messages[115]: unknown variant `image_url`, expected `text`",
        )

        self.assertEqual(name, "image_url")
        self.assertEqual(
            fallback["messages"][0]["content"],
            [
                {"type": "text", "text": "What is shown here?"},
                {
                    "type": "text",
                    "text": "[Image unavailable: this upstream accepts text-only messages.]",
                },
            ],
        )
        self.assertEqual(body["messages"][0]["content"][1]["type"], "image_url")
        self.assertNotIn("data:image/png", json.dumps(fallback))

    def test_unrelated_bad_request_does_not_remove_images(self):
        body = {
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}],
        }

        fallback, name = router._next_chat_compat_fallback(
            body,
            400,
            "Invalid request body",
        )

        self.assertIsNone(fallback)
        self.assertIsNone(name)

    def test_unsupported_response_format_is_removed_without_losing_schema_prompt(self):
        body = {
            "messages": [{"role": "system", "content": "Return only JSON matching this schema: {}"}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "result"}},
        }

        fallback, name = router._next_chat_compat_fallback(
            body,
            400,
            "Unsupported parameter: response_format",
        )

        self.assertEqual(name, "response_format")
        self.assertNotIn("response_format", fallback)
        self.assertEqual(fallback["messages"], body["messages"])
        self.assertIn("response_format", body)

        unavailable, unavailable_name = router._next_chat_compat_fallback(
            body,
            400,
            "This response_format type is unavailable now",
        )
        self.assertEqual(unavailable_name, "response_format")
        self.assertNotIn("response_format", unavailable)

    def test_unsupported_max_completion_tokens_becomes_max_tokens(self):
        body = {"messages": [], "max_completion_tokens": 2048}

        fallback, name = router._next_chat_compat_fallback(
            body,
            400,
            "max_completion_tokens is not supported by this model",
        )

        self.assertEqual(name, "max_completion_tokens")
        self.assertNotIn("max_completion_tokens", fallback)
        self.assertEqual(fallback["max_tokens"], 2048)
        self.assertNotIn("max_tokens", body)

    def test_optional_chat_parameters_are_removed_individually(self):
        body = {
            "messages": [],
            "parallel_tool_calls": True,
            "reasoning_effort": "high",
        }
        cases = (
            ("parallel_tool_calls", "Unknown parameter: parallel_tool_calls"),
            ("reasoning_effort", "reasoning_effort is not supported"),
        )
        for parameter, error in cases:
            with self.subTest(parameter=parameter):
                fallback, name = router._next_chat_compat_fallback(body, 400, error)
                self.assertEqual(name, parameter)
                self.assertNotIn(parameter, fallback)
                other = "reasoning_effort" if parameter == "parallel_tool_calls" else "parallel_tool_calls"
                self.assertIn(other, fallback)

    def test_auth_balance_and_general_bad_requests_do_not_retry(self):
        body = {
            "messages": [],
            "response_format": {"type": "json_object"},
            "max_completion_tokens": 100,
        }
        cases = (
            (401, "Unsupported parameter: response_format"),
            (400, "Insufficient balance"),
            (400, "Invalid API key"),
            (400, "Invalid response_format schema"),
            (400, "Malformed request"),
        )
        for status, error in cases:
            with self.subTest(status=status, error=error):
                self.assertEqual(
                    router._next_chat_compat_fallback(body, status, error),
                    (None, None),
                )


class RuntimeControlTests(unittest.TestCase):
    def test_reload_never_restarts_and_reports_manual_port_restart(self):
        with (
            patch.object(router, "load_config", return_value={"port": 9876}),
            patch.object(router, "get_current_port", return_value=8765),
            patch.object(router.stats, "load") as load_stats,
        ):
            result = router._reload_runtime_state()

        self.assertTrue(result["restart_required"])
        self.assertFalse(result["restarting"])
        load_stats.assert_called_once_with()
        self.assertFalse(hasattr(router, "set_restart_callback"))


class ChatToResponsesTests(unittest.TestCase):
    def test_upstream_error_becomes_response_failed(self):
        converter = router.ChatToResponses("deepseek-v4-flash")

        streamed = converter.on_chunk({
            "type": "error",
            "code": "invalid_request_error",
            "message": "Bad request.",
            "param": "messages",
        })
        final = converter.final_events()

        self.assertEqual(streamed, [])
        self.assertEqual(final[-1]["type"], "response.failed")
        self.assertEqual(final[-1]["response"]["status"], "failed")
        self.assertEqual(final[-1]["response"]["error"]["code"], "invalid_request_error")
        self.assertEqual(final[-1]["response"]["error"]["message"], "Bad request.")
        self.assertEqual(final[-1]["response"]["error"]["param"], "messages")
        self.assertNotIn("response.completed", [event["type"] for event in final])

    def test_length_and_content_filter_become_response_incomplete(self):
        for finish_reason, expected_reason in (
            ("length", "max_output_tokens"),
            ("content_filter", "content_filter"),
        ):
            converter = router.ChatToResponses("deepseek-v4-flash")
            converter.on_chunk({
                "choices": [{"delta": {"content": "partial"}, "finish_reason": finish_reason}],
            })

            final = converter.final_events()

            self.assertEqual(final[-1]["type"], "response.incomplete")
            self.assertEqual(final[-1]["response"]["status"], "incomplete")
            self.assertEqual(
                final[-1]["response"]["incomplete_details"]["reason"],
                expected_reason,
            )
            self.assertNotIn("response.completed", [event["type"] for event in final])

    def test_reasoning_summary_events_use_codex_response_fields(self):
        converter = router.ChatToResponses("deepseek-v4-flash")

        streamed = converter.on_chunk({
            "choices": [{"delta": {"reasoning_content": "Inspect first."}, "finish_reason": None}],
        })
        delta = next(event for event in streamed if event["type"] == "response.reasoning_summary_text.delta")
        self.assertEqual(delta["summary_index"], 0)
        self.assertEqual(delta["delta"], "Inspect first.")

        final = converter.final_events()
        done = next(event for event in final if event["type"] == "response.reasoning_summary_text.done")
        self.assertEqual(done["summary_index"], 0)
        self.assertEqual(done["text"], "Inspect first.")
        self.assertNotIn("summary", done)

    def test_plain_text_is_buffered_and_marked_as_final_answer(self):
        converter = router.ChatToResponses("deepseek-v4-flash")

        first = converter.on_chunk({
            "choices": [{"delta": {"content": "Finished."}, "finish_reason": None}],
        })
        self.assertEqual(first, [])

        closing = converter.on_chunk({
            "choices": [{"delta": {}, "finish_reason": "stop"}],
        })
        added = next(event for event in closing if event["type"] == "response.output_item.added")
        self.assertEqual(added["item"]["phase"], "final_answer")

        final = converter.final_events()
        done = next(
            event for event in final
            if event["type"] == "response.output_item.done" and event["item"]["type"] == "message"
        )
        completed = final[-1]["response"]
        self.assertEqual(done["item"]["phase"], "final_answer")
        self.assertEqual(completed["output"][0]["phase"], "final_answer")

    def test_text_before_a_tool_call_is_marked_as_commentary(self):
        converter = router.ChatToResponses("deepseek-v4-flash")
        converter.on_chunk({
            "choices": [{"delta": {"content": "I will inspect it."}, "finish_reason": None}],
        })

        events = converter.on_chunk({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_1",
                        "function": {"name": "shell_command", "arguments": "{}"},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
        })

        added = [event for event in events if event["type"] == "response.output_item.added"]
        self.assertEqual(added[0]["item"]["type"], "message")
        self.assertEqual(added[0]["item"]["phase"], "commentary")
        self.assertEqual(added[1]["item"]["type"], "function_call")
        final = converter.final_events()
        done = next(
            event for event in final
            if event["type"] == "response.output_item.done" and event["item"]["type"] == "message"
        )
        self.assertEqual(done["item"]["phase"], "commentary")

    def test_custom_chat_tool_call_is_restored_to_responses_custom_call(self):
        converter = router.ChatToResponses("deepseek-v4-flash", custom_tool_names={"apply_patch"})
        events = converter.on_chunk({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_patch",
                        "function": {
                            "name": "apply_patch",
                            "arguments": json.dumps({"input": "*** Begin Patch\n*** End Patch"}),
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
        })

        added = next(event for event in events if event["type"] == "response.output_item.added")
        self.assertEqual(added["item"]["type"], "custom_tool_call")
        self.assertNotIn("response.function_call_arguments.delta", [event["type"] for event in events])

        final = converter.final_events()
        input_done = next(event for event in final if event["type"] == "response.custom_tool_call_input.done")
        item_done = next(
            event for event in final
            if event["type"] == "response.output_item.done" and event["item"]["type"] == "custom_tool_call"
        )
        self.assertEqual(input_done["input"], "*** Begin Patch\n*** End Patch")
        self.assertEqual(item_done["item"]["call_id"], "call_patch")
        self.assertEqual(item_done["item"]["input"], "*** Begin Patch\n*** End Patch")

    def test_namespace_chat_call_is_restored_with_namespace(self):
        body = {
            "tools": [{
                "type": "namespace",
                "name": "codex_app",
                "tools": [{
                    "type": "function",
                    "name": "read_thread",
                    "parameters": {"type": "object", "properties": {}},
                }],
            }],
        }
        bridge = router._build_tool_bridge(body)
        converter = router.ChatToResponses("deepseek-v4-flash", tool_bridge=bridge)
        converter.on_chunk({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_thread",
                        "function": {
                            "name": "codex_app__read_thread",
                            "arguments": '{"threadId":"thread_1"}',
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
        })

        final = converter.final_events()
        item_done = next(
            event for event in final
            if event["type"] == "response.output_item.done" and event["item"]["type"] == "function_call"
        )
        self.assertEqual(item_done["item"]["name"], "read_thread")
        self.assertEqual(item_done["item"]["namespace"], "codex_app")
        self.assertTrue(any(event["type"] == "response.function_call_arguments.done" for event in final))

    def test_tool_search_chat_call_is_restored_for_codex_execution(self):
        body = {
            "tools": [{
                "type": "tool_search",
                "execution": "client",
                "description": "Discover deferred client tools.",
            }],
        }
        bridge = router._build_tool_bridge(body)
        converter = router.ChatToResponses("deepseek-v4-flash", tool_bridge=bridge)
        converter.on_chunk({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_search",
                        "function": {
                            "name": "tool_search",
                            "arguments": '{"query":"spawn a sub-agent","limit":8}',
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
        })

        final = converter.final_events()
        item_done = next(
            event for event in final
            if event["type"] == "response.output_item.done"
            and event["item"]["type"] == "tool_search_call"
        )
        self.assertEqual(item_done["item"]["call_id"], "call_search")
        self.assertEqual(item_done["item"]["execution"], "client")
        self.assertEqual(item_done["item"]["arguments"]["limit"], 8)
        self.assertNotIn("name", item_done["item"])


class UsageSnifferTests(unittest.TestCase):
    def test_sniffs_failed_incomplete_and_completed_statuses(self):
        failed = router.UsageSniffer()
        failed.feed(b'data: {"type":"response.failed"}\n\n')
        self.assertEqual(failed.status, "failed")

        incomplete = router.UsageSniffer()
        incomplete.feed(b'data: {"type":"response.incomplete"}\n\n')
        self.assertEqual(incomplete.status, "incomplete")

        completed = router.UsageSniffer()
        completed.feed(
            b'data: {"type":"response.completed","response":{'
            b'"status":"completed","usage":{"total_tokens":5}}}\n\n'
        )
        self.assertEqual(completed.status, "completed")
        self.assertEqual(completed.usage["total_tokens"], 5)


class ResponsesToChatTests(unittest.TestCase):
    def test_incomplete_terminal_event_marks_length_finish_reason(self):
        converter = router.ResponsesToChat("responses-model")
        converter.on_event({
            "type": "response.incomplete",
            "response": {
                "status": "incomplete",
                "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
            },
        })

        final = converter.final_chunk()

        self.assertEqual(final["choices"][0]["finish_reason"], "length")
        self.assertEqual(final["usage"]["total_tokens"], 5)


if __name__ == "__main__":
    unittest.main()
