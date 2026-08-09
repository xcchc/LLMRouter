import json
import unittest

import httpx

import router


def tool_calls_from(chunks):
    calls = []
    for chunk in chunks:
        choices = chunk.get("choices") or []
        if choices:
            calls.extend((choices[0].get("delta") or {}).get("tool_calls") or [])
    return calls


class ResponsesToChatToolBridgeTests(unittest.IsolatedAsyncioTestCase):
    def test_plain_text_still_finishes_with_stop(self):
        converter = router.ResponsesToChat("gpt-5.6")
        chunks = converter.on_event({
            "type": "response.output_text.delta",
            "delta": "done",
        })

        self.assertEqual(chunks[-1]["choices"][0]["delta"]["content"], "done")
        self.assertEqual(converter.final_chunk()["choices"][0]["finish_reason"], "stop")

    def test_function_call_keeps_streaming_and_flattens_namespace(self):
        converter = router.ResponsesToChat("gpt-5.6")
        chunks = converter.on_event({
            "type": "response.output_item.added",
            "item": {
                "id": "fc_thread",
                "type": "function_call",
                "call_id": "call_thread",
                "namespace": "codex_app",
                "name": "read_thread",
                "arguments": "",
            },
        })
        chunks += converter.on_event({
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_thread",
            "delta": '{"threadId":"thread_1"}',
        })
        duplicate = converter.on_event({
            "type": "response.completed",
            "response": {
                "status": "completed",
                "output": [{
                    "id": "fc_thread_done",
                    "type": "function_call",
                    "call_id": "call_thread",
                    "namespace": "codex_app",
                    "name": "read_thread",
                    "arguments": '{"threadId":"thread_1"}',
                }],
            },
        })

        calls = tool_calls_from(chunks)
        self.assertEqual(calls[0]["id"], "call_thread")
        self.assertEqual(calls[0]["function"]["name"], "codex_app__read_thread")
        self.assertEqual(calls[1]["function"]["arguments"], '{"threadId":"thread_1"}')
        self.assertEqual(tool_calls_from(duplicate), [])
        self.assertEqual(converter.final_chunk()["choices"][0]["finish_reason"], "tool_calls")

    def test_custom_tool_call_waits_for_complete_input_and_wraps_it(self):
        converter = router.ResponsesToChat("gpt-5.6")
        chunks = converter.on_event({
            "type": "response.output_item.added",
            "item": {
                "id": "ctc_patch",
                "type": "custom_tool_call",
                "call_id": "call_patch",
                "name": "apply_patch",
                "input": "",
            },
        })
        self.assertEqual(len(tool_calls_from(chunks)), 1)
        self.assertEqual(converter.on_event({
            "type": "response.custom_tool_call_input.delta",
            "item_id": "ctc_patch",
            "delta": "*** Begin Patch\n",
        }), [])
        converter.on_event({
            "type": "response.custom_tool_call_input.delta",
            "item_id": "ctc_patch",
            "delta": "*** End Patch",
        })
        done = converter.on_event({
            "type": "response.custom_tool_call_input.done",
            "item_id": "ctc_patch",
            "input": "*** Begin Patch\n*** End Patch",
        })
        duplicate = converter.on_event({
            "type": "response.output_item.done",
            "item": {
                "id": "ctc_patch",
                "type": "custom_tool_call",
                "call_id": "call_patch",
                "name": "apply_patch",
                "input": "*** Begin Patch\n*** End Patch",
            },
        })

        arguments = tool_calls_from(done)[0]["function"]["arguments"]
        self.assertEqual(json.loads(arguments), {"input": "*** Begin Patch\n*** End Patch"})
        self.assertEqual(tool_calls_from(duplicate), [])
        self.assertEqual(converter.final_chunk()["choices"][0]["finish_reason"], "tool_calls")

    def test_tool_search_can_be_recovered_from_completed_output(self):
        converter = router.ResponsesToChat("gpt-5.6")
        chunks = converter.on_event({
            "type": "response.completed",
            "response": {
                "status": "completed",
                "output": [{
                    "id": "tsc_search",
                    "type": "tool_search_call",
                    "call_id": "call_search",
                    "status": "completed",
                    "execution": "client",
                    "arguments": {"query": "spawn a sub-agent", "limit": 8},
                }],
            },
        })

        calls = tool_calls_from(chunks)
        self.assertEqual(calls[0]["id"], "call_search")
        self.assertEqual(calls[0]["function"]["name"], "tool_search")
        self.assertEqual(json.loads(calls[1]["function"]["arguments"])["limit"], 8)
        self.assertEqual(converter.final_chunk()["choices"][0]["finish_reason"], "tool_calls")

    async def test_json_response_synthesizes_all_tool_types_as_chat_chunks(self):
        response = httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={
                "id": "resp_tools",
                "object": "response",
                "status": "completed",
                "output": [
                    {
                        "id": "fc_one",
                        "type": "function_call",
                        "call_id": "call_one",
                        "name": "lookup",
                        "arguments": {"q": "one"},
                    },
                    {
                        "id": "ctc_one",
                        "type": "custom_tool_call",
                        "call_id": "call_two",
                        "name": "apply_patch",
                        "input": "patch body",
                    },
                    {
                        "id": "tsc_one",
                        "type": "tool_search_call",
                        "call_id": "call_three",
                        "execution": "client",
                        "arguments": {"query": "more tools"},
                    },
                ],
            },
        )
        converter = router.ResponsesToChat("gpt-5.6")
        chunks = [chunk async for chunk in converter.convert(response)]

        calls = tool_calls_from(chunks)
        initial_calls = [call for call in calls if call.get("id")]
        self.assertEqual([call["function"]["name"] for call in initial_calls], [
            "lookup",
            "apply_patch",
            "tool_search",
        ])
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "tool_calls")


if __name__ == "__main__":
    unittest.main()
