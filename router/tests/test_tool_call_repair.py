import itertools
import json
import unittest

import router


class ResponsesRepairTests(unittest.TestCase):
    def test_missing_output_is_filled_after_its_call(self):
        body = {
            "input": [
                {"type": "message", "role": "user", "content": []},
                {"type": "function_call", "call_id": "call_a", "name": "pwsh", "arguments": "{}"},
                {"type": "message", "role": "user", "content": []},
            ],
        }
        filled = router.repair_dangling_tool_calls(body, "responses")

        self.assertEqual(filled, 1)
        types = [item.get("type") for item in body["input"]]
        self.assertEqual(types, [
            "message", "function_call", "function_call_output", "message",
        ])
        self.assertEqual(body["input"][2]["call_id"], "call_a")

    def test_answered_calls_are_left_alone(self):
        body = {
            "input": [
                {"type": "function_call", "call_id": "call_a", "name": "pwsh"},
                {"type": "function_call_output", "call_id": "call_a", "output": "done"},
            ],
        }
        snapshot = json.dumps(body, sort_keys=True)
        self.assertEqual(router.repair_dangling_tool_calls(body, "responses"), 0)
        self.assertEqual(json.dumps(body, sort_keys=True), snapshot)

    def test_partial_answers_only_fill_the_missing_ones(self):
        body = {
            "input": [
                {"type": "function_call", "call_id": "call_a", "name": "pwsh"},
                {"type": "function_call", "call_id": "call_b", "name": "read"},
                {"type": "function_call_output", "call_id": "call_a", "output": "ok"},
            ],
        }
        self.assertEqual(router.repair_dangling_tool_calls(body, "responses"), 1)
        outputs = [i for i in body["input"] if i["type"] == "function_call_output"]
        # 只补缺失的那一条，已有的 call_a 结果原样保留。
        self.assertEqual(
            sorted(o["call_id"] for o in outputs), ["call_a", "call_b"],
        )
        # 占位结果必须紧跟在它自己的调用后面。
        repaired_types = [i.get("type") for i in body["input"]]
        self.assertEqual(
            repaired_types,
            ["function_call", "function_call", "function_call_output", "function_call_output"],
        )
        self.assertEqual(body["input"][2]["call_id"], "call_b")

    def test_custom_and_tool_search_outputs_match_their_call_type(self):
        body = {
            "input": [
                {"type": "custom_tool_call", "call_id": "call_c", "name": "apply_patch"},
                {"type": "tool_search_call", "call_id": "call_d", "execution": "client"},
            ],
        }
        self.assertEqual(router.repair_dangling_tool_calls(body, "responses"), 2)
        by_call = {
            i["call_id"]: i["type"] for i in body["input"] if i.get("call_id")
        }
        self.assertEqual(by_call["call_c"], "custom_tool_call_output")
        self.assertEqual(by_call["call_d"], "tool_search_output")

    def test_body_without_calls_is_untouched(self):
        body = {"input": [{"type": "message", "role": "user", "content": []}]}
        snapshot = json.dumps(body, sort_keys=True)
        self.assertEqual(router.repair_dangling_tool_calls(body, "responses"), 0)
        self.assertEqual(json.dumps(body, sort_keys=True), snapshot)


class ChatRepairTests(unittest.TestCase):
    def test_missing_tool_result_is_appended_after_the_batch(self):
        body = {
            "messages": [
                {"role": "user", "content": "跑一下"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "call_a", "type": "function", "function": {"name": "pwsh", "arguments": "{}"}},
                        {"id": "call_b", "type": "function", "function": {"name": "read", "arguments": "{}"}},
                    ],
                },
                {"role": "tool", "tool_call_id": "call_a", "content": "ok"},
            ],
        }
        filled = router.repair_dangling_tool_calls(body, "chat")

        self.assertEqual(filled, 1)
        roles = [m["role"] for m in body["messages"]]
        self.assertEqual(roles, ["user", "assistant", "tool", "tool"])
        self.assertEqual(body["messages"][-1]["tool_call_id"], "call_b")

    def test_complete_batches_are_untouched(self):
        body = {
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [{"id": "call_a", "function": {"name": "pwsh", "arguments": "{}"}}],
                },
                {"role": "tool", "tool_call_id": "call_a", "content": "ok"},
                {"role": "user", "content": "继续"},
            ],
        }
        snapshot = json.dumps(body, sort_keys=True)
        self.assertEqual(router.repair_dangling_tool_calls(body, "chat"), 0)
        self.assertEqual(json.dumps(body, sort_keys=True), snapshot)

    def test_placeholder_keeps_the_batch_contiguous(self):
        """占位结果必须紧跟在它自己的批次后面，不能被后来的消息隔开。"""
        body = {
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [{"id": "call_a", "function": {"name": "pwsh", "arguments": "{}"}}],
                },
                {"role": "user", "content": "继续"},
            ],
        }
        self.assertEqual(router.repair_dangling_tool_calls(body, "chat"), 1)
        roles = [m["role"] for m in body["messages"]]
        self.assertEqual(roles, ["assistant", "tool", "user"])

    def test_parallel_call_ids_each_get_one_placeholder(self):
        ids = [f"call_{i}" for i in range(5)]
        body = {
            "messages": [{
                "role": "assistant",
                "tool_calls": [
                    {"id": cid, "function": {"name": "pwsh", "arguments": "{}"}} for cid in ids
                ],
            }],
        }
        self.assertEqual(router.repair_dangling_tool_calls(body, "chat"), 5)
        self.assertEqual([m["tool_call_id"] for m in body["messages"][1:]], ids)


class OrphanRepairTests(unittest.TestCase):
    def test_responses_orphan_output_is_removed(self):
        """上游对"有结果、没调用"的孤儿项同样报错，转发前要删掉。"""
        body = {
            "input": [
                {"type": "function_call", "call_id": "call_a", "name": "pwsh", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "call_a", "output": "ok"},
                {"type": "function_call_output", "call_id": "call_ghost", "output": "stale"},
            ],
        }
        repaired = router.repair_dangling_tool_calls(body, "responses")

        self.assertEqual(repaired, 1)
        self.assertEqual(
            [item["type"] for item in body["input"]],
            ["function_call", "function_call_output"],
        )
        self.assertEqual(body["input"][1]["call_id"], "call_a")

    def test_chat_orphan_tool_result_is_removed(self):
        body = {
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [{"id": "call_a", "function": {"name": "pwsh", "arguments": "{}"}}],
                },
                {"role": "tool", "tool_call_id": "call_a", "content": "ok"},
                {"role": "tool", "tool_call_id": "call_ghost", "content": "stale"},
            ],
        }
        repaired = router.repair_dangling_tool_calls(body, "chat")

        self.assertEqual(repaired, 1)
        self.assertEqual([m["role"] for m in body["messages"]], ["assistant", "tool"])
        self.assertEqual(body["messages"][1]["tool_call_id"], "call_a")


class RepairWiringTests(unittest.TestCase):
    def test_chat_and_responses_dispatch_by_wire(self):
        chat_body = {"messages": [{
            "role": "assistant",
            "tool_calls": [{"id": "call_x", "function": {"name": "pwsh", "arguments": "{}"}}],
        }]}
        responses_body = {"input": [{"type": "function_call", "call_id": "call_y", "name": "pwsh"}]}

        self.assertEqual(router.repair_dangling_tool_calls(chat_body, "chat"), 1)
        self.assertEqual(router.repair_dangling_tool_calls(responses_body, "responses"), 1)

    def test_supplier_can_opt_out(self):
        self.assertTrue(router._should_repair_tool_calls({"name": "pixel"}))
        self.assertFalse(router._should_repair_tool_calls({"name": "pixel", "repair_tool_calls": False}))
        self.assertTrue(router._should_repair_tool_calls({"name": "pixel", "repair_tool_calls": True}))


class UpstreamErrorTextTests(unittest.TestCase):
    def test_nested_upstream_error_text_is_unwrapped(self):
        """像素把错误包了三层转义，客户端不该看到一坨 \\\" 转义 JSON。"""
        raw = json.dumps({
            "error": {
                "message": json.dumps({
                    "error": {
                        "code": "tool_call_mismatch",
                        "message": json.dumps({
                            "code": 11148,
                            "msg": "tool calls and tool results do not match",
                        }),
                    },
                }),
            },
        })
        frame = router.upstream_stream_failure_event("chat", "m", raw)
        payload = json.loads(frame.decode("utf-8").split("data: ", 1)[1])

        message = payload["error"]["message"]
        self.assertIn("tool calls and tool results do not match", message)
        self.assertIn("11148", message)
        self.assertNotIn('\\"', message)


if __name__ == "__main__":
    unittest.main()
