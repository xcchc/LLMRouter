import unittest

import router


class StructuredChunkTests(unittest.TestCase):
    def test_structured_content_is_normalized_before_buffering(self):
        converter = router.ChatToResponses("deepseek-v4-flash")

        streamed = converter.on_chunk({
            "choices": [{
                "delta": {
                    "content": [
                        {"type": "text", "text": "Hello "},
                        {"type": "output_text", "text": {"value": "world"}},
                        {"content": [{"text": "!"}]},
                    ],
                },
                "finish_reason": "stop",
            }],
        })
        final = converter.final_events()

        deltas = [
            event["delta"]
            for event in streamed
            if event["type"] == "response.output_text.delta"
        ]
        done = next(event for event in final if event["type"] == "response.output_text.done")
        self.assertEqual(deltas, ["Hello world!"])
        self.assertEqual(done["text"], "Hello world!")
        self.assertTrue(all(isinstance(part, str) for part in converter.msg_text))

    def test_structured_reasoning_content_and_reasoning_are_normalized(self):
        converter = router.ChatToResponses("deepseek-v4-flash")

        first = converter.on_chunk({
            "choices": [{
                "delta": {
                    "reasoning_content": {
                        "summary": [
                            {"type": "summary_text", "text": "Inspect "},
                            {"content": "carefully."},
                        ],
                    },
                },
                "finish_reason": None,
            }],
        })
        second = converter.on_chunk({
            "choices": [{
                "delta": {
                    "reasoning_content": [],
                    "reasoning": {"content": [{"text": " Then finish."}]},
                },
                "finish_reason": "stop",
            }],
        })
        final = converter.final_events()

        deltas = [
            event["delta"]
            for event in first + second
            if event["type"] == "response.reasoning_summary_text.delta"
        ]
        done = next(
            event for event in final
            if event["type"] == "response.reasoning_summary_text.done"
        )
        self.assertEqual(deltas, ["Inspect carefully.", " Then finish."])
        self.assertEqual(done["text"], "Inspect carefully. Then finish.")
        self.assertTrue(all(isinstance(part, str) for part in converter.rs_text))

    def test_unknown_structured_content_uses_stable_json_text(self):
        converter = router.ChatToResponses("deepseek-v4-flash")

        converter.on_chunk({
            "choices": [{
                "message": {"content": {"z": 2, "a": {"label": "kept"}}},
                "finish_reason": "stop",
            }],
        })
        final = converter.final_events()

        done = next(event for event in final if event["type"] == "response.output_text.done")
        self.assertEqual(done["text"], '{"a": {"label": "kept"}, "z": 2}')


if __name__ == "__main__":
    unittest.main()
