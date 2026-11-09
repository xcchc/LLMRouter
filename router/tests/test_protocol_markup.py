import unittest

import router


class ProtocolMarkupTests(unittest.TestCase):
    """畸形协议标记必须被剥掉，正常文本必须原样保留。"""

    def test_leaked_call_fragments_are_stripped(self):
        cases = {
            '="{" command":"ok">Let me just call the tool properly.</thinking>\n\n':
                "Let me just call the tool properly.\n\n",
            '" command":"> I made a malformed tool call. Let me retry properly.  \n':
                "I made a malformed tool call. Let me retry properly.  \n",
            '<｜begin▁of▁sentence｜>var">I made another malformed call.\n':
                "I made another malformed call.\n",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw[:40]):
                self.assertEqual(router._strip_protocol_markup(raw), expected)

    def test_marker_only_text_disappears(self):
        for raw in ("｜DSML｜ calls>\n", "｜DSML｜ calls>", "<｜end▁of▁sentence｜>"):
            with self.subTest(raw=raw):
                self.assertEqual(router._strip_protocol_markup(raw).strip(), "")

    def test_head_only_noise_is_trimmed(self):
        self.assertEqual(
            router._strip_protocol_markup("@@Hmm, I see. Let me continue."),
            "Hmm, I see. Let me continue.",
        )

    def test_ordinary_text_is_untouched(self):
        for text in (
            "我已经把仓库从 README 到源码都核对过一遍了。",
            "def foo(x):\n    return x + 1\n",
            "Use `cat file | grep of | head`.",
            '<div class="a">内容</div>',
            "| a | b |\n|---|---|\n| 1 | 2 |",
            "a < b 且 b > c，这是比较运算符。",
            "对比 ｜ 分隔符的用法",
            "The ratio of 0.5 is fine, and 3 of 5 items match.",
        ):
            with self.subTest(text=text[:30]):
                self.assertEqual(router._strip_protocol_markup(text), text)

    def test_ordinary_text_in_the_middle_is_untouched(self):
        text = "第一段正常正文。\n\n第二段正常正文，含 = 与 \" 号。\n"
        self.assertEqual(router._strip_protocol_markup(text, at_start=False), text)


class MarkupScrubberTests(unittest.TestCase):
    """标记被流切成多块时也要清干净。"""

    def _run(self, chunks):
        scrubber = router._MarkupScrubber()
        out = "".join(scrubber.feed(chunk) for chunk in chunks) + scrubber.flush()
        return out

    def test_marker_split_across_chunks(self):
        out = self._run([
            "<｜begin", "▁of▁sentence", "｜>var\">I made another", " malformed call.",
        ])
        self.assertEqual(out, "I made another malformed call.")

    def test_dsml_marker_split_across_chunks(self):
        out = self._run(["正常开头 ", "｜DSML", "｜ calls>", " 后续正常文字"])
        self.assertEqual(out, "正常开头  后续正常文字")

    def test_call_fragment_split_across_chunks(self):
        out = self._run([
            '="{" command', '":"ok">Let me just',
            " call the tool properly.</thinking>", "然后正常内容继续",
        ])
        self.assertEqual(out, "Let me just call the tool properly.然后正常内容继续")

    def test_character_by_character_stream(self):
        text = '<｜begin▁of▁sentence｜>var">bad</thinking>正常内容在这里。'
        self.assertEqual(self._run(list(text)), "bad正常内容在这里。")

    def test_long_plain_text_is_forwarded(self):
        text = "这是一段很长的正常正文，中间没有任何标记。" * 4
        self.assertEqual(self._run([text]), text)

    def test_replies_starting_with_quote_or_equals_are_not_held(self):
        """正常回复不能因为开头像残骸就被逐块暂存（会变成"打字停顿"）。"""
        for text in (
            '"这是引用的开头，后面还有内容。"',
            '="A1" 是单元格地址',
            "```python\nprint(1)\n```",
            "> 引用块的第一行\n> 第二行",
            "价格 <100 元，请确认",
        ):
            chunks = [text[i:i + 3] for i in range(0, len(text), 3)]
            with self.subTest(text=text[:24]):
                self.assertEqual(self._run(chunks), text)

    def test_plain_text_is_not_delayed(self):
        scrubber = router._MarkupScrubber()
        self.assertEqual(scrubber.feed("普通回答"), "普通回答")


class ChatToResponsesMarkupTests(unittest.TestCase):
    def test_leaked_markup_never_reaches_output_text(self):
        converter = router.ChatToResponses("deepseek-v4.1-flash")
        events = converter.on_chunk({
            "choices": [{
                "delta": {"content": "｜DSML｜ calls>\n"},
                "finish_reason": "tool_calls",
            }],
        })
        events += converter.final_events()

        text = "".join(
            event["delta"] for event in events
            if event["type"] == "response.output_text.delta"
        )
        self.assertEqual(text.strip(), "")
        done = [
            event for event in events if event["type"] == "response.output_text.done"
        ]
        self.assertTrue(all(not event["text"].strip() for event in done))

    def test_tool_arguments_are_not_touched(self):
        converter = router.ChatToResponses("deepseek-v4.1-flash")
        converter.on_chunk({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_1",
                        "function": {
                            "name": "pwsh",
                            "arguments": '{"command":"echo <｜begin▁of▁sentence｜>"}',
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
        })
        args = "".join(converter.tool_calls[0]["args"])
        self.assertEqual(args, '{"command":"echo <｜begin▁of▁sentence｜>"}')


class ResponsesToChatMarkupTests(unittest.TestCase):
    def test_leaked_markup_is_dropped_from_content_delta(self):
        converter = router.ResponsesToChat("deepseek-v4.1-flash")
        chunks = converter.on_event({
            "type": "response.output_text.delta",
            "delta": "｜DSML｜ calls>",
        })
        self.assertEqual(chunks, [])

    def test_plain_text_survives(self):
        converter = router.ResponsesToChat("deepseek-v4.1-flash")
        chunks = converter.on_event({
            "type": "response.output_text.delta",
            "delta": "好的，我继续。",
        })
        self.assertEqual(chunks[-1]["choices"][0]["delta"]["content"], "好的，我继续。")


if __name__ == "__main__":
    unittest.main()
