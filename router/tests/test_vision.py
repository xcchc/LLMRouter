# -*- coding: utf-8 -*-
import unittest
from unittest.mock import AsyncMock, patch

import vision


class ImageHandlingModeTests(unittest.TestCase):
    def test_defaults_to_send_as_is(self):
        self.assertEqual(vision.image_handling_mode({}, "deepseek-v4-flash"), "send-as-is")
        self.assertEqual(vision.image_handling_mode({"image_handling": {}}, "deepseek-v4-flash"), "send-as-is")
        self.assertEqual(
            vision.image_handling_mode({"image_handling": {"other": "vlm"}}, "deepseek-v4-flash"),
            "send-as-is",
        )

    def test_supplier_level_string_mode(self):
        self.assertEqual(vision.image_handling_mode({"image_handling": "strip"}, "deepseek-v4-flash"), "strip")

    def test_model_map(self):
        supplier = {"image_handling": {"deepseek-v4-flash": "vlm"}}
        self.assertEqual(vision.image_handling_mode(supplier, "deepseek-v4-flash"), "vlm")

    def test_prompt_mode_defaults_to_main_model(self):
        self.assertEqual(vision.vlm_prompt_mode({}), "main-model")
        self.assertEqual(vision.vlm_prompt_mode({"vlm_prompt_mode": "template"}), "template")

    def test_prompt_uses_latest_user_question_and_icon_guidance(self):
        body = {
            "input": [
                {"type": "message", "role": "user", "content": "先看一下图片"},
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "这是什么软件的图标？"},
                        {"type": "input_image", "image_url": "u1"},
                    ],
                },
            ]
        }
        prompt = vision.build_vlm_prompt(body)
        self.assertIn("这是什么软件的图标", prompt)
        self.assertIn("具体软件、产品或品牌名称", prompt)

    def test_prompt_ignores_tool_image_boilerplate(self):
        body = {
            "messages": [
                {"role": "user", "content": "请找出截图里的报错原因"},
                {"role": "user", "content": "Image returned by the preceding tool call."},
            ]
        }
        self.assertEqual(vision._request_context(body), "请找出截图里的报错原因")

    def test_prompt_context_stops_at_each_image_message(self):
        body = {
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "第一张是什么软件图标？"},
                        {"type": "input_image", "image_url": "u1"},
                    ],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "这个呢？"},
                        {"type": "input_image", "image_url": "u2"},
                    ],
                },
            ]
        }
        locations = vision._image_locations(body)
        first = vision.build_vlm_prompt(body, locations[0])
        second = vision.build_vlm_prompt(body, locations[1])
        self.assertNotIn("这个呢", first)
        self.assertIn("这个呢", second)
        self.assertIn("只分析当前这一张图片", second)


class StripImageTests(unittest.TestCase):
    def test_strip_chat_image(self):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
                    ],
                }
            ]
        }
        self.assertTrue(vision.strip_images(body))
        self.assertEqual(
            body["messages"][0]["content"][1],
            {"type": "text", "text": vision.STRIP_PLACEHOLDER},
        )

    def test_strip_responses_image(self):
        body = {
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_image", "image_url": "data:image/png;base64,AAA"}],
                }
            ]
        }
        self.assertTrue(vision.strip_images(body))
        self.assertEqual(
            body["input"][0]["content"][0],
            {"type": "input_text", "text": vision.STRIP_PLACEHOLDER},
        )

    def test_no_images_returns_false(self):
        body = {"messages": [{"role": "user", "content": "plain text"}]}
        self.assertFalse(vision.strip_images(body))


class VlmApplyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        vision._cache.clear()
        vision._planner_cache.clear()

    def tearDown(self):
        vision._cache.clear()
        vision._planner_cache.clear()

    async def test_vlm_injects_description_into_chat(self):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello"},
                        {"type": "image_url", "image_url": {"url": "u1"}},
                    ],
                }
            ]
        }
        supplier = {
            "name": "text",
            "image_handling": {"deepseek-v4-flash": "vlm"},
            "vlm_supplier": "vision",
            "vlm_model": "gpt-5.6-luna",
        }
        cfg = {"suppliers": [{"name": "vision", "base_url": "https://vision.test/v1", "api_key": "sk-test"}]}
        with patch("vision.analyze_urls", new=AsyncMock(return_value={"u1": "一只猫的图片"})):
            await vision.apply_image_policy(body, supplier, cfg, "deepseek-v4-flash")
        self.assertEqual(
            body["messages"][0]["content"][1],
            {"type": "text", "text": "[图片描述] 一只猫的图片"},
        )

    async def test_vlm_supports_responses_input(self):
        body = {
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_image", "image_url": "u1"}],
                }
            ]
        }
        supplier = {
            "image_handling": {"deepseek-v4-flash": "vlm"},
            "vlm_supplier": "vision",
            "vlm_model": "gpt-5.6-luna",
        }
        cfg = {"suppliers": [{"name": "vision", "base_url": "https://vision.test/v1", "api_key": "sk-test"}]}
        with patch("vision.analyze_urls", new=AsyncMock(return_value={"u1": "图片描述"})):
            await vision.apply_image_policy(body, supplier, cfg, "deepseek-v4-flash")
        self.assertEqual(
            body["input"][0]["content"][0],
            {"type": "input_text", "text": "[图片描述] 图片描述"},
        )

    async def test_missing_provider_replaces_with_notice(self):
        body = {
            "messages": [
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "u1"}}]}
            ]
        }
        supplier = {"image_handling": {"deepseek-v4-flash": "vlm"}}
        cfg = {"suppliers": []}
        await vision.apply_image_policy(body, supplier, cfg, "deepseek-v4-flash")
        self.assertEqual(
            body["messages"][0]["content"][0],
            {"type": "text", "text": vision.VLM_MISSING_PROVIDER},
        )

    async def test_vlm_failure_uses_placeholder(self):
        body = {
            "messages": [
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "u1"}}]}
            ]
        }
        supplier = {
            "image_handling": {"deepseek-v4-flash": "vlm"},
            "vlm_supplier": "vision",
            "vlm_model": "gpt-5.6-luna",
        }
        cfg = {"suppliers": [{"name": "vision", "base_url": "https://vision.test/v1", "api_key": "sk-test"}]}
        with patch("vision.analyze_urls", new=AsyncMock(return_value={"u1": vision.VLM_FAILURE_PLACEHOLDER})):
            await vision.apply_image_policy(body, supplier, cfg, "deepseek-v4-flash")
        self.assertEqual(
            body["messages"][0]["content"][0],
            {"type": "text", "text": "[图片描述] " + vision.VLM_FAILURE_PLACEHOLDER},
        )

    async def test_multiple_history_images_are_analyzed_separately(self):
        body = {
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "第一张是什么软件图标？"},
                        {"type": "input_image", "image_url": "u1"},
                    ],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "这个呢？"},
                        {"type": "input_image", "image_url": "u2"},
                    ],
                },
            ]
        }
        supplier = {
            "base_url": "https://text.test/v1",
            "api_key": "sk-text",
            "wire_api": "chat",
            "image_handling": {"deepseek-v4-flash": "vlm"},
            "vlm_supplier": "vision",
            "vlm_model": "vision-model",
            "vlm_prompt_mode": "template",
        }
        cfg = {
            "suppliers": [
                {
                    "name": "vision",
                    "base_url": "https://vision.test/v1",
                    "api_key": "sk-vision",
                }
            ]
        }
        analyze = AsyncMock(side_effect=[{"u1": "Visual Studio Code"}, {"u2": "Microsoft Edge"}])
        with patch("vision.analyze_urls", new=analyze):
            await vision.apply_image_policy(body, supplier, cfg, "deepseek-v4-flash")
        self.assertEqual(body["input"][0]["content"][1]["text"], "[图片描述] Visual Studio Code")
        self.assertEqual(body["input"][1]["content"][1]["text"], "[图片描述] Microsoft Edge")
        self.assertEqual(analyze.await_count, 2)
        self.assertEqual(analyze.await_args_list[0].args[0], ["u1"])
        self.assertEqual(analyze.await_args_list[1].args[0], ["u2"])

    async def test_main_model_planner_is_added_to_vlm_prompt(self):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "这个呢？"},
                        {"type": "image_url", "image_url": {"url": "u1"}},
                    ],
                }
            ]
        }
        supplier = {
            "base_url": "https://text.test/v1",
            "api_key": "sk-text",
            "wire_api": "chat",
            "image_handling": {"deepseek-v4-flash": "vlm"},
            "vlm_supplier": "vision",
            "vlm_model": "vision-model",
            "vlm_prompt_mode": "main-model",
        }
        cfg = {
            "suppliers": [
                {
                    "name": "vision",
                    "base_url": "https://vision.test/v1",
                    "api_key": "sk-vision",
                }
            ]
        }
        analyze = AsyncMock(return_value={"u1": "Microsoft Edge"})
        with patch(
            "vision._call_prompt_planner",
            new=AsyncMock(return_value="识别当前图标的具体软件名称，并区分相似标志。"),
        ) as planner, patch("vision.analyze_urls", new=analyze):
            await vision.apply_image_policy(body, supplier, cfg, "deepseek-v4-flash")
        planner.assert_awaited_once()
        prompt = analyze.await_args.kwargs["prompt"]
        self.assertIn("主模型生成的视觉任务", prompt)
        self.assertIn("识别当前图标的具体软件名称", prompt)

    async def test_main_model_planner_prompt_is_cached(self):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "识别这个图标"},
                        {"type": "image_url", "image_url": {"url": "u1"}},
                    ],
                }
            ]
        }
        location = vision._image_locations(body)[0]
        supplier = {
            "base_url": "https://text.test/v1",
            "api_key": "sk-text",
            "wire_api": "chat",
            "vlm_prompt_mode": "main-model",
        }
        with patch(
            "vision._call_prompt_planner",
            new=AsyncMock(return_value="识别具体软件名称"),
        ) as planner:
            first = await vision.generate_vlm_prompt(body, location, supplier, "deepseek-v4-flash")
            second = await vision.generate_vlm_prompt(body, location, supplier, "deepseek-v4-flash")
        self.assertEqual(first, second)
        planner.assert_awaited_once()

    async def test_cache_hit_skips_network(self):
        provider = {"base_url": "https://vision.test/v1", "api_key": "sk-test", "model": "gpt-5.6-luna"}
        prompt = "identify this icon"
        context_key = "|".join((provider["base_url"], provider["model"], "chat", prompt))
        vision._cache_put("u1", "cached description", context_key)
        with patch("vision._call_vlm_batch_with_retry", new=AsyncMock(return_value="fresh")) as call:
            result = await vision.analyze_urls(["u1"], provider, prompt=prompt)
        self.assertEqual(result["u1"], "cached description")
        call.assert_not_awaited()

    async def test_cache_is_scoped_to_the_user_prompt(self):
        provider = {"base_url": "https://vision.test/v1", "api_key": "sk-test", "model": "vision"}
        with patch(
            "vision._call_vlm_batch_with_retry",
            new=AsyncMock(side_effect=["generic description", "identified icon"]),
        ) as call:
            first = await vision.analyze_urls(["u1"], provider, prompt="describe it")
            second = await vision.analyze_urls(["u1"], provider, prompt="identify the software")
        self.assertEqual(first["u1"], "generic description")
        self.assertEqual(second["u1"], "identified icon")
        self.assertEqual(call.await_count, 2)

    async def test_retry_then_success(self):
        provider = {"base_url": "https://vision.test/v1", "api_key": "sk-test", "model": "gpt-5.6-luna"}
        with patch(
            "vision._call_vlm_batch",
            new=AsyncMock(side_effect=[RuntimeError("VLM API 503"), "ok"]),
        ), patch("vision.asyncio.sleep", new=AsyncMock()):
            text = await vision._call_vlm_batch_with_retry(["u1"], provider)
        self.assertEqual(text, "ok")

    async def test_vlm_provider_preserves_responses_wire_api(self):
        supplier = {"vlm_supplier": "vision", "vlm_model": "gpt-5.6-luna"}
        cfg = {
            "suppliers": [
                {
                    "name": "vision",
                    "base_url": "https://vision.test/v1",
                    "api_key": "sk-test",
                    "wire_api": "responses",
                }
            ]
        }
        provider = vision._vlm_provider(cfg, supplier)
        self.assertEqual(provider["wire_api"], "responses")

    async def test_calls_responses_vlm_with_responses_payload(self):
        response = _FakeResponse(
            {
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "a chart screenshot"}],
                    }
                ]
            }
        )
        client = _FakeAsyncClient(response)
        provider = {
            "base_url": "https://vision.test/v1",
            "api_key": "sk-test",
            "model": "gpt-5.6-luna",
            "wire_api": "responses",
        }
        with patch("vision.httpx.AsyncClient", return_value=client):
            text = await vision._call_vlm_batch(["data:image/png;base64,AAA"], provider)
        self.assertEqual(text, "a chart screenshot")
        self.assertEqual(client.url, "https://vision.test/v1/responses")
        self.assertIn("input", client.payload)
        content = client.payload["input"][0]["content"]
        self.assertEqual(content[0]["type"], "input_text")
        self.assertEqual(content[1]["type"], "input_image")
        self.assertEqual(content[1]["detail"], "high")

    async def test_calls_chat_vlm_with_chat_payload(self):
        response = _FakeResponse({"choices": [{"message": {"content": "a cat"}}]})
        client = _FakeAsyncClient(response)
        provider = {
            "base_url": "https://vision.test/v1",
            "api_key": "sk-test",
            "model": "vision-model",
            "wire_api": "chat",
        }
        with patch("vision.httpx.AsyncClient", return_value=client):
            text = await vision._call_vlm_batch(["u1"], provider)
        self.assertEqual(text, "a cat")
        self.assertEqual(client.url, "https://vision.test/v1/chat/completions")
        self.assertIn("messages", client.payload)
        image = client.payload["messages"][0]["content"][0]["image_url"]
        self.assertEqual(image["detail"], "high")

    async def test_calls_chat_main_model_prompt_planner(self):
        response = _FakeResponse({"choices": [{"message": {"content": "inspect the current icon"}}]})
        client = _FakeAsyncClient(response)
        provider = {
            "base_url": "https://text.test/v1",
            "api_key": "sk-test",
            "model": "deepseek-v4-flash",
            "wire_api": "chat",
        }
        with patch("vision.httpx.AsyncClient", return_value=client):
            text = await vision._call_prompt_planner(provider, "这个呢？")
        self.assertEqual(text, "inspect the current icon")
        self.assertEqual(client.url, "https://text.test/v1/chat/completions")
        self.assertIn("messages", client.payload)

    async def test_calls_responses_main_model_prompt_planner(self):
        response = _FakeResponse(
            {
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "read the chart values"}],
                    }
                ]
            }
        )
        client = _FakeAsyncClient(response)
        provider = {
            "base_url": "https://text.test/v1",
            "api_key": "sk-test",
            "model": "text-model",
            "wire_api": "responses",
        }
        with patch("vision.httpx.AsyncClient", return_value=client):
            text = await vision._call_prompt_planner(provider, "图里有什么趋势？")
        self.assertEqual(text, "read the chart values")
        self.assertEqual(client.url, "https://text.test/v1/responses")
        self.assertIn("input", client.payload)


class _FakeResponse:
    def __init__(self, data, status_code=200):
        self._data = data
        self.status_code = status_code
        self.text = ""

    def json(self):
        return self._data


class _FakeAsyncClient:
    def __init__(self, response):
        self.response = response
        self.url = None
        self.payload = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, json, headers):
        self.url = url
        self.payload = json
        return self.response


if __name__ == "__main__":
    unittest.main()
