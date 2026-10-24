import unittest
from unittest.mock import patch

import router


class ModelCatalogTests(unittest.IsolatedAsyncioTestCase):
    async def test_find_supplier_returns_supplier_settings_as_is(self):
        supplier = {"name": "pixel", "openai_sanitize": True, "forward_prompt_cache_key": False}
        cfg = {
            "suppliers": [supplier],
            "model_map": {"pixel/gpt-5.6-sol": "pixel"},
            "model_overrides": {
                "pixel/gpt-5.6-sol": {
                    "openai_sanitize": False,
                    "forward_prompt_cache_key": True,
                },
            },
        }

        resolved = await router.find_supplier(cfg, "pixel/gpt-5.6-sol")

        self.assertEqual(resolved["name"], "pixel")
        self.assertIs(resolved["openai_sanitize"], True)
        self.assertIs(resolved["forward_prompt_cache_key"], False)
        self.assertIsNot(resolved, supplier)
        self.assertIs(supplier["openai_sanitize"], True)
        self.assertIs(supplier["forward_prompt_cache_key"], False)

    def test_supplier_name_cannot_contain_the_public_id_separator(self):
        base = {"name": "xkiro", "base_url": "https://api.xkiro.com/v1"}

        self.assertIsNone(router.validate_supplier(base))
        self.assertIsNotNone(router.validate_supplier({**base, "name": "xki/ro"}))

    def test_multipart_model_field_is_rewritten_in_place(self):
        raw = (
            b'--b\r\nContent-Disposition: form-data; name="model"\r\n\r\n'
            b"xkiro/gpt-image-2\r\n"
            b'--b\r\nContent-Disposition: form-data; name="prompt"\r\n\r\n'
            b"a cat\r\n--b--\r\n"
        )

        rewritten = router._replace_multipart_model(raw, "xkiro/gpt-image-2", "gpt-image-2")

        self.assertEqual(router._multipart_model(rewritten), "gpt-image-2")
        self.assertIn(b"a cat", rewritten)
        # 值不匹配时原样返回，避免误改别的字段。
        self.assertEqual(router._replace_multipart_model(raw, "other", "x"), raw)

    def test_configured_model_list_excludes_upstream_only_models(self):
        cfg = {
            "suppliers": [
                {
                    "name": "xkiro",
                    "models": ["manual-model", "mapped-model"],
                },
                {
                    "name": "other",
                    "models": ["other-manual"],
                },
            ],
            "model_map": {
                "xkiro/mapped-model": "xkiro",
                "xkiro/routed-model": "xkiro",
            },
            "model_blacklist": {},
        }

        with patch.dict(
            router._model_cache,
            {"xkiro": {"ids": ["upstream-only"], "at": 0}},
            clear=True,
        ):
            models = router.configured_model_list(cfg)

        self.assertEqual(
            [model["id"] for model in models],
            ["other/other-manual", "xkiro/manual-model", "xkiro/mapped-model", "xkiro/routed-model"],
        )
        self.assertNotIn("upstream-only", [model["id"] for model in models])

    def test_configured_model_list_qualifies_duplicate_model_names(self):
        cfg = {
            "suppliers": [
                {"name": "pixel", "models": ["deepseek-v4.1-flash"]},
                {"name": "xkiro", "models": ["deepseek-v4.1-flash"]},
            ],
            "model_map": {},
            "model_blacklist": {},
        }

        models = router.configured_model_list(cfg)

        self.assertEqual(
            [model["id"] for model in models],
            ["pixel/deepseek-v4.1-flash", "xkiro/deepseek-v4.1-flash"],
        )
        self.assertEqual(
            [model["owned_by"] for model in models],
            ["pixel", "xkiro"],
        )

    def test_configured_model_list_keeps_slashes_inside_model_names(self):
        cfg = {
            "suppliers": [{"name": "xkiro", "models": ["z-ai/glm-5.3"]}],
            "model_map": {"xkiro/xiaomi/mimo-v2.6-pro": "xkiro"},
            "model_blacklist": {},
        }

        models = router.configured_model_list(cfg)

        self.assertEqual(
            [model["id"] for model in models],
            ["xkiro/xiaomi/mimo-v2.6-pro", "xkiro/z-ai/glm-5.3"],
        )

    def test_configured_model_list_is_sorted_by_model_name(self):
        cfg = {
            "suppliers": [{"name": "xkiro", "models": ["z-ai/glm-5.3", "MiniMax-M3.1-Flash"]}],
            "model_map": {
                "pixel/gpt-6-astra": "pixel",
                "pixel/gpt-5.10-x": "pixel",
                "pixel/gpt-5.9-x": "pixel",
            },
            "model_blacklist": {},
        }

        ids = [model["id"] for model in router.configured_model_list(cfg)]

        # 忽略大小写、数字段按数值比较，Agent 拉到的列表顺序稳定可预期。
        self.assertEqual(
            ids,
            [
                "pixel/gpt-5.9-x",
                "pixel/gpt-5.10-x",
                "pixel/gpt-6-astra",
                "xkiro/MiniMax-M3.1-Flash",
                "xkiro/z-ai/glm-5.3",
            ],
        )

    async def test_resolve_model_target_splits_only_the_first_slash(self):
        cfg = {
            "suppliers": [{"name": "xkiro", "models": ["z-ai/glm-5.3"]}],
            "model_map": {},
        }

        supplier, upstream = await router.resolve_model_target(cfg, "xkiro/z-ai/glm-5.3")

        self.assertEqual(supplier["name"], "xkiro")
        self.assertEqual(upstream, "z-ai/glm-5.3")

    async def test_resolve_model_target_trusts_prefix_without_a_model_list(self):
        cfg = {"suppliers": [{"name": "xkiro", "models": []}], "model_map": {}}

        with patch.dict(router._model_cache, {}, clear=True):
            supplier, upstream = await router.resolve_model_target(cfg, "xkiro/z-ai/glm-5.3")

        self.assertEqual(supplier["name"], "xkiro")
        self.assertEqual(upstream, "z-ai/glm-5.3")

    async def test_resolve_model_target_ignores_a_bare_model_name(self):
        # 旧样式（不带供应商前缀）不再兜底：客户端必须用 <供应商>/<模型名>。
        cfg = {
            "suppliers": [{"name": "xkiro", "models": ["z-ai/glm-5.3"]}],
            "model_map": {"xkiro/z-ai/glm-5.3": "xkiro"},
        }

        supplier, upstream = await router.resolve_model_target(cfg, "z-ai/glm-5.3")

        self.assertIsNone(supplier)
        self.assertEqual(upstream, "z-ai/glm-5.3")

    async def test_resolve_model_target_strips_only_the_supplier_prefix(self):
        # 供应商名恰好等于模型命名空间时（openai 渠道的 openai/gpt-5），
        # 公开名是 openai/openai/gpt-5，转发时只去掉第一个前缀。
        cfg = {
            "suppliers": [{"name": "openai", "models": ["openai/gpt-5"]}],
            "model_map": {},
        }

        supplier, upstream = await router.resolve_model_target(cfg, "openai/openai/gpt-5")

        self.assertEqual(supplier["name"], "openai")
        self.assertEqual(upstream, "openai/gpt-5")

    async def test_resolve_model_target_ignores_default_supplier(self):
        cfg = {
            "suppliers": [{"name": "pixel", "models": ["pixel-only"]}],
            "model_map": {},
            "default_supplier": "pixel",
        }

        supplier, upstream = await router.resolve_model_target(cfg, "unknown-model")

        self.assertIsNone(supplier)
        self.assertEqual(upstream, "unknown-model")

    async def test_resolve_model_target_returns_none_for_unknown_prefix(self):
        cfg = {"suppliers": [{"name": "pixel", "models": ["pixel-only"]}], "model_map": {}}

        supplier, upstream = await router.resolve_model_target(cfg, "nope/model")

        self.assertIsNone(supplier)
        self.assertEqual(upstream, "nope/model")

    def test_configured_model_list_applies_supplier_hidden_models(self):
        cfg = {
            "suppliers": [{
                "name": "xkiro",
                "models": ["visible-manual", "hidden-manual"],
            }],
            "model_map": {
                "xkiro/visible-route": "xkiro",
                "xkiro/hidden-route": "xkiro",
            },
            "model_blacklist": {
                "xkiro": ["hidden-manual", "hidden-route"],
            },
        }

        models = router.configured_model_list(cfg)

        self.assertEqual(
            [model["id"] for model in models],
            ["xkiro/visible-manual", "xkiro/visible-route"],
        )

    async def test_v1_models_uses_only_configured_models(self):
        cfg = {
            "suppliers": [{
                "name": "xkiro",
                "models": ["added-manually"],
            }],
            "model_map": {
                "xkiro/added-by-route": "xkiro",
            },
            "model_blacklist": {},
        }

        with patch.object(router, "load_config", return_value=cfg):
            result = await router.get_models()

        self.assertEqual(
            [model["id"] for model in result["data"]],
            ["xkiro/added-by-route", "xkiro/added-manually"],
        )

    async def test_dashboard_merged_catalog_still_includes_upstream_models(self):
        cfg = {
            "suppliers": [{"name": "xkiro", "models": []}],
            "model_map": {},
            "model_blacklist": {},
        }

        async def fake_supplier_model_ids(_supplier, _cfg):
            return ["upstream-model"]

        with patch.object(router, "supplier_model_ids", new=fake_supplier_model_ids):
            models = await router.merged_model_list(cfg)

        self.assertEqual([model["id"] for model in models], ["upstream-model"])

    async def test_v1_models_publishes_the_configured_output_limit(self):
        cfg = {
            "suppliers": [{
                "name": "pixel",
                "models": ["deepseek-example"],
            }],
            "model_map": {"pixel/routed-example": "pixel"},
            "model_blacklist": {},
            "model_max_output_tokens": {
                "pixel/deepseek-example": 65536,
                "pixel/routed-example": 32768,
            },
        }

        with patch.object(router, "load_config", return_value=cfg):
            result = await router.get_models()

        by_id = {model["id"]: model for model in result["data"]}
        self.assertEqual(by_id["pixel/deepseek-example"]["max_output_tokens"], 65536)
        self.assertEqual(by_id["pixel/routed-example"]["max_output_tokens"], 32768)

    async def test_v1_models_omits_the_output_limit_when_not_configured(self):
        cfg = {
            "suppliers": [{"name": "pixel", "models": ["plain-example"]}],
            "model_map": {},
            "model_blacklist": {},
        }

        with patch.object(router, "load_config", return_value=cfg):
            result = await router.get_models()

        self.assertNotIn("max_output_tokens", result["data"][0])

    def test_normalized_config_keeps_the_output_limit_map(self):
        cfg = {
            "suppliers": [],
            "model_map": {},
            "model_max_output_tokens": {"deepseek-example": 65536},
        }

        result = router.normalized_config(cfg)

        self.assertEqual(result["model_max_output_tokens"], {"deepseek-example": 65536})


class DashboardModelRouteTests(unittest.TestCase):
    def test_model_route_uses_supplier_select_and_upstream_fetch(self):
        dashboard = (router.BASE_DIR / "dashboard.html").read_text(encoding="utf-8")

        self.assertIn('<select id="newModelName"', dashboard)
        self.assertIn("refreshModelRouteOptions($('#newModelSupplier').value, '')", dashboard)
        self.assertIn("/api/suppliers/' + encodeURIComponent(supplierName) + '/fetch-models'", dashboard)
        self.assertNotIn('<input id="newModelName"', dashboard)

    def test_model_route_supports_query_delete_drag_sort_and_top_level_pricing(self):
        dashboard = (router.BASE_DIR / "dashboard.html").read_text(encoding="utf-8")

        self.assertIn("/api/model_map?model=' + encodeURIComponent(model)", dashboard)
        self.assertIn("function qualifiedModelId(", dashboard)
        self.assertIn('data-model="${esc(k)}"', dashboard)
        self.assertIn('draggable="true"', dashboard)
        self.assertIn("setupModelMapDrag()", dashboard)
        self.assertIn("data-price-model", dashboard)
        self.assertNotIn('id="tblMerged"', dashboard)
        self.assertNotIn("function loadMerged()", dashboard)

    def test_model_route_uses_page_scroll_without_wrapping_card(self):
        dashboard = (router.BASE_DIR / "dashboard.html").read_text(encoding="utf-8")
        start = dashboard.index('<section id="tab-models"')
        end = dashboard.index('<section id="tab-settings"', start)
        models_tab = dashboard[start:end]

        self.assertIn('<section id="tab-models" class="tabpane page-pane">', models_tab)
        self.assertIn('<div class="model-route-table"><table id="tblModelMap">', models_tab)
        self.assertNotIn('<div class="card">', models_tab)
        self.assertNotIn('<div class="tblwrap"><table id="tblModelMap">', models_tab)


class ModelRouteMutationTests(unittest.IsolatedAsyncioTestCase):
    class JsonRequest:
        def __init__(self, data):
            self.data = data

        async def json(self):
            return self.data

    async def test_put_model_map_drops_model_overrides(self):
        cfg = {"suppliers": [{"name": "pixel"}], "model_map": {}}
        request = self.JsonRequest({
            "model_map": {"pixel/gpt-5.6-sol": "pixel"},
            "model_overrides": {
                "pixel/gpt-5.6-sol": {
                    "openai_sanitize": False,
                    "forward_prompt_cache_key": True,
                },
            },
            "default_supplier": "",
        })

        with (
            patch.object(router, "load_config", return_value=cfg),
            patch.object(router, "save_config") as save_config,
        ):
            result = await router.api_put_model_map(request)

        self.assertEqual(result, {"ok": True})
        self.assertNotIn("model_overrides", cfg)
        save_config.assert_called_once_with(cfg)

    async def test_add_model_lets_two_suppliers_expose_the_same_model_name(self):
        # 同名模型挂在两个渠道：公开名带供应商前缀，所以不会互相覆盖。
        cfg = {
            "suppliers": [{"name": "pixel"}, {"name": "xkiro"}],
            "model_map": {"pixel/deepseek-v4.1-flash": "pixel"},
        }
        request = self.JsonRequest({
            "model": "deepseek-v4.1-flash",
            "supplier": "xkiro",
            "mode": "map",
        })

        with (
            patch.object(router, "load_config", return_value=cfg),
            patch.object(router, "save_config") as save_config,
        ):
            response = await router.api_add_model(request)

        self.assertEqual(response, {"ok": True})
        self.assertEqual(
            cfg["model_map"],
            {
                "pixel/deepseek-v4.1-flash": "pixel",
                "xkiro/deepseek-v4.1-flash": "xkiro",
            },
        )
        save_config.assert_called_once_with(cfg)
        self.assertEqual(
            [m["id"] for m in router.configured_model_list(cfg)],
            ["pixel/deepseek-v4.1-flash", "xkiro/deepseek-v4.1-flash"],
        )

    async def test_add_model_manual_mode_allows_duplicate_names(self):
        cfg = {
            "suppliers": [
                {"name": "pixel", "models": ["deepseek-v4.1-flash"]},
                {"name": "xkiro", "models": []},
            ],
            "model_map": {},
        }
        request = self.JsonRequest({
            "model": "deepseek-v4.1-flash",
            "supplier": "xkiro",
            "mode": "manual",
        })

        with (
            patch.object(router, "load_config", return_value=cfg),
            patch.object(router, "save_config"),
        ):
            response = await router.api_add_model(request)

        self.assertEqual(response, {"ok": True})
        self.assertEqual(cfg["suppliers"][1]["models"], ["deepseek-v4.1-flash"])
        self.assertEqual(
            [m["id"] for m in router.configured_model_list(cfg)],
            ["pixel/deepseek-v4.1-flash", "xkiro/deepseek-v4.1-flash"],
        )

    async def test_delete_model_map_keeps_slash_in_model_id(self):
        cfg = {
            "model_map": {"openai/gpt-5.6-sol": "pixel"},
            "model_overrides": {"openai/gpt-5.6-sol": {"openai_sanitize": False}},
        }

        with (
            patch.object(router, "load_config", return_value=cfg),
            patch.object(router, "save_config") as save_config,
        ):
            result = await router.api_delete_model_map("openai/gpt-5.6-sol")

        self.assertEqual(result, {"ok": True})
        self.assertEqual(cfg["model_map"], {})
        self.assertNotIn("model_overrides", cfg)
        save_config.assert_called_once_with(cfg)

    async def test_delete_model_price_keeps_slash_in_model_id(self):
        cfg = {"model_prices": {"openai/gpt-5.6-sol": {"price_output_per_1m": 1}}}

        with (
            patch.object(router, "load_config", return_value=cfg),
            patch.object(router, "save_config") as save_config,
        ):
            result = await router.api_delete_model_price("openai/gpt-5.6-sol")

        self.assertEqual(result, {"ok": True})
        self.assertEqual(cfg["model_prices"], {})
        save_config.assert_called_once_with(cfg)


class ConfigMigrationTests(unittest.TestCase):
    def test_migrates_bare_model_ids_to_the_public_style(self):
        cfg = {
            "suppliers": [{"name": "pixel", "models": []}, {"name": "xkiro", "models": []}],
            "model_map": {"deepseek-v4.1-flash": "pixel", "z-ai/glm-5.3": "xkiro"},
            "model_prices": {"deepseek-v4.1-flash": {"price_output_per_1m": 1}},
            "model_max_output_tokens": {"deepseek-v4.1-flash": 65536},
            "agent_profiles": {"dsh": {"default_model": "z-ai/glm-5.3"}},
        }

        changed = router.migrate_model_ids(cfg)

        self.assertTrue(changed)
        self.assertEqual(
            cfg["model_map"],
            {"pixel/deepseek-v4.1-flash": "pixel", "xkiro/z-ai/glm-5.3": "xkiro"},
        )
        self.assertEqual(list(cfg["model_prices"]), ["pixel/deepseek-v4.1-flash"])
        self.assertEqual(list(cfg["model_max_output_tokens"]), ["pixel/deepseek-v4.1-flash"])
        self.assertEqual(cfg["agent_profiles"]["dsh"]["default_model"], "xkiro/z-ai/glm-5.3")

    def test_migration_is_idempotent_and_keeps_model_names_starting_with_the_supplier(self):
        cfg = {
            "suppliers": [
                {"name": "openai", "models": ["openai/gpt-5"]},
                {"name": "xkiro", "models": []},
            ],
            "model_map": {"openai/openai/gpt-5": "openai", "xkiro/z-ai/glm-5.3": "xkiro"},
            "agent_profiles": {"dsh": {"default_model": "xkiro/z-ai/glm-5.3"}},
        }

        self.assertFalse(router.migrate_model_ids(cfg))
        self.assertEqual(
            cfg["model_map"],
            {"openai/openai/gpt-5": "openai", "xkiro/z-ai/glm-5.3": "xkiro"},
        )
        self.assertEqual(cfg["agent_profiles"]["dsh"]["default_model"], "xkiro/z-ai/glm-5.3")

    def test_migration_leaves_unresolvable_names_alone(self):
        cfg = {"suppliers": [], "model_prices": {"mystery": {"price_output_per_1m": 1}}}

        self.assertFalse(router.migrate_model_ids(cfg))
        self.assertEqual(list(cfg["model_prices"]), ["mystery"])


class DashboardOverviewTests(unittest.TestCase):
    def test_recent_requests_show_first_token_latency_and_status_filters(self):
        dashboard = (router.BASE_DIR / "dashboard.html").read_text(encoding="utf-8")

        self.assertIn('id="chkOk"', dashboard)
        self.assertIn('id="chkErr"', dashboard)
        self.assertIn("<th>首字 (s)</th>", dashboard)
        self.assertIn("r.first_token_ms", dashboard)
        self.assertNotIn("推理 Tokens", dashboard)
        self.assertNotIn("r.reasoning_tokens", dashboard)

    def test_token_speed_appears_in_cards_recent_and_breakdowns(self):
        dashboard = (router.BASE_DIR / "dashboard.html").read_text(encoding="utf-8")

        self.assertIn("<th>速度 (t/s)</th>", dashboard)
        self.assertIn("fmtSpeed(r.output_tokens, r.duration_ms, r.first_token_ms)", dashboard)
        self.assertIn("fmtSpeedAgg(s.total.speed)", dashboard)
        self.assertIn("fmtSpeedAgg(m.speed)", dashboard)
        # 平均首字：顶部卡片 + 两个统计表
        self.assertIn("avg_first_token_ms", dashboard)
        self.assertIn("fmtFirst(m.avg_first_token_ms)", dashboard)
        self.assertIn("<th>输出</th><th>首字 (s)</th><th>速度</th><th>总量</th><th>费用</th>", dashboard)

    def test_time_filter_defaults_to_twelve_hours_and_supports_all_time(self):
        dashboard = (router.BASE_DIR / "dashboard.html").read_text(encoding="utf-8")

        self.assertIn('id="nHours" min="1" max="720" step="1" value="12"', dashboard)
        self.assertIn("let timeHours = 12;", dashboard)
        self.assertIn("timeHours = v === '0' ? 0 :", dashboard)

    def test_model_picker_and_route_table_sort_by_model_name(self):
        dashboard = (router.BASE_DIR / "dashboard.html").read_text(encoding="utf-8")

        self.assertIn("function cmpModelName(a, b){", dashboard)
        self.assertIn("unique.sort(cmpModelName);", dashboard)
        self.assertIn("Object.keys(mm).sort(cmpModelName)", dashboard)

    def test_model_route_has_no_default_supplier_fallback(self):
        dashboard = (router.BASE_DIR / "dashboard.html").read_text(encoding="utf-8")

        self.assertNotIn("defaultSupplier", dashboard)
        self.assertNotIn("default_supplier", dashboard)


if __name__ == "__main__":
    unittest.main()
