import unittest
from unittest.mock import patch

import router


class ModelCatalogTests(unittest.IsolatedAsyncioTestCase):
    async def test_find_supplier_applies_model_level_overrides(self):
        supplier = {"name": "pixel", "openai_sanitize": True, "forward_prompt_cache_key": False}
        cfg = {
            "suppliers": [supplier],
            "model_map": {"gpt-5.6-sol": "pixel"},
            "model_overrides": {
                "gpt-5.6-sol": {
                    "openai_sanitize": False,
                    "forward_prompt_cache_key": True,
                },
            },
        }

        resolved = await router.find_supplier(cfg, "gpt-5.6-sol")

        self.assertEqual(resolved["name"], "pixel")
        self.assertIs(resolved["openai_sanitize"], False)
        self.assertIs(resolved["forward_prompt_cache_key"], True)
        self.assertIs(supplier["openai_sanitize"], True)
        self.assertIs(supplier["forward_prompt_cache_key"], False)

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
                "mapped-model": "xkiro",
                "routed-model": "xkiro",
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
            ["mapped-model", "routed-model", "manual-model", "other-manual"],
        )
        self.assertNotIn("upstream-only", [model["id"] for model in models])

    def test_configured_model_list_applies_supplier_hidden_models(self):
        cfg = {
            "suppliers": [{
                "name": "xkiro",
                "models": ["visible-manual", "hidden-manual"],
            }],
            "model_map": {
                "visible-route": "xkiro",
                "hidden-route": "xkiro",
            },
            "model_blacklist": {
                "xkiro": ["hidden-manual", "hidden-route"],
            },
        }

        models = router.configured_model_list(cfg)

        self.assertEqual(
            [model["id"] for model in models],
            ["visible-route", "visible-manual"],
        )

    async def test_v1_models_uses_only_configured_models(self):
        cfg = {
            "suppliers": [{
                "name": "xkiro",
                "models": ["added-manually"],
            }],
            "model_map": {
                "added-by-route": "xkiro",
            },
            "model_blacklist": {},
        }

        with patch.object(router, "load_config", return_value=cfg):
            result = await router.get_models()

        self.assertEqual(
            [model["id"] for model in result["data"]],
            ["added-by-route", "added-manually"],
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

    async def test_put_model_map_saves_model_overrides(self):
        cfg = {"suppliers": [{"name": "pixel"}], "model_map": {}}
        request = self.JsonRequest({
            "model_map": {"gpt-5.6-sol": "pixel"},
            "model_overrides": {
                "gpt-5.6-sol": {
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
        self.assertEqual(
            cfg["model_overrides"]["gpt-5.6-sol"],
            {"openai_sanitize": False, "forward_prompt_cache_key": True},
        )
        save_config.assert_called_once_with(cfg)

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
        self.assertEqual(cfg.get("model_overrides"), {})
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


if __name__ == "__main__":
    unittest.main()
