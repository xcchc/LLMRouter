import unittest
from unittest.mock import patch

import httpx

import router


class FakeImageClient:
    requests = []
    closed = False

    def __init__(self, *args, **kwargs):
        type(self).closed = False

    def build_request(self, method, url, **kwargs):
        request = httpx.Request(method, url, **kwargs)
        type(self).requests.append(request)
        return request

    async def send(self, request, stream=False):
        return httpx.Response(
            200,
            request=request,
            headers={"content-type": "application/json"},
            content=b'{"data": [{"b64_json": "AAAA"}]}',
        )

    async def aclose(self):
        type(self).closed = True


class ImageFakeRequest:
    def __init__(self, body=None, raw=b"", content_type="application/json"):
        self.json_body = body
        self.raw = raw
        self.headers = {"content-type": content_type}

    async def json(self):
        return self.json_body

    async def body(self):
        return self.raw


class ImageProxyTests(unittest.IsolatedAsyncioTestCase):
    async def _relay(self, request, path, supplier):
        FakeImageClient.requests = []
        records = []

        async def find_supplier(_cfg, _model):
            return supplier

        with (
            patch.object(router, "load_config", return_value={}),
            patch.object(router, "find_supplier", new=find_supplier),
            patch.object(router.httpx, "AsyncClient", FakeImageClient),
            patch.object(router.stats, "record", side_effect=records.append),
        ):
            response = await router._relay_image(request, path)
        return response, records

    async def test_generations_uses_mapped_supplier_and_v1_path(self):
        supplier = {
            "name": "image-provider",
            "base_url": "https://image.example/v1",
            "api_key": "test-key",
        }
        response, records = await self._relay(
            ImageFakeRequest({"model": "gpt-image-2", "prompt": "red cube"}),
            "images/generations",
            supplier,
        )

        sent = FakeImageClient.requests[0]
        self.assertEqual(str(sent.url), "https://image.example/v1/images/generations")
        self.assertEqual(sent.headers["authorization"], "Bearer test-key")
        self.assertEqual(b'"red cube"' in sent.content, True)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(records[0]["model"], "image-provider/gpt-image-2")
        self.assertEqual(records[0]["supplier"], "image-provider")
        self.assertTrue(FakeImageClient.closed)

    async def test_edits_preserves_multipart_body_and_adds_v1_for_domain_base(self):
        raw = (
            b"--b1\r\n"
            b'Content-Disposition: form-data; name="model"\r\n\r\n'
            b"gpt-image-2\r\n"
            b"--b1--\r\n"
        )
        supplier = {
            "name": "image-provider",
            "base_url": "https://image.example",
            "api_key": "test-key",
        }
        response, records = await self._relay(
            ImageFakeRequest(raw=raw, content_type="multipart/form-data; boundary=b1"),
            "images/edits",
            supplier,
        )

        sent = FakeImageClient.requests[0]
        self.assertEqual(str(sent.url), "https://image.example/v1/images/edits")
        self.assertEqual(sent.content, raw)
        self.assertIn("multipart/form-data", sent.headers["content-type"])
        self.assertEqual(response.status_code, 200)
        self.assertEqual(records[0]["model"], "image-provider/gpt-image-2")

    async def test_unknown_model_returns_404(self):
        async def find_supplier(_cfg, _model):
            return None

        with (
            patch.object(router, "load_config", return_value={}),
            patch.object(router, "find_supplier", new=find_supplier),
        ):
            response = await router._relay_image(
                ImageFakeRequest({"model": "unknown-image"}),
                "images/generations",
            )

        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
