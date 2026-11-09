"""上游"内容到达前失败"的自动重发回归测试。

真实故障形态：上游先回 200，再在事件流里塞一个 error 帧（例如中转网关的
"Upstream service temporarily unavailable"），或者在吐出任何内容前断流。
以前这类失败会原样透传给客户端，客户端只能看到一次中断；现在路由器会在
任何内容到达客户端之前重发整条请求。
"""

import unittest
from unittest.mock import patch

import httpx

import router

ERROR_FRAME = b'data: {"error":{"message":"Upstream service temporarily unavailable"}}\n\n'
CONTENT_FRAME = b'data: {"choices":[{"index":0,"delta":{"content":"ok"}}]}\n\n'
ROLE_FRAME = b'data: {"choices":[{"index":0,"delta":{"role":"assistant"}}]}\n\n'
TAIL_FRAMES = (
    b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
    b'"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}}\n\n'
    b"data: [DONE]\n\n"
)


class ScriptedStream(httpx.AsyncByteStream):
    def __init__(self, chunks=(), error=None):
        if isinstance(chunks, (bytes, bytearray)):
            chunks = (bytes(chunks),)
        self.chunks = chunks
        self.error = error

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk
        if self.error is not None:
            raise self.error


class ScriptedClient:
    """按脚本依次返回上游响应；脚本用尽后重复最后一项。

    每一项可以是字节块元组，也可以是 ``(字节块元组, 迭代中抛出的异常)``。
    """

    script = []
    calls = 0
    requests = []

    def __init__(self, *args, **kwargs):
        pass

    def build_request(self, method, url, **kwargs):
        return httpx.Request(method, url, **kwargs)

    async def send(self, request, stream=False):
        index = min(type(self).calls, len(type(self).script) - 1)
        type(self).calls += 1
        type(self).requests.append(request)
        step = type(self).script[index]
        if isinstance(step, Exception):
            raise step
        if (
            isinstance(step, tuple)
            and len(step) == 2
            and (step[1] is None or isinstance(step[1], BaseException))
        ):
            chunks, error = step
        else:
            chunks, error = step, None
        return httpx.Response(
            200,
            request=request,
            headers={"content-type": "text/event-stream"},
            stream=ScriptedStream(chunks, error),
        )

    async def aclose(self):
        pass


class FakeRequest:
    headers = {}

    def __init__(self, body):
        self.body = body

    async def json(self):
        return self.body


class EarlyStreamRetryTests(unittest.IsolatedAsyncioTestCase):
    async def _run(
        self,
        script,
        supplier_extra=None,
        incoming_wire="chat",
        upstream_wire="chat",
        request_body=None,
    ):
        supplier = {
            "name": "pixel",
            "base_url": "https://api.pixel.test",
            "api_key": "test-key",
            "wire_api": upstream_wire,
        }
        if supplier_extra:
            supplier.update(supplier_extra)

        async def find_supplier(_cfg, _model):
            return dict(supplier)

        records = []
        ScriptedClient.script = list(script)
        ScriptedClient.calls = 0
        ScriptedClient.requests = []
        body = request_body or {
            "model": "pixel/deepseek-v4.1-flash",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }
        with (
            patch.object(router, "load_config", return_value={}),
            patch.object(router, "find_supplier", new=find_supplier),
            patch.object(router.httpx, "AsyncClient", ScriptedClient),
            patch.object(router.stats, "record", side_effect=records.append),
            # 重发退避在测试里没有意义，别让它拖慢用例。
            patch.object(router, "EARLY_STREAM_RETRY_BACKOFF", 0.0),
        ):
            response = await router.relay(FakeRequest(body), incoming_wire)
            payload = b"".join([chunk async for chunk in response.body_iterator])
        return payload, records, ScriptedClient.calls

    async def test_error_frame_before_content_is_retried_and_hidden(self):
        payload, records, calls = await self._run([
            (ERROR_FRAME,),
            (CONTENT_FRAME + TAIL_FRAMES,),
        ])

        self.assertEqual(calls, 2)
        # 客户端只看到第二次尝试的正常内容，第一段的错误帧不会漏出去。
        self.assertIn(b'"content":"ok"', payload)
        self.assertNotIn(b"temporarily unavailable", payload)
        self.assertEqual(records[0]["status"], "ok")
        self.assertEqual(records[0]["retry_count"], 1)
        self.assertEqual(records[0]["output_tokens"], 2)
        self.assertEqual(records[0]["total_tokens"], 12)

    async def test_error_frame_is_surfaced_after_retry_budget(self):
        payload, records, calls = await self._run([(ERROR_FRAME,)])

        self.assertEqual(calls, router.EARLY_STREAM_MAX_RETRIES + 1)
        self.assertIn(b"temporarily unavailable", payload)
        self.assertEqual(records[0]["status"], "error")
        self.assertEqual(records[0]["retry_count"], router.EARLY_STREAM_MAX_RETRIES)
        self.assertIn("Upstream service temporarily unavailable", records[0]["error"])

    async def test_transport_error_before_content_is_retried(self):
        payload, records, calls = await self._run([
            (
                (b"",),
                httpx.RemoteProtocolError(
                    "peer closed connection without sending complete message body "
                    "(incomplete chunked read)"
                ),
            ),
            (CONTENT_FRAME + TAIL_FRAMES,),
        ])

        self.assertEqual(calls, 2)
        self.assertIn(b'"content":"ok"', payload)
        self.assertNotIn(b"upstream_stream_error", payload)
        self.assertEqual(records[0]["status"], "ok")
        self.assertEqual(records[0]["retry_count"], 1)

    async def test_normal_stream_is_forwarded_without_retry(self):
        payload, records, calls = await self._run([(CONTENT_FRAME + TAIL_FRAMES,)])

        self.assertEqual(calls, 1)
        self.assertIn(b'"content":"ok"', payload)
        self.assertEqual(records[0]["retry_count"], 0)

    async def test_role_preamble_is_replayed_to_the_client(self):
        payload, records, calls = await self._run([
            (ROLE_FRAME + CONTENT_FRAME + TAIL_FRAMES,),
        ])

        self.assertEqual(calls, 1)
        # 探测阶段缓存的开场帧必须原样补回去，不能丢。
        self.assertIn(b'"role":"assistant"', payload)
        self.assertIn(b'"content":"ok"', payload)

    async def test_empty_done_stream_is_not_retried(self):
        payload, records, calls = await self._run([(b"data: [DONE]\n\n",)])

        # 合法的空响应不做重发，避免对上游反复施压。
        self.assertEqual(calls, 1)
        self.assertIn(b"[DONE]", payload)

    async def test_supplier_can_disable_early_retries(self):
        payload, records, calls = await self._run(
            [(ERROR_FRAME,)],
            supplier_extra={"early_stream_retries": 0},
        )

        self.assertEqual(calls, 1)
        self.assertIn(b"temporarily unavailable", payload)
        self.assertEqual(records[0]["retry_count"], 0)

    async def test_responses_wire_early_failure_is_retried(self):
        failed = (
            b'data: {"type":"response.failed","response":{"status":"failed",'
            b'"error":{"message":"upstream blew up"}}}\n\n'
        )
        completed = (
            b'data: {"type":"response.output_text.delta","delta":"ok"}\n\n'
            b'data: {"type":"response.completed","response":{"status":"completed",'
            b'"usage":{"input_tokens":5,"output_tokens":1,"total_tokens":6}}}\n\n'
        )
        payload, records, calls = await self._run(
            [(failed,), (completed,)],
            incoming_wire="responses",
            upstream_wire="responses",
            request_body={
                "model": "pixel/deepseek-v4.1-flash",
                "input": "hi",
                "stream": True,
            },
        )

        self.assertEqual(calls, 2)
        self.assertIn(b"response.completed", payload)
        self.assertNotIn(b"upstream blew up", payload)
        self.assertEqual(records[0]["status"], "ok")
        self.assertEqual(records[0]["retry_count"], 1)


if __name__ == "__main__":
    unittest.main()
