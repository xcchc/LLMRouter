# -*- coding: utf-8 -*-
"""请求统计：内存记录 + stats.jsonl 持久化"""
import datetime
import json
import re
import sys
import threading
from pathlib import Path

if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).resolve().parent  # exe 所在目录
else:
    BASE_DIR = Path(__file__).resolve().parent
STATS_FILE = BASE_DIR / "stats.jsonl"
MAX_LINES = 100000   # 文件超过该行数时裁剪
KEEP_LINES = 50000   # 裁剪后保留的行数

_lock = threading.Lock()
_records: list = []


def _normalize_model_id(row: dict) -> None:
    """历史记录里的裸模型名统一补成 ``<供应商>/<模型名>``。"""
    model = str(row.get("model") or "")
    supplier = str(row.get("supplier") or "")
    if model and supplier and not model.startswith(supplier + "/"):
        row["model"] = f"{supplier}/{model}"


def load() -> None:
    """启动时把 stats.jsonl 载入内存，并顺手裁剪过大的文件。"""
    global _records
    if not STATS_FILE.exists():
        return
    try:
        with open(STATS_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) > MAX_LINES:
            lines = lines[-KEEP_LINES:]
            with open(STATS_FILE, "w", encoding="utf-8") as f:
                f.writelines(lines)
        rows = []
        for ln in lines:
            ln = ln.strip()
            if not ln:
                continue
            try:
                row = json.loads(ln)
            except Exception:
                continue
            if isinstance(row, dict):
                _normalize_model_id(row)
                rows.append(row)
        _records = rows
    except Exception:
        pass


def clear() -> None:
    global _records
    with _lock:
        _records = []
        try:
            STATS_FILE.unlink(missing_ok=True)
        except Exception:
            pass


def record(req: dict) -> None:
    """记录一次请求。req 字段：ts/model/supplier/upstream_wire/incoming_wire/status/http_status/
    input_tokens/output_tokens/total_tokens/duration_ms/first_token_ms/reasoning_effort/error"""
    if isinstance(req, dict):
        _normalize_model_id(req)
    with _lock:
        _records.append(req)
        try:
            with open(STATS_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(req, ensure_ascii=False) + "\n")
        except Exception:
            pass


def recent(limit: int = 100, offset: int = 0) -> list:
    """取最近记录（最新在前？返回仍是时间正序的切片），offset 用于分页。"""
    with _lock:
        end = len(_records) - offset
        start = max(0, end - limit)
        return list(_records[start:end])


def count() -> int:
    with _lock:
        return len(_records)


def recent_filtered(
    limit: int = 50,
    offset: int = 0,
    hours: int = 0,
    start: float = None,
    end: float = None,
    model: str = "",
    status: str = "",
) -> tuple:
    """按模型、状态和时间范围过滤最近记录，用于"最近请求"分页。start/end 为 Unix 秒。

    ``status``：``ok`` 只看成功，``error`` 只看失败（status 不是 ok），空表示全部。
    """
    with _lock:
        rows = list(_records)
    if model:
        rows = [r for r in rows if (r.get("model") or "?") == model]
    if status == "ok":
        rows = [r for r in rows if (r.get("status") or "") == "ok"]
    elif status:
        rows = [r for r in rows if (r.get("status") or "") != "ok"]
    if hours and hours > 0:
        cutoff = time_now() - hours * 3600
        rows = [r for r in rows if (r.get("ts") or 0) >= cutoff]
    if start is not None:
        rows = [r for r in rows if (r.get("ts") or 0) >= float(start)]
    if end is not None:
        rows = [r for r in rows if (r.get("ts") or 0) <= float(end)]
    end_idx = len(rows) - offset
    start_idx = max(0, end_idx - limit)
    return len(rows), list(rows[start_idx:end_idx])


def _day_start_ts() -> float:
    now = datetime.datetime.now()
    return now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


def _num(v) -> int:
    try:
        return int(v or 0)
    except Exception:
        return 0


def _first_token_ms(r: dict):
    """首字耗时（毫秒）；没有记录时返回 None。"""
    v = r.get("first_token_ms")
    if isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0:
        return float(v)
    return None


def _gen_ms(r: dict) -> float:
    """生成耗时 = 总耗时 − 首字耗时。

    首字之后才是真正的出字阶段，用它当分母算速度更准；非流式请求
    （首字 = 总耗时）和老记录没有首字数据时退回总耗时，避免分母为 0。
    """
    dur = r.get("duration_ms")
    dur = float(dur) if isinstance(dur, (int, float)) and not isinstance(dur, bool) and dur > 0 else 0.0
    first = _first_token_ms(r)
    if first is not None and 0 <= first < dur:
        return dur - first
    return dur


def _price(v) -> float:
    """价格转浮点（保留小数，如 0.2 元）；非数字返回 0.0。"""
    try:
        if v in (None, ""):
            return 0.0
        return float(v)
    except Exception:
        return 0.0


def _merge_tokens(agg: dict, r: dict) -> None:
    agg["input"] += _num(r.get("input_tokens"))
    agg["cached"] += _num(r.get("cached_tokens"))
    agg["output"] += _num(r.get("output_tokens"))
    agg["total"] += _num(r.get("total_tokens"))


def _cost_of(r: dict, p: dict) -> float:
    """???(???) / ???? / ???????????????????"""
    if not (p.get("in") or p.get("out")):
        return 0.0
    inp = _num(r.get("input_tokens"))
    cached = min(_num(r.get("cached_tokens")), inp)
    in_price = p.get("in", 0)
    in_cached_price = p.get("in_cached")
    if in_cached_price is None:
        in_cached_price = in_price
    return (inp - cached) * in_price / 1e6 + cached * in_cached_price / 1e6 + _num(r.get("output_tokens")) * p.get("out", 0) / 1e6


def _model_sort_key(text: str) -> list:
    """模型名排序键：忽略大小写，数字段按数值比较（gpt-5.9 排在 gpt-5.10 前）。"""
    parts = re.split(r"(\d+)", str(text or "").casefold())
    return [(0, int(part)) if part.isdigit() else (1, part) for part in parts]


def summary(cfg: dict, model: str = "", hours: int = 24, start: float = None, end: float = None) -> dict:
    """汇总：总量 / 按模型 / 按供应商 / 时间曲线 / 错误数 / 平均耗时 / 缓存命中
    model 非空时只统计该模型；hours>0 时只统计最近 N 小时（0=全部）；start/end 为 Unix 秒。"""
    with _lock:
        rows = list(_records)
    all_models = sorted({r.get("model") or "?" for r in rows}, key=_model_sort_key)
    if model:
        rows = [r for r in rows if (r.get("model") or "?") == model]
    now_ts = time_now()
    day_start = _day_start_ts()
    custom_range = start is not None or end is not None
    if start is not None:
        rows = [r for r in rows if (r.get("ts") or 0) >= float(start)]
    if end is not None:
        rows = [r for r in rows if (r.get("ts") or 0) <= float(end)]
    use_daily = bool(hours and hours > 48) or not hours
    range_start_ts = None
    range_end_ts = now_ts
    if custom_range:
        start_ts = float(start) if start is not None else (min((r.get("ts") or now_ts) for r in rows) if rows else now_ts - 86400)
        end_ts = float(end) if end is not None else now_ts
        use_daily = (end_ts - start_ts) > 48 * 3600
    elif hours and hours > 0:
        range_start_ts = max(0, now_ts - hours * 3600)
        rows = [r for r in rows if (r.get("ts") or 0) >= range_start_ts]
    prices = {}
    for s in cfg.get("suppliers", []):
        name = s.get("name", "")
        prices[name] = {
            "in": _price(s.get("price_input_per_1m")),
            "out": _price(s.get("price_output_per_1m")),
            "in_cached": None if s.get("price_input_cached_per_1m") in (None, "") else _price(s.get("price_input_cached_per_1m")),
            "currency": s.get("currency", "CNY"),
        }
    model_prices = {}
    for m, v in (cfg.get("model_prices") or {}).items():
        if isinstance(v, dict):
            model_prices[m] = {
                "in": None if v.get("price_input_per_1m") in (None, "") else _price(v.get("price_input_per_1m")),
                "out": None if v.get("price_output_per_1m") in (None, "") else _price(v.get("price_output_per_1m")),
                "in_cached": None if v.get("price_input_cached_per_1m") in (None, "") else _price(v.get("price_input_cached_per_1m")),
            }
    total = {"requests": 0, "errors": 0, "failed": 0, "input": 0, "cached": 0, "output": 0, "total": 0, "cost": 0.0, "ms": 0, "first": 0.0, "first_n": 0}
    today = {"requests": 0, "errors": 0, "input": 0, "cached": 0, "output": 0, "total": 0, "ms": 0, "first": 0.0, "first_n": 0}
    by_model = {}
    by_supplier = {}
    hourly = {}
    durations = []
    for r in rows:
        ts = r.get("ts") or 0
        status = r.get("status", "")
        gen_ms = _gen_ms(r)
        first_ms = _first_token_ms(r)
        total["requests"] += 1
        if status != "ok":
            total["errors"] += 1
        _merge_tokens(total, r)
        total["ms"] += gen_ms
        if first_ms is not None:
            total["first"] += first_ms
            total["first_n"] += 1
        if ts >= day_start:
            today["requests"] += 1
            if status != "ok":
                today["errors"] += 1
            _merge_tokens(today, r)
            today["ms"] += gen_ms
            if first_ms is not None:
                today["first"] += first_ms
                today["first_n"] += 1
        m = r.get("model") or "?"
        sup = r.get("supplier") or "?"
        for key, bucket in ((m, by_model), (sup, by_supplier)):
            b = bucket.setdefault(key, {"requests": 0, "errors": 0, "input": 0, "cached": 0, "output": 0, "total": 0, "cost": 0.0, "ms": 0, "first": 0.0, "first_n": 0})
            b["requests"] += 1
            if status != "ok":
                b["errors"] += 1
            _merge_tokens(b, r)
            b["ms"] += gen_ms
            if first_ms is not None:
                b["first"] += first_ms
                b["first_n"] += 1
            p = dict(prices.get(sup) or {})
            mp = model_prices.get(m)
            if mp:
                if mp.get("in") is not None:
                    p["in"] = mp["in"]
                if mp.get("out") is not None:
                    p["out"] = mp["out"]
                if mp.get("in_cached") is not None:
                    p["in_cached"] = mp["in_cached"]
                elif mp.get("in") is not None:
                    p["in_cached"] = mp["in"]
            c = _cost_of(r, p)
            b["cost"] += c
            if key is sup:
                total["cost"] += c
        # 时间曲线（按小时或按天）
        if use_daily:
            bucket_key = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
            hour_key = bucket_key
        else:
            bucket_key = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:00")
            hour_key = bucket_key
        h = hourly.setdefault(bucket_key, {"requests": 0, "total": 0})
        h["requests"] += 1
        h["total"] += _num(r.get("total_tokens"))
        if r.get("duration_ms") is not None:
            durations.append(r["duration_ms"])
    # 生成连续的桶：<=48 小时按小时，其余按天
    buckets = []
    if custom_range:
        if use_daily:
            d0 = datetime.datetime.fromtimestamp(start_ts).replace(hour=0, minute=0, second=0, microsecond=0)
            d1 = datetime.datetime.fromtimestamp(end_ts).replace(hour=0, minute=0, second=0, microsecond=0)
            days = max(1, int((d1 - d0).total_seconds() // 86400) + 1)
            day_label = "%Y-%m-%d" if days > 31 else "%m-%d"
            for i in range(days):
                dt = d0 + datetime.timedelta(days=i)
                key = dt.strftime("%Y-%m-%d")
                b = hourly.get(key, {"requests": 0, "total": 0})
                buckets.append({"hour": dt.strftime(day_label), "requests": b["requests"], "tokens": b["total"]})
        else:
            h0 = datetime.datetime.fromtimestamp(start_ts).replace(minute=0, second=0, microsecond=0)
            h1 = datetime.datetime.fromtimestamp(end_ts).replace(minute=0, second=0, microsecond=0)
            hour_count = max(1, int((h1 - h0).total_seconds() // 3600) + 1)
            hour_label = "%H:00" if hour_count <= 24 else "%m-%d %H:00"
            for i in range(hour_count):
                dt = h0 + datetime.timedelta(hours=i)
                key = dt.strftime("%Y-%m-%d %H:00")
                b = hourly.get(key, {"requests": 0, "total": 0})
                buckets.append({"hour": dt.strftime(hour_label), "requests": b["requests"], "tokens": b["total"]})
    elif hours and hours > 0:
        if use_daily:
            d0 = datetime.datetime.fromtimestamp(range_start_ts).replace(hour=0, minute=0, second=0, microsecond=0)
            d1 = datetime.datetime.fromtimestamp(range_end_ts).replace(hour=0, minute=0, second=0, microsecond=0)
            days = max(1, int((d1 - d0).total_seconds() // 86400) + 1)
            day_label = "%Y-%m-%d" if days > 31 else "%m-%d"
            for i in range(days):
                dt = d0 + datetime.timedelta(days=i)
                key = dt.strftime("%Y-%m-%d")
                b = hourly.get(key, {"requests": 0, "total": 0})
                buckets.append({"hour": dt.strftime(day_label), "requests": b["requests"], "tokens": b["total"]})
        else:
            h0 = datetime.datetime.fromtimestamp(range_start_ts).replace(minute=0, second=0, microsecond=0)
            h1 = datetime.datetime.fromtimestamp(range_end_ts).replace(minute=0, second=0, microsecond=0)
            hour_count = max(1, int((h1 - h0).total_seconds() // 3600) + 1)
            hour_label = "%H:00" if hour_count <= 24 else "%m-%d %H:00"
            for i in range(hour_count):
                dt = h0 + datetime.timedelta(hours=i)
                key = dt.strftime("%Y-%m-%d %H:00")
                b = hourly.get(key, {"requests": 0, "total": 0})
                buckets.append({"hour": dt.strftime(hour_label), "requests": b["requests"], "tokens": b["total"]})
    else:
        # 全部：从最早一条记录所在日期到今天。
        d1 = datetime.datetime.fromtimestamp(now_ts).replace(hour=0, minute=0, second=0, microsecond=0)
        first_ts = min((r.get("ts") or now_ts) for r in rows) if rows else now_ts
        d0 = datetime.datetime.fromtimestamp(first_ts).replace(hour=0, minute=0, second=0, microsecond=0)
        total_days = max(1, min(400, int((d1 - d0).total_seconds() // 86400) + 1))
        d0 = d1 - datetime.timedelta(days=total_days - 1)
        for i in range(total_days):
            dt = d0 + datetime.timedelta(days=i)
            key = dt.strftime("%Y-%m-%d")
            b = hourly.get(key, {"requests": 0, "total": 0})
            buckets.append({"hour": dt.strftime("%m-%d"), "requests": b["requests"], "tokens": b["total"]})
    avg_duration = round(sum(durations) / len(durations), 1) if durations else 0

    def _finish(bucket: dict) -> dict:
        """补上 token 速度和平均首字。

        速度 = 输出 token 总和 ÷ 生成耗时总和（总耗时减首字），不是对每次请求的速度求平均。
        """
        ms = bucket.pop("ms", 0) or 0
        first = bucket.pop("first", 0.0) or 0.0
        first_n = bucket.pop("first_n", 0) or 0
        bucket["speed"] = round(bucket.get("output", 0) / (ms / 1000), 1) if ms > 0 else 0
        bucket["avg_first_token_ms"] = round(first / first_n, 1) if first_n else 0
        return bucket

    return {
        "total": _finish(total),
        "today": _finish(today),
        "by_model": [_finish(dict(by_model[k], name=k)) for k in sorted(by_model, key=_model_sort_key)],
        "by_supplier": [_finish(dict(by_supplier[k], name=k)) for k in sorted(by_supplier)],
        "hourly": buckets,
        "avg_duration_ms": avg_duration,
        "avg_duration_seconds": avg_duration / 1000,
        "total_records": len(rows),
        "all_models": all_models,
    }


def time_now() -> float:
    import time
    return time.time()
