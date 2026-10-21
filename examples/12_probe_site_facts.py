#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
12_probe_site_facts.py -- 站点客观事实探测器

用途
----
C3 命题⑤ 构念效度验证的数据采集端。

🔴 两条设计红线
--------------
1. **本脚本只采「可自动验证的二值事实」**（文件存不存在、robots 里写没写、
   HTML 里有没有某个标签）。**它不做任何主观判断、不打分。**
2. **本脚本的输出不得同时充当「agent_readiness 的输入」与「验证 agent_readiness
   的判据」**。两者用同一批事实 = 第二次同义反复。
   正确用法见 `11_construct_validity.py` 与 `the construct-validity plan`。

礼貌与边界
----------
- 每个域内串行，域间并发上限 4；同域请求间隔 >= 1.0s
- 自报 UA，不伪装浏览器
- 只取 HEAD/浅层 GET，不跟重定向到登录墙，不做任何认证
- 单请求超时 8s，整域预算 45s，失败记 null（**绝不记 False** —— 不可达 != 不存在）

用法
----
    python examples/12_probe_site_facts.py --domains domains.txt --out path/to/site_facts.json
    python examples/12_probe_site_facts.py --selftest
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
from html.parser import HTMLParser

UA = "GEO-construct-validity-probe/1.0 (academic measurement study; contact: local user)"

TIMEOUT = 8.0
PER_DOMAIN_BUDGET = 45.0
MIN_INTERVAL = 1.0
MAX_WORKERS = 4

#: 常见 MCP 声明端点。注意：这些路径**只用于记录事实**，
#: 不代表「有 MCP 就一定 agent 可调用」—— 那需要真实 agent 实验。
MCP_PATHS = (
    "/.well-known/mcp",
    "/.well-known/mcp.json",
    "/mcp.json",
    "/.mcp.json",
    "/llms.txt",
    "/llms-full.txt",
)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

_ctx = ssl.create_default_context()
_ctx.check_hostname = False
_ctx.verify_mode = ssl.CERT_NONE  # 探测期不因证书问题中断（记录在结果里）


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """禁用自动重定向。

    ⚠️ 2026-10-05 修的 bug：urllib 默认跟随重定向，于是
    `fetch("http://x")` 拿到的是**最终** 200 而不是原始 301，
    导致 `redirects_http_to_https` 恒为 False（54 个域名全报 0，纯属假数据）。
    要判断「是否强制 HTTPS」必须看**第一跳**的状态码。
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_opener = urllib.request.build_opener(_NoRedirect, urllib.request.HTTPSHandler(context=_ctx))


def fetch_raw(url: str, method: str = "GET", timeout: float = TIMEOUT):
    """同 fetch()，但**不跟随重定向**。用于判断第一跳状态码。"""
    req = urllib.request.Request(url, method=method, headers={
        "User-Agent": UA, "Accept": "*/*", "Accept-Language": "en;q=0.9",
    })
    t0 = time.time()
    try:
        with _opener.open(req, timeout=timeout) as r:
            return r.status, None, r.headers.get("Location"), time.time() - t0, None
    except urllib.error.HTTPError as e:
        return e.code, None, e.headers.get("Location") if e.headers else None, \
            time.time() - t0, f"HTTP {e.code}"
    except Exception as e:  # noqa: BLE001
        return None, None, None, time.time() - t0, type(e).__name__


def fetch(url: str, method: str = "GET", timeout: float = TIMEOUT):
    """返回 (status|None, body_bytes|None, content_type|None, elapsed_sec, error|None)"""
    req = urllib.request.Request(url, method=method, headers={
        "User-Agent": UA,
        "Accept": "*/*",
        "Accept-Language": "en;q=0.9",
    })
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_ctx) as r:
            body = r.read(200_000) if method == "GET" else None
            return r.status, body, r.headers.get("Content-Type"), time.time() - t0, None
    except urllib.error.HTTPError as e:
        return e.code, None, None, time.time() - t0, f"HTTP {e.code}"
    except Exception as e:  # noqa: BLE001
        return None, None, None, time.time() - t0, type(e).__name__


def exists(url: str) -> dict:
    """探一个 URL 是否存在。区分「不存在」与「探不到」—— 这是本脚本的核心纪律。"""
    st, body, ct, el, err = fetch(url, "GET")
    out = {
        "url": url, "status": st, "content_type": ct,
        "elapsed_sec": round(el, 3), "error": err,
    }
    if st == 200 and body is not None:
        out["present"] = True
        out["bytes"] = len(body)
    elif st is not None and 200 <= st < 300:
        out["present"] = True
    elif st in (401, 403, 405):
        # 有响应但不给看 —— 记为 unknown_present，不能记 False
        out["present"] = None
        out["note"] = "auth/challenge response — 存在但不可读"
    elif st is not None and 400 <= st < 500:
        out["present"] = False
    else:
        out["present"] = None      # 网络失败/超时 -> 不可判定
        out["note"] = "unreachable — NOT the same as absent"
    return out


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------

class _HeadParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.lang = None
        self.h1 = 0
        self.has_jsonld = False
        self.has_og = False
        self.has_viewport = False
        self.has_canonical = False
        self.title = None
        self.links = []
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "html":
            self.lang = a.get("lang")
        elif tag == "h1":
            self.h1 += 1
        elif tag == "title":
            self._in_title = True
        elif tag == "link":
            rel = (a.get("rel") or "").lower()
            if "canonical" in rel:
                self.has_canonical = True
        elif tag == "meta":
            prop = (a.get("property") or a.get("name") or "").lower()
            if prop == "og:title" or prop.startswith("og:"):
                self.has_og = True
            if prop == "viewport":
                self.has_viewport = True
        elif tag == "script":
            t = (a.get("type") or "").lower()
            if t == "application/ld+json":
                self.has_jsonld = True

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title and self.title is None:
            s = data.strip()
            if s:
                self.title = s[:120]


def parse_html(body: bytes) -> dict:
    try:
        html = body.decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return {}
    p = _HeadParser()
    try:
        p.feed(html)
    except Exception:  # noqa: BLE001
        pass
    return {
        "lang_declared": p.lang,
        "h1_count": p.h1,
        "has_jsonld": p.has_jsonld,
        "has_opengraph": p.has_og,
        "has_viewport": p.has_viewport,
        "has_canonical": p.has_canonical,
        "title_len": len(p.title) if p.title else 0,
        "html_bytes": len(body),
    }


ROBOT_AGENTS = ("gptbot", "claudebot", "ccbot", "perplexitybot", "google-extended")


def parse_robots(body: bytes) -> dict:
    """只提取「有没有点名放行某个已知 AI bot」这一条客观事实。"""
    try:
        txt = body.decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return {}
    txt_l = txt.lower()
    out = {"robots_bytes": len(body)}
    named = {}
    for bot in ROBOT_AGENTS:
        # 形如 "User-agent: GPTBot" 的行
        named[bot] = bool(re.search(rf"user-agent\s*:\s*{re.escape(bot)}", txt_l))
    out["mentions_ai_agents"] = named
    out["n_ai_agents_named"] = sum(1 for v in named.values() if v)
    out["has_sitemap_line"] = "sitemap:" in txt_l
    return out


def llms_txt_shape(body: bytes) -> dict:
    try:
        txt = body.decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return {}
    heads = re.findall(r"^#{1,3}\s+(.+)$", txt, flags=re.MULTILINE)
    links = re.findall(r"https?://[^\s\)\]]+", txt)
    return {
        "llms_txt_bytes": len(body),
        "llms_txt_n_sections": len(heads),
        "llms_txt_n_links": len(links),
        "llms_txt_mentions_mcp": bool(re.search(r"\bmcp\b", txt, re.I)),
    }


# ---------------------------------------------------------------------------
# 单域探测
# ---------------------------------------------------------------------------

def probe_domain(domain: str, scheme: str = "https") -> dict:
    t0 = time.time()
    base = f"{scheme}://{domain}"
    out: dict = {
        "domain": domain,
        "probed_at": time.time(),
        "ua": UA,
        "facts": {},
        "unreachable": [],
    }
    f = out["facts"]
    last = [0.0]

    def pace():
        dt = time.time() - last[0]
        if dt < MIN_INTERVAL:
            time.sleep(MIN_INTERVAL - dt)
        last[0] = time.time()

    def budget_left():
        return PER_DOMAIN_BUDGET - (time.time() - t0)

    # 1) 首页
    st, body, ct, el, err = None, None, None, None, None
    if budget_left() > 0:
        pace()
        st, body, ct, el, err = fetch(base, "GET")
    f["homepage"] = {
        "status": st, "content_type": ct, "elapsed_sec": round(el, 3) if el else None,
        "error": err,
    }
    if st == 200 and body:
        f["homepage"].update(parse_html(body))
    elif st is None:
        out["unreachable"].append("homepage")

    # 2) robots.txt
    if budget_left() > 0:
        pace()
        rst, rbody, _, _, rerr = fetch(base + "/robots.txt", "GET")
        f["robots"] = {"status": rst, "error": rerr}
        if rst == 200 and rbody:
            f["robots"].update(parse_robots(rbody))
        elif rst is None:
            out["unreachable"].append("robots.txt")

    # 3) 端点
    for path in MCP_PATHS:
        if budget_left() <= 0:
            out["unreachable"].append(path)
            continue
        pace()
        r = exists(base + path)
        key = path.strip("/").replace("/", "_").replace(".", "_")
        f["endpoint_" + key] = r
        if r["present"] is None:
            out["unreachable"].append(path)
        if path in ("/llms.txt", "/llms-full.txt") and r.get("present"):
            st2, b2, _, _, _ = fetch(base + path, "GET")
            if st2 == 200 and b2:
                f["endpoint_" + key].update(llms_txt_shape(b2))

    # 4) HTTP -> HTTPS（必须看第一跳，见 fetch_raw 的说明）
    if scheme == "https":
        st3, _, loc, _, _ = fetch_raw(f"http://{domain}", "GET")
        f["http_first_hop_status"] = st3
        f["http_to_https_location"] = loc
        f["redirects_http_to_https"] = (st3 in (301, 302, 307, 308)
                                        and bool(loc) and str(loc).lower().startswith("https"))

    f["_probe_seconds"] = round(time.time() - t0, 2)
    return out


def flatten_for_validity(rec: dict) -> dict:
    """把探测结果压成 11_construct_validity 需要的 probe dict。

    ⚠️ 三态纪律：True / False / None(未知) 全都保留。**不可达绝不塌成 False。**
    """
    f = rec.get("facts", {})
    rob = f.get("robots") or {}
    named = rob.get("mentions_ai_agents") or {}
    probe = {
        "llms_txt": (f.get("endpoint_llms_txt") or {}).get("present"),
        "llms_full_txt": (f.get("endpoint_llms_full_txt") or {}).get("present"),
        "has_wellknown_mcp": (f.get("endpoint__well-known_mcp") or {}).get("present"),
        "mcp_json": (f.get("endpoint_mcp_json") or {}).get("present"),
        "dot_mcp_json": (f.get("endpoint__mcp_json") or {}).get("present"),
        "robots_present": (None if rob.get("status") is None else rob.get("status") == 200),
        "robots_names_ai_agents": rob.get("n_ai_agents_named"),
        "robots_sitemap": rob.get("has_sitemap_line"),
        "has_jsonld": f.get("homepage", {}).get("has_jsonld"),
        "has_opengraph": f.get("homepage", {}).get("has_opengraph"),
        "has_canonical": f.get("homepage", {}).get("has_canonical"),
        "has_viewport": f.get("homepage", {}).get("has_viewport"),
        "lang_declared": bool(f.get("homepage", {}).get("lang_declared")),
        "redirects_http_to_https": f.get("redirects_http_to_https"),
        "homepage_ok": f.get("homepage", {}).get("status") == 200,
    }
    return probe


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------

def selftest() -> int:
    fails = []

    def chk(name, cond):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    print("== 12_probe_site_facts selftest ==")

    html = b"""<html lang="en"><head><title>T</title>
    <meta property="og:title" content="x"><meta name="viewport" content="width=device-width">
    <link rel="canonical" href="https://e.com/">
    <script type="application/ld+json">{}</script></head><body><h1>A</h1><h1>B</h1></body></html>"""
    p = _HeadParser()
    p.feed(html.decode())
    chk("lang 解析", p.lang == "en")
    chk("h1 计数 = 2", p.h1 == 2)
    chk("JSON-LD 检出", p.has_jsonld)
    chk("OpenGraph 检出", p.has_og)
    chk("viewport 检出", p.has_viewport)
    chk("canonical 检出", p.has_canonical)
    chk("title_len", p.title == "T")

    r = parse_robots(b"User-agent: GPTBot\nAllow: /\nUser-agent: *\nDisallow:\nSitemap: http://x/s.xml")
    chk("robots 点名 GPTBot", r["mentions_ai_agents"]["gptbot"] is True)
    chk("robots 未点名 ClaudeBot", r["mentions_ai_agents"]["claudebot"] is False)
    chk("robots sitemap 行", r["has_sitemap_line"] is True)
    chk("robots 计数 = 1", r["n_ai_agents_named"] == 1)

    r2 = parse_robots(b"User-agent: *\nDisallow: /")
    chk("无 AI bot 点名 -> 0", r2["n_ai_agents_named"] == 0)

    l = llms_txt_shape(b"# Title\n\n## Section\n\nSee https://a.com and https://b.com\nMCP: yes")
    chk("llms.txt 段落数", l["llms_txt_n_sections"] == 2)
    chk("llms.txt 链接数 = 2", l["llms_txt_n_links"] == 2)
    chk("llms.txt 提 MCP", l["llms_txt_mentions_mcp"] is True)

    # 三态纪律
    rec = {"facts": {
        "endpoint_llms_txt": {"present": None},
        "endpoint__well-known_mcp": {"present": False},
        "robots": {"status": 200, "mentions_ai_agents": {"gptbot": True},
                   "n_ai_agents_named": 1, "has_sitemap_line": True},
        "homepage": {"status": 200, "has_jsonld": True, "lang_declared": "en"},
        "redirects_http_to_https": True,
    }}
    fl = flatten_for_validity(rec)
    chk("不可达 -> None（不塌成 False）", fl["llms_txt"] is None)
    chk("明确 404 -> False", fl["has_wellknown_mcp"] is False)
    chk("JSON-LD -> True", fl["has_jsonld"] is True)
    chk("robots 计数透传", fl["robots_names_ai_agents"] == 1)

    chk("UA 自报且不伪装浏览器", "GEO-construct-validity-probe" in UA and "Mozilla" not in UA)
    chk("域内限速 >= 1s", MIN_INTERVAL >= 1.0)
    chk("并发上限 <= 4", MAX_WORKERS <= 4)

    # 2026-10-05 修复的回归防护：不依赖网络，直接验证重定向被禁用。
    # urllib 默认会跟随重定向，导致 http->https 探测拿到最终 200 而非第一跳 301。
    nr = _NoRedirect()
    got = nr.redirect_request(None, None, 301, "Moved", {}, "https://example.com/")
    chk("_NoRedirect 禁用了重定向（返回 None）", got is None)
    chk("fetch_raw 使用禁用重定向的 opener", _opener.handlers != []
        and any(isinstance(h, _NoRedirect) for h in _opener.handlers))
    import inspect
    src = inspect.getsource(probe_domain)
    chk("probe_domain 用 fetch_raw 而非 fetch 测 http://",
        'fetch_raw(f"http://{domain}"' in src)
    chk("记录了第一跳状态码", "http_first_hop_status" in src)
    chk("跳转判定同时要求 Location 指向 https",
        'startswith("https")' in src)

    print(f"\n{len(fails)} failed" if fails else "\nall checks passed")
    for f in fails:
        print(f"  FAILED: {f}")
    return 1 if fails else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="12_probe_site_facts.py",
        description="站点客观事实探测器（只采事实，不打分）")
    p.add_argument("--domains", default=None, help="每行一个域名")
    p.add_argument("--out", default=None, help="输出 JSON 路径")
    p.add_argument("--workers", type=int, default=MAX_WORKERS)
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args(argv)

    if a.selftest:
        return selftest()
    if not a.domains:
        print("ERROR: 需要 --domains", file=sys.stderr)
        return 2

    with open(a.domains, "r", encoding="utf-8-sig") as fh:
        domains = [ln.strip().split()[0] for ln in fh
                   if ln.strip() and not ln.strip().startswith("#")]

    results = []
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(probe_domain, d): d for d in domains}
        for i, fu in enumerate(cf.as_completed(futs), 1):
            d = futs[fu]
            try:
                rec = fu.result()
            except Exception as e:  # noqa: BLE001
                rec = {"domain": d, "facts": {}, "unreachable": ["probe_failed"],
                       "error": type(e).__name__}
            results.append(rec)
            reach = "OK " if not rec.get("unreachable") else "PART"
            print(f"[{i}/{len(domains)}] {reach} {d} "
                  f"({rec.get('facts', {}).get('_probe_seconds', '?')}s)", flush=True)

    results.sort(key=lambda r: r["domain"])
    payload = {
        "meta": {
            "n_domains": len(domains),
            "n_fully_reachable": sum(1 for r in results if not r.get("unreachable")),
            "elapsed_sec": round(time.time() - t0, 1),
            "ua": UA,
            "min_interval_sec": MIN_INTERVAL,
            "max_workers": a.workers,
            "three_state_discipline": "present=True / present=False / present=None(不可判定)",
        },
        "records": results,
        "flat": [{"domain": r["domain"], **flatten_for_validity(r)} for r in results],
    }

    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print(f"\nwrote {a.out}")
        print(f"完全可达 {payload['meta']['n_fully_reachable']}/{len(domains)}，"
              f"总耗时 {payload['meta']['elapsed_sec']}s")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
