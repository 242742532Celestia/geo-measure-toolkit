#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""geo_page_features.py -- 页面级候选特征抽取器（**不出分数**）

用途
----
把公开的 GEO 检查清单转成**可复算的原始特征向量**，供两处使用：

1. `examples/11_construct_validity.py` 的 C3 命题⑤ 构念效度方案
   （`the construct-validity plan` §5）—— 那里需要一个
   **独立探测脚本**产出的客观事实 dict。
2. C3 / E4 的**候选预测变量**：这些特征**是否**预测真实被引用，
   要用真实引用数据去估，不能用别人的手设权重表假定。

特征清单的来源是 HeiGe-GEO-SEO 的 6 维评分卡（MIT）。
**只借它的检查项清单，不借它的分数、权重与档位。**

🔴 三条红线
-----------
1. **本工具不出总分、不出权重、不出档位、不出 GEO/SEO 分。**
   特征是**待验证的候选预测变量**，不是分数。
   那套手设权重（18/16/16/22/16/12）没有任何引用数据支撑，
   直接用等于把「规则符合度」当成「被引用概率」——这正是
   `the tautology-audit spec` 要挡的那类同义反复。
   自检里有一条守卫会**递归扫描输出**，确保不出现 score/grade/weight 键。

2. **三态纪律**：True / False / None(不可判定)。
   不可判定**绝不塌成 False** —— 把「没采到」记成「没做到」本身就是伪造。

3. **不引入被检验对象的计分逻辑**。本模块不复现任何评分函数，
   不 import 任何外部打分库。

中文语料的口径纪律 🔴
--------------------
上游评分卡把 D2 写成「前 100 词」，实现却是 `[:140]`（**字符**）；
D3 / D4 也是「词」。**对中文页面，「词」没有空白分词口径**，
140 字符远不等于 100 词。本模块因此：

- 长度 / 句长一律**同时**给「字符」与「词元」两个口径；
- 文本判定为中文主导时，**词基指标一律置 None 并注明原因**，
  不用一个静默错误的数字去污染后续验证；
- D2 的阈值单位固定为**字符**，并在 `feature_meta` 里写死。

第三种状态：无法可靠判定
------------------------
除 True/False/None 外还有一类特征**根本不实现**（如「主谓宾结构」需句法分析、
「CDN/WAF 是否拦 AI UA」需网络探测）。这类特征值给 None、
`status` 记 `not_implemented` 并写明原因，**不给一个看起来像数据的比例**。

完整性纪律
----------
上游评分卡**自称**「22 个检查项」，但它的实现里有 **26 个** check
（A4 B4 C4 D5 E5 F4 = 26；「22」是维度 D 的权重，被误写成了项数）。
本模块把 26 个 check ID 写死为 `UPSTREAM_CHECKS`，
自检会逐个断言：**每个 check 要么在这里被测，要么被 12 号探测器覆盖，
要么显式 not_implemented 且带原因** —— 不允许有静默的空缺。

用法
----
    python tools/geo_page_features.py --html page.html --out features.json
    python tools/geo_page_features.py --html page.html --robots robots.txt \
                                     --llms-txt llms.txt --page-url https://x.com/p
    python tools/geo_page_features.py --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from html.parser import HTMLParser

SCHEMA = "geo_page_features/1"
VERSION = "1.0"

#: 上游 6 维评分卡的**全部** check ID（以实现为准，不是以文档自称为准）。
UPSTREAM_CHECKS = (
    "A1", "A2", "A3", "A4",
    "B1", "B2", "B3", "B4",
    "C1", "C2", "C3", "C4",
    "D1", "D2", "D3", "D4", "D5",
    "E1", "E2", "E3", "E4", "E5",
    "F1", "F2", "F3", "F4",
)

#: 由 `examples/12_probe_site_facts.py` 覆盖的**站点级**事实。
#: 这些 check 在本工具里不重复实现（页面级工具不该去探站点）。
COVERED_BY_PROBE = {
    "B2": "examples/12_probe_site_facts.py::endpoint_llms_full_txt",
    "B3": "examples/12_probe_site_facts.py::endpoint__well-known_mcp / mcp_json",
    "B4": "examples/12_probe_site_facts.py::robots.sitemap + sitemap 端点",
}

#: 显式不实现的 check + 原因。
NOT_IMPLEMENTED = {
    "A4": "CDN/WAF 是否对 AI UA 另设拦截，静态文件不可判定；"
          "需网络侧探测，且与本工具的「页面级」定位不同",
    "E4": "「句子多为清晰主-谓-宾结构」需要句法分析；"
          "用正则近似会产出看起来像数据的比例，污染后续验证",
    "F3": "实体跨页一致性是**站点级**属性，页面级不可判定；"
          "本页只报实体提及数（弱代理，reliability 已标注）",
    "B4_site": "",
}
NOT_IMPLEMENTED.pop("B4_site")

#: check -> 维度
CHECK_DIM = {
    "A1": "A", "A2": "A", "A3": "A", "A4": "A",
    "B1": "B", "B2": "B", "B3": "B", "B4": "B",
    "C1": "C", "C2": "C", "C3": "C", "C4": "C",
    "D1": "D", "D2": "D", "D3": "D", "D4": "D", "D5": "D",
    "E1": "E", "E2": "E", "E3": "E", "E4": "E", "E5": "E",
    "F1": "F", "F2": "F", "F3": "F", "F4": "F",
}

#: D2 答案前置阈值，单位固定为**字符**。
D2_ANSWER_FIRST_CHARS = 140

#: D4 篇幅区间（词元口径；中文页面不适用）。
D4_MIN_TOKENS = 800
D4_MAX_TOKENS = 1500

#: 判定「中文主导」的 CJK 字符占比阈值。
CJK_DOMINANT_RATIO = 0.30

_CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
_CJK_PUNCT = re.compile(r"[\u3000-\u303f\uff00-\uffef]")
_LETTER = re.compile(r"[A-Za-z]")
_NUM_UNIT = re.compile(r"\d+(?:[.,]\d+)?\s*(?:%|％|px|rem|em|kg|g|ml|l|cm|mm|nm|hz|khz|mhz|ghz|"
                       r"ms|s|min|h|天|日|月|年|次|倍|人|元|美元|万|亿|件|款|项|条|种|款型)?")
_QA_MARK = re.compile(r"(是什么|什么是|如何|怎么|怎样|为什么|为何|区别|区别是|哪个好|推荐|"
                       r"how to|what is|why|which\b)", re.IGNORECASE)
_DEFIN = re.compile(r"(是|为|指|意味着|包括|属于|叫做|称为|\bis\b|\bare\b|\bmeans\b|\brefers to\b)",
                    re.IGNORECASE)
_DEFIN_HEAD = re.compile(r"(是什么|什么是|如何|怎么|怎样|为什么|区别|概览|简介|定义|"
                         r"what is|how to|why\b|guide to|overview)", re.IGNORECASE)
_CTA = re.compile(r"(立即购买|马上购买|立即注册|免费试用|限时优惠|马上领取|立即下载|"
                   r"加入购物车|立即咨询|马上咨询|联系我们|立即开始|"
                   r"buy now|sign up|get started|free trial|contact us|shop now)", re.IGNORECASE)
_PAYWALL = re.compile(r"(订阅后阅读|开通会员|付费后可见|请先登录|会员专享全文|"
                       r"subscribe to (read|continue)|sign in to (read|continue)|"
                       r"paywall|metered content)", re.IGNORECASE)
_AUTHOR_ATTR = re.compile(r"(author|byline|writer|post-author|entry-author)", re.IGNORECASE)
_KG_HOSTS = ("wikipedia.org", "wikidata.org", "linkedin.com", "crunchbase.com")
_AUTH_DOMAINS = (".gov", ".edu", ".gov.cn", "wikipedia.org", "wikidata.org",
                 "baike.baidu.com", "nih.gov", "who.int", "arxiv.org",
                 "doi.org", "scholar.google.com")


# ---------------------------------------------------------------------------
# 文本工具
# ---------------------------------------------------------------------------

def is_cjk_dominant(text: str) -> bool:
    """中文主导判定。词元（空白分词）口径对中文无效，必须先判。"""
    if not text:
        return False
    cjk = len(_CJK.findall(text))
    letters = cjk + len(_LETTER.findall(text))
    if letters == 0:
        return False
    return cjk / letters >= CJK_DOMINANT_RATIO


def count_cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    cjk = len(_CJK.findall(text))
    letters = cjk + len(_LETTER.findall(text))
    return 0.0 if letters == 0 else round(cjk / letters, 4)


def split_sentences(text: str) -> list:
    """断句。**小数与域名不得当句界**。

    🔴 上游因为「3.5倍」被拆成两句，把 D1/D3/E4 三项指标全部扭曲
    （数据密集内容 —— GEO 最鼓励的写法 —— 测得最不准）。
    这里先用占位符保护 `数字.数字`，再断句。
    """
    if not text:
        return []
    # 用 lambda 做替换：替换串里的 \xNN 会被 re 当成转义模板而报错。
    # 哨兵取 Unicode 私有区字符，正常 HTML 不会出现。
    _SENT = "\ue000"
    protected = re.sub(r"(\d)\.(\d)",
                       lambda m: m.group(1) + _SENT + m.group(2), text)
    parts = re.split(r"(?<=[。！？!?；;])|(?<=\.)\s+|\n+", protected)
    out = []
    for p in parts:
        p = p.replace(_SENT, ".").strip()
        if p:
            out.append(p)
    return out


def visible_text_placeholder_safe(text: str) -> str:
    return text


# ---------------------------------------------------------------------------
# HTML 解析
# ---------------------------------------------------------------------------

class _PageParser(HTMLParser):
    """只采可自动判定的事实。**不做任何主观判断。**"""

    SKIP_TAGS = {"script", "style", "noscript", "template", "svg", "iframe"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title = None
        self.meta = []
        self.ldjson = []
        self.headings = []          # (level, text)
        self.links = []             # href
        self.n_list = 0
        self.n_table = 0
        self.n_dl = 0
        self.byline_text = ""
        self.text_parts = []
        self._skip = 0
        self._in_title = False
        self._title_buf = []
        self._heading = None
        self._heading_buf = []
        self._byline_depth = 0
        self._byline_buf = []
        self._in_author_meta = False

    # -- 标签 --
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in self.SKIP_TAGS:
            self._skip += 1
            return
        if tag == "title":
            self._in_title = True
            self._title_buf = []
        elif tag == "meta":
            name = (a.get("name") or a.get("property") or "").lower()
            content = a.get("content") or ""
            self.meta.append({"name": name, "content": content})
            if name in ("author", "article:author", "og:article:author"):
                self._in_author_meta = True
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._heading = int(tag[1])
            self._heading_buf = []
        elif tag == "a" and a.get("href"):
            self.links.append(a["href"])
        elif tag in ("ul", "ol"):
            self.n_list += 1
        elif tag == "table":
            self.n_table += 1
        elif tag in ("dl", "details"):
            self.n_dl += 1
        elif tag == "script" and (a.get("type") or "").lower() == "application/ld+json":
            self._in_ldjson = True
        # 署名容器
        cls = " ".join(filter(None, [a.get("class"), a.get("id"), a.get("itemprop")]))
        if _AUTHOR_ATTR.search(cls or "") or (a.get("rel") or "").lower() == "author":
            self._byline_depth += 1
            if self._byline_depth == 1:
                self._byline_buf = []

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag in self.SKIP_TAGS and self._skip:
            self._skip -= 1

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS:
            if self._skip:
                self._skip -= 1
            return
        if tag == "title":
            self._in_title = False
            if self.title is None:
                t = "".join(self._title_buf).strip()
                if t:
                    self.title = t
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6") and self._heading is not None:
            txt = "".join(self._heading_buf).strip()
            self.headings.append((self._heading, txt))
            self._heading = None
            self._heading_buf = []
        elif tag == "meta":
            self._in_author_meta = False
        elif self._byline_depth:
            self._byline_depth -= 1

    def handle_data(self, data):
        if self._skip:
            return
        if self._in_title:
            self._title_buf.append(data)
            return
        if self._heading is not None:
            self._heading_buf.append(data)
        if self._byline_depth:
            self._byline_buf.append(data)
        self.text_parts.append(data)

    def close(self):
        super().close()
        if self.title is None and self._title_buf:
            t = "".join(self._title_buf).strip()
            if t:
                self.title = t


def parse_page(html_text: str) -> dict:
    p = _PageParser()
    try:
        p.feed(html_text)
        p.close()
    except Exception:  # noqa: BLE001
        pass
    text = re.sub(r"\s+", " ", " ".join(p.text_parts)).strip()
    return {
        "title": p.title,
        "meta": p.meta,
        "headings": p.headings,
        "links": p.links,
        "n_list": p.n_list,
        "n_table": p.n_table,
        "n_dl": p.n_dl,
        "byline": re.sub(r"\s+", " ", "".join(p._byline_buf)).strip(),
        "text": text,
    }


# ---------------------------------------------------------------------------
# JSON-LD
# ---------------------------------------------------------------------------

def _walk(node, fn):
    if isinstance(node, dict):
        fn(node)
        for v in node.values():
            _walk(v, fn)
    elif isinstance(node, list):
        for v in node:
            _walk(v, fn)


def parse_jsonld(blocks: list) -> dict:
    parsed, invalid = [], []
    for raw in blocks:
        raw = (raw or "").strip()
        if not raw:
            continue          # 空块不算错（SSR 框架常输出空占位）
        try:
            parsed.append(json.loads(raw))
        except Exception:  # noqa: BLE001
            invalid.append(raw[:80])
    types = set()
    dup_type = False

    def collect(n):
        nonlocal dup_type
        t = n.get("@type")
        if isinstance(t, str):
            types.add(t)
        elif isinstance(t, list):
            if len(t) != len(set(map(str, t))):
                dup_type = True
            for x in t:
                types.add(x)

    for p in parsed:
        _walk(p, collect)
    return {"parsed": parsed, "types": types, "invalid_blocks": invalid,
            "dup_type": dup_type}


def max_property_count(parsed: list) -> int:
    best = 0

    def prop(n):
        nonlocal best
        keys = [k for k in n.keys() if not k.startswith("@")]
        best = max(best, len(keys))
    for p in parsed:
        _walk(p, prop)
    return best


def _nodes_of_type(parsed: list, type_name: str) -> list:
    out = []

    def collect(n):
        t = n.get("@type")
        if t == type_name or (isinstance(t, list) and type_name in t):
            out.append(n)
    for p in parsed:
        _walk(p, collect)
    return out


def faq_qa_complete(parsed: list) -> bool:
    """FAQPage 存在**且**每个 Question 都有 acceptedAnswer/Answer/text。

    「标了 FAQPage 却没问答」是**schema 造假**（上游称之为会被判作弊反噬），
    必须判 False 而不是 True。
    """
    faqs = _nodes_of_type(parsed, "FAQPage")
    if not faqs:
        return False
    total, answered = 0, 0
    for f in faqs:
        for q in f.get("mainEntity") or []:
            if not isinstance(q, dict):
                continue
            total += 1
            ans = q.get("acceptedAnswer") or q.get("suggestedAnswer") or {}
            has_text = False
            if isinstance(ans, dict):
                txt = ans.get("text")
                if isinstance(txt, str) and txt.strip():
                    has_text = True
                else:
                    for sub in ans.get("@graph") or []:
                        if isinstance(sub, dict) and isinstance(sub.get("text"), str) \
                                and sub["text"].strip():
                            has_text = True
            if has_text:
                answered += 1
    return total > 0 and answered == total


def has_author_person(parsed: list) -> bool:
    if _nodes_of_type(parsed, "Person"):
        return True
    for p in parsed:
        hit = {"v": False}

        def collect(n):
            if n.get("@type") in ("Article", "BlogPosting", "NewsArticle", "WebPage"):
                a = n.get("author")
                if a:
                    hit["v"] = True
        _walk(p, collect)
        if hit["v"]:
            return True
    return False


def sameas_kg(parsed: list) -> tuple:
    """返回 (是否指向知识图谱, sameAs 链接总数)。"""
    urls = []

    def collect(n):
        s = n.get("sameAs")
        if isinstance(s, str):
            urls.append(s)
        elif isinstance(s, list):
            urls.extend([x for x in s if isinstance(x, str)])
    for p in parsed:
        _walk(p, collect)
    kg = any(any(h in u.lower() for h in _KG_HOSTS) for u in urls)
    return kg, len(urls)


# ---------------------------------------------------------------------------
# robots / llms.txt
# ---------------------------------------------------------------------------

def parse_robots(text: str) -> dict:
    """UA 组 -> {'disallow': [...], 'allow': [...]}。

    🔴 两个必须处理的解析坑（否则同一份 robots 会得出相反结论）：
      1. `Crawl-delay` / `Host` 等指令**终结** UA 堆叠 ——
         否则下一组的 Disallow 会错挂到前一个 UA 头上。
      2. UTF-8 BOM —— Windows 记事本保存的 robots.txt 常带，
         不剥则首行 user-agent 解析失败。
    """
    groups, sitemaps = {}, []
    current, pending_new = [], True
    for raw in (text or "").lstrip("\ufeff").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        field, value = line.split(":", 1)
        field = field.strip().lower()
        value = value.strip()
        if field == "user-agent":
            if not pending_new:
                current = []
            ua = value.lower()
            groups.setdefault(ua, {"disallow": [], "allow": []})
            current.append(ua)
            pending_new = True
        elif field in ("disallow", "allow"):
            pending_new = False
            for ua in current or ["*"]:
                groups.setdefault(ua, {"disallow": [], "allow": []})
                groups[ua][field].append(value)
        elif field == "sitemap":
            sitemaps.append(value)
        else:
            pending_new = False
    return {"groups": groups, "sitemaps": sitemaps}


def ua_allowed(robots: dict, ua: str) -> bool:
    groups = robots["groups"]
    grp = groups.get(ua.lower())
    if grp is None:
        grp = groups.get("*")
    if grp is None:
        return True          # 没有任何规则 -> 默认放行
    blocked = any(d.strip() == "/" for d in grp["disallow"])
    if blocked and any(a.strip() == "/" for a in grp["allow"]):
        return True
    return not blocked


def llms_txt_wellformed(text: str) -> bool:
    """llms.txt 结构合格：有 H1、至少一个分节、至少一个链接。"""
    if not text:
        return False
    h1 = re.findall(r"^#\s+\S", text, flags=re.MULTILINE)
    heads = re.findall(r"^#{1,3}\s+\S", text, flags=re.MULTILINE)
    links = re.findall(r"https?://[^\s\)\]]+", text)
    return bool(h1) and len(heads) >= 2 and len(links) >= 1


# ---------------------------------------------------------------------------
# 特征抽取
# ---------------------------------------------------------------------------

def _meta_first(meta: list, names: tuple) -> str:
    for n in names:
        for m in meta:
            if m.get("name") == n and (m.get("content") or "").strip():
                return m["content"].strip()
    return ""


def extract_features(html_text: str, robots_text=None, llms_txt_text=None,
                     page_url: str = None) -> dict:
    """返回 {feature_name: raw_value} 与 {feature_name: meta}。"""
    page = parse_page(html_text)
    text = page["text"]
    cjk = is_cjk_dominant(text)
    tokens = text.split()
    n_tok = len(tokens)
    n_char = len(text)
    sents = split_sentences(text)
    meta = page["meta"]
    values, meta_out = {}, {}

    def put(name, value, check, name_cn, kind, unit, definition,
            status="measured", note="", reliability="solid"):
        values[name] = value
        meta_out[name] = {
            "check": check, "dimension": CHECK_DIM.get(check, "?"),
            "name": name_cn, "kind": kind, "unit": unit,
            "definition": definition, "status": status, "note": note,
            "reliability": reliability,
        }

    # ---- A：爬虫准入（文本可判定部分）----
    for cid, ua, cn in (("A1", "gptbot", "GPTBot/OAI-SearchBot 准入"),
                        ("A2", "claudebot", "ClaudeBot/Claude-SearchBot 准入"),
                        ("A3", "perplexitybot", "PerplexityBot 准入")):
        if robots_text is None:
            put(f"a_{cid.lower()}_allowed", None, cid, cn, "bool", None,
                "robots.txt 中该 UA 是否未被 Disallow: / 全面屏蔽",
                note="未提供 robots.txt -> 不可判定（不塌成 False）")
        else:
            r = parse_robots(robots_text)
            put(f"a_{cid.lower()}_allowed", ua_allowed(r, ua), cid, cn, "bool", None,
                "robots.txt 中该 UA 是否未被 Disallow: / 全面屏蔽",
                note="robots.txt 中无该 UA 且无 * 组规则 -> 按 HTTP 语义默认放行")

    # ---- B：AI 发现文件 ----
    if llms_txt_text is None:
        put("b_llms_txt_wellformed", None, "B1", "llms.txt 存在且结构合格",
            "bool", None, "有 H1、>=1 分节、>=1 链接",
            note="未提供 llms.txt -> 不可判定")
    else:
        put("b_llms_txt_wellformed", llms_txt_wellformed(llms_txt_text), "B1",
            "llms.txt 存在且结构合格", "bool", None,
            "有 H1、>=1 分节、>=1 链接")

    # ---- C：结构化数据 ----
    blocks = re.findall(r"<script[^>]+application/ld\+json[^>]*>(.*?)</script>",
                        html_text, flags=re.IGNORECASE | re.DOTALL)
    jl = parse_jsonld(blocks)
    parsed = jl["parsed"]
    put("c_faqpage_qa_complete", faq_qa_complete(parsed), "C1",
        "FAQPage schema 且 Q&A 配对完整", "bool", None,
        "存在 FAQPage 且每个 Question 都有 acceptedAnswer 文本",
        note="标了 FAQPage 却无问答 = schema 造假，判 False")
    core = {"WebSite", "Organization", "Article"}
    put("c_core_schema_types_count", len(jl["types"] & core), "C2",
        "核心 schema 覆盖数", "int", "类",
        "WebSite/Organization/Article 中出现的类数")
    put("c_schema_max_property_count", max_property_count(parsed), "C3",
        "Schema 丰富度", "int", "属性数", "任一节点的最多非 @ 属性数")
    put("c_schema_all_valid", (not jl["invalid_blocks"]) and (not jl["dup_type"]),
        "C4", "Schema 有效性", "bool", None,
        "JSON-LD 全部可解析且无重复 @type")

    # ---- D：内容可抽取性 ----
    claim_sents = [s for s in sents
                   if _NUM_UNIT.search(s) or _QA_MARK.search(s) or _DEFIN.search(s)]
    put("d_claim_sentence_count", len(claim_sents), "D1", "可抽取事实句数",
        "int", "句", "含数字/单位、问答式标记或定义式谓语的句子数")
    put("d_claim_density_per_100_chars",
        None if n_char == 0 else round(len(claim_sents) * 100.0 / n_char, 3),
        "D1", "主张密度（字符口径）", "float", "句/百字符",
        "可抽取事实句数 / 正文字符数 * 100")
    put("d_claim_density_per_100_tokens",
        None if (cjk or n_tok == 0) else round(len(claim_sents) * 100.0 / n_tok, 3),
        "D1", "主张密度（词元口径）", "float", "句/百词元",
        "可抽取事实句数 / 词元数 * 100",
        note=("中文主导文本：空白分词无口径，词基指标不可判定 -> None"
              if cjk else ""))

    # D2：答案前置。阈值单位固定为**字符**。
    first_claim_end = None
    if claim_sents:
        first_claim_end = text.find(claim_sents[0]) + len(claim_sents[0])
    put("d_first_claim_end_char", first_claim_end, "D2",
        "首个可抽取事实句结束位置", "int", "字符",
        "首个可抽取事实句结束处距正文开头的字符数")
    put("d_answer_first_140", None if first_claim_end is None
        else (first_claim_end <= D2_ANSWER_FIRST_CHARS), "D2",
        "答案前置（阈值单位=字符）", "bool", None,
        f"首个可抽取事实句是否结束于前 {D2_ANSWER_FIRST_CHARS} 字符内",
        note=f"阈值 {D2_ANSWER_FIRST_CHARS} **字符**（非词）。"
             "上游自检表写「前 100 词」而实现是 [:140] 字符，"
             "中文页上两者相差数倍，本工具固定为字符口径并写死。")

    # D3：句长
    if sents:
        put("d_avg_sentence_chars",
            round(sum(len(s) for s in sents) / len(sents), 2), "D3",
            "平均句长（字符口径）", "float", "字符/句", "正文句平均字符数")
        put("d_avg_sentence_tokens",
            None if cjk else round(sum(len(s.split()) for s in sents) / len(sents), 2),
            "D3", "平均句长（词元口径）", "float", "词元/句",
            "正文句平均词元数",
            note=("中文主导文本：空白分词无口径 -> None" if cjk else ""))
    else:
        put("d_avg_sentence_chars", None, "D3", "平均句长（字符口径）",
            "float", "字符/句", "正文句平均字符数")
        put("d_avg_sentence_tokens", None, "D3", "平均句长（词元口径）",
            "float", "词元/句", "正文句平均词元数",
            note=("中文主导文本：空白分词无口径 -> None" if cjk else ""))

    # D4：篇幅
    put("d_text_chars", n_char, "D4", "正文长度（字符）", "int", "字符",
        "可见正文字符数")
    put("d_text_tokens", n_tok, "D4", "正文长度（词元）", "int", "词元",
        "可见正文空白分词数",
        note=("中文主导文本：词元数不具词义，仅作长度参考" if cjk else ""))
    put("d_in_800_1500_tokens",
        None if cjk else (D4_MIN_TOKENS <= n_tok <= D4_MAX_TOKENS), "D4",
        "篇幅落在 800~1500 词元区间", "bool", None,
        f"{D4_MIN_TOKENS}~{D4_MAX_TOKENS} 词元",
        note=("中文主导文本：上游区间以「词」计，对中文不可直接套用 -> None；"
              "中文篇幅须另定口径" if cjk else ""))

    # D5：统计与外部引用
    n_stat = sum(1 for s in sents if _NUM_UNIT.search(s))
    put("d_numeric_statement_count", n_stat, "D5", "含数字的句子数", "int", "句",
        "含数字（可带单位）的句子数")
    abs_links = [h for h in page["links"] if re.match(r"^https?://", h, re.I)]
    put("d_absolute_link_count", len(abs_links), "D5", "绝对外链数", "int", "个",
        "指向绝对 URL 的链接数")
    if page_url:
        self_host = re.sub(r"^https?://", "", page_url.lower()).split("/")[0]
        self_host = re.sub(r"^www\.", "", self_host)
        ext = []
        for h in abs_links:
            host = re.sub(r"^https?://", "", h.lower()).split("/")[0]
            host = re.sub(r"^www\.", "", host)
            if host != self_host:
                ext.append(h)
        put("d_external_link_count", len(ext), "D5", "站外链接数", "int", "个",
            "host 与本页 host 不同的绝对链接数",
            note=f"self_host={self_host}")
    else:
        put("d_external_link_count", None, "D5", "站外链接数", "int", "个",
            "host 与本页 host 不同的绝对链接数",
            note="未提供 --page-url -> 站内/站外不可区分 -> None")

    # ---- E：结构与可解析性 ----
    h1s = [h for h in page["headings"] if h[0] == 1]
    levels = [h[0] for h in page["headings"]]
    skip = False
    prev = 0
    for lv in levels:
        if prev and lv > prev + 1:
            skip = True
            break
        prev = lv
    put("e_h1_count", len(h1s), "E1", "H1 数量", "int", "个", "一级标题个数")
    put("e_heading_levels_ok", (len(h1s) == 1 and not skip), "E1",
        "标题层级合理", "bool", None, "单一 H1 且无跳级")
    put("e_has_list_or_table", (page["n_list"] > 0 or page["n_table"] > 0), "E2",
        "含列表或表格", "bool", None, "存在 ul/ol 或 table")
    put("e_list_count", page["n_list"], "E2", "列表块数", "int", "个", "ul/ol 块数")
    put("e_table_count", page["n_table"], "E2", "表格数", "int", "个", "table 元素数")
    def_head = [t for _, t in page["headings"] if t and _DEFIN_HEAD.search(t)]
    put("e_definition_qa_heading_count", len(def_head), "E3",
        "定义式/问答式标题数", "int", "个",
        "标题命中「是什么/如何/为什么/定义/what is/how to」等")
    put("e_has_definition_qa_heading", bool(def_head), "E3",
        "含定义式或问答式标题", "bool", None, "存在至少一个此类标题")

    cta_n = len(_CTA.findall(text))
    paywall = bool(_PAYWALL.search(text))
    put("e_cta_count", cta_n, "E5", "CTA 数量", "int", "处",
        "命中的行动号召短语数")
    put("e_paywall_marker", paywall, "E5", "付费墙/登录墙标记", "bool", None,
        "正文出现订阅/登录后可见类标记")
    if cjk:
        put("e_max_token_repeat_ratio", None, "E5", "最高词频占比", "float", "比例",
            "出现最多的词元占比",
            note="中文主导文本：词频（关键词堆砌）需分词，本工具不做 -> None")
    else:
        counts = {}
        for t in tokens:
            k = t.lower()
            if len(k) >= 3:
                counts[k] = counts.get(k, 0) + 1
        put("e_max_token_repeat_ratio",
            (max(counts.values()) / n_tok) if (counts and n_tok) else None,
            "E5", "最高词频占比", "float", "比例", "出现最多的词元占比")
    put("e_no_anti_signal", (cta_n <= 3 and not paywall), "E5",
        "无反引用信号", "bool", None, "CTA <= 3 处且无付费墙标记",
        note="关键词堆砌一项见 e_max_token_repeat_ratio（中文为 None）")

    # ---- F：信任、实体、权威 ----
    byline = page["byline"] or _meta_first(meta, ("author", "article:author",
                                                  "og:article:author"))
    person = has_author_person(parsed)
    put("f_has_author_byline", bool(byline), "F1", "有具名作者署名", "bool", None,
        "byline 容器或 author meta 非空")
    put("f_has_person_schema", person, "F1", "有 Person/author schema", "bool",
        None, "JSON-LD 含 Person，或 Article 的 author 字段非空")
    put("f_author_and_person", (bool(byline) and person), "F1",
        "署名 + Person schema 兼具", "bool", None, "两个子条件的合取")
    kg, n_same = sameas_kg(parsed)
    put("f_sameas_kg_link", kg, "F2", "sameAs 指向知识图谱", "bool", None,
        "sameAs 含 wikipedia/wikidata/linkedin/crunchbase")
    put("f_sameas_count", n_same, "F2", "sameAs 链接数", "int", "个",
        "JSON-LD 中 sameAs 字符串总数")

    ent_tokens = set()
    for m in meta:
        nm = (m.get("name") or "")
        if nm in ("og:site_name", "application-name", "og:title"):
            ent_tokens.update(w for w in re.findall(r"[A-Z][A-Za-z0-9]{2,}", nm))
    for _, t in page["headings"]:
        ent_tokens.update(re.findall(r"[A-Z][A-Za-z0-9]{2,}", t or ""))
    put("f_entity_mention_count", len(ent_tokens), "F3", "本页实体提及数", "int",
        "个", "标题/meta 中出现的大写专名 token 数（**弱代理**）",
        reliability="weak",
        note="仅本页；跨页实体一致性是站点级属性，见 not_implemented F3")

    years = []
    for m in meta:
        nm = (m.get("name") or "")
        if nm in ("article:modified_time", "article:published_time",
                  "og:updated_time", "last-modified", "date"):
            y = re.search(r"(19|20)\d{2}", m.get("content") or "")
            if y:
                years.append(int(y.group(0)))
    # 非捕获组：带捕获组时 findall 只返回 (19|20)，int() 后变成 20 而不是 2026。
    body_years = [int(y) for y in re.findall(r"(?:19|20)\d{2}", text)]
    put("f_has_datemodified", bool(years), "F4", "有 dateModified 类元数据",
        "bool", None, "meta 中存在修改/发布时间字段")
    put("f_max_year_in_meta", max(years) if years else None, "F4",
        "meta 中最新年份", "int", "年", "修改/发布时间字段里的最大年份")
    put("f_max_year_in_body", max(body_years) if body_years else None, "F4",
        "正文出现的最大年份", "int", "年", "正文文本中出现的最大年份")

    return {"values": values, "meta": meta_out, "page": page,
            "cjk_dominant": cjk, "cjk_ratio": count_cjk_ratio(text),
            "n_sentences": len(sents), "n_tokens": n_tok, "n_chars": n_char,
            "claim_sentences": len(claim_sents)}


# ---------------------------------------------------------------------------
# 输出装配
# ---------------------------------------------------------------------------

FORBIDDEN_KEY_PATTERNS = ("score", "grade", "weight", "total_points", "rating")


def _scan_forbidden(node, path="", hits=None):
    """递归扫描输出，确保**不出现**分数/权重/档位类键。"""
    if hits is None:
        hits = []
    if isinstance(node, dict):
        for k, v in node.items():
            if any(p in str(k).lower() for p in FORBIDDEN_KEY_PATTERNS):
                hits.append(f"{path}.{k}")
            _scan_forbidden(v, f"{path}.{k}", hits)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _scan_forbidden(v, f"{path}[{i}]", hits)
    return hits


def build_payload(html_text: str, robots_text=None, llms_txt_text=None,
                  page_url: str = None, html_path: str = None) -> dict:
    ex = extract_features(html_text, robots_text, llms_txt_text, page_url)
    values, fmeta = ex["values"], ex["meta"]

    # 只允许 bool/None 进 probe_flat —— 11_construct_validity 的
    # convergent_validity() 用 `1.0 if v else 0.0` 把 probe 值压成 0/1，
    # **任何数值特征放进去都会被静默压成布尔**。这里直接挡住。
    probe_flat, excluded = {}, []
    for name, v in values.items():
        if isinstance(v, bool) or v is None:
            probe_flat[name] = v
        else:
            excluded.append(name)

    coverage = {}
    for cid in UPSTREAM_CHECKS:
        if cid in NOT_IMPLEMENTED:
            coverage[cid] = {"status": "not_implemented", "reason": NOT_IMPLEMENTED[cid]}
        elif cid in COVERED_BY_PROBE:
            coverage[cid] = {"status": "covered_by_probe",
                             "where": COVERED_BY_PROBE[cid]}
        else:
            hits = [n for n, m in fmeta.items() if m["check"] == cid]
            coverage[cid] = {"status": "measured",
                             "features": sorted(hits)} if hits else {
                "status": "GAP", "reason": "未实现且未登记 —— 自检会报出"}

    payload = {
        "schema": SCHEMA,
        "meta": {
            "extractor": "tools/geo_page_features.py",
            "version": VERSION,
            "deterministic": True,
            "llm_used": False,
            "ranking_emitted": False,
            "note": "本工具**不出**总分/权重/档位。特征是待验证的候选预测变量。",
            "html_path": html_path,
            "page_url": page_url,
            "cjk_dominant": ex["cjk_dominant"],
            "cjk_ratio": ex["cjk_ratio"],
            "n_chars": ex["n_chars"],
            "n_tokens": ex["n_tokens"],
            "n_sentences": ex["n_sentences"],
        },
        "features": values,
        "feature_meta": fmeta,
        "probe_flat": probe_flat,
        "probe_flat_excluded_numeric": excluded,
        "coverage": coverage,
        "discipline": {
            "three_state": "True / False / None(不可判定)；None 绝不塌成 False",
            "no_ranking": "输出中不存在总分/权重/档位类键（自检递归扫描键名）",
            "probe_flat_typing": "probe_flat 只含 bool|None —— "
                                 "11_construct_validity 的 convergent_validity() "
                                 "用 `1.0 if v else 0.0` 会把数值静默压成布尔",
            "word_vs_char": "词基指标在中文主导文本下一律 None（空白分词无口径）",
        },
    }
    return payload


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------

_GOOD_HTML = """<!doctype html>
<html lang="zh-CN"><head>
<meta charset="utf-8">
<title>VPN 是什么 · 2026 选购指南</title>
<meta name="author" content="李明">
<meta name="article:modified_time" content="2026-08-14">
<link rel="canonical" href="https://example.com/vpn">
<script type="application/ld+json">
{"@context":"https://schema.org","@type":"FAQPage","mainEntity":[
 {"@type":"Question","name":"VPN 是什么？",
  "acceptedAnswer":{"@type":"Answer","text":"VPN 是一种加密隧道协议，2026 年主流协议为 WireGuard。"}},
 {"@type":"Question","name":"VPN 多少钱？",
  "acceptedAnswer":{"@type":"Answer","text":"个人版每月 30 元，家庭版每月 60 元。"}}]}
</script>
<script type="application/ld+json">
{"@context":"https://schema.org","@type":"Article","name":"VPN 是什么",
 "author":{"@type":"Person","name":"李明"},
 "sameAs":["https://zh.wikipedia.org/wiki/VPN"]}
</script>
</head>
<body>
<article>
<h1>VPN 是什么</h1>
<p>VPN 是一种加密隧道协议。它在 2026 年被广泛使用。</p>
<p>速度提升 3.5倍，延迟降低 40%。</p>
<h2>如何选择</h2>
<ul><li>看协议</li><li>看日志政策</li></ul>
<h2>常见问题</h2>
<table><tr><td>协议</td><td>WireGuard</td></tr></table>
<p>更多细节见 <a href="https://example.com/more">本页</a> 与
<a href="https://www.rfc-editor.org/rfc/rfc9110">RFC 9110</a>。</p>
</article>
</body></html>"""

_POOR_HTML = """<!doctype html>
<html><head><title>Product</title></head>
<body><div>Buy now</div><div>Subscribe to read more</div></body></html>"""

_FAKE_FAQ_HTML = """<!doctype html>
<html><head><title>T</title>
<script type="application/ld+json">
{"@context":"https://schema.org","@type":"FAQPage","mainEntity":[]}
</script></head><body><h1>T</h1></body></html>"""


def selftest() -> int:
    checks = []

    def ck(name, cond, extra=""):
        checks.append((name, bool(cond)))
        if not cond:
            print(f"  [FAIL] {name}" + (f"  （{extra}）" if extra else ""))

    print("== geo_page_features selftest ==")

    # ---- 1. 特征集完整性：26 个 check 全部有交代 ----
    payload = build_payload(_GOOD_HTML, page_url="https://example.com/vpn",
                            html_path="<good>")
    cov = payload["coverage"]
    ck("上游 check 清单为 26 项（文档自称 22，实现是 26）",
       len(UPSTREAM_CHECKS) == 26, f"实际 {len(UPSTREAM_CHECKS)}")
    ck("覆盖表覆盖全部 26 项", len(cov) == 26, f"实际 {len(cov)}")
    gaps = [c for c, v in cov.items() if v["status"] == "GAP"]
    ck("无未交代的 check（不允许静默空缺）", not gaps, f"GAP={gaps}")
    ni = [c for c, v in cov.items() if v["status"] == "not_implemented"]
    ck("不实现的 check 全部带原因",
       all(cov[c].get("reason") for c in ni), f"{ni}")
    cb = [c for c, v in cov.items() if v["status"] == "covered_by_probe"]
    ck("被 12 号探测器覆盖的 check 全部带去向",
       all(cov[c].get("where") for c in cb), f"{cb}")

    # ---- 2. 绝不产出分数：递归扫键 ----
    hits = _scan_forbidden(payload)
    ck("输出中无 score/grade/weight 类键", not hits, f"{hits[:5]}")
    ck("meta 显式声明 ranking_emitted=False",
       payload["meta"]["ranking_emitted"] is False)
    ck("values 里没有任何键名含 score/weight/grade",
       not [k for k in payload["features"]
            if any(p in k.lower() for p in FORBIDDEN_KEY_PATTERNS)])

    # ---- 3. 三态纪律 ----
    p_none = build_payload(_GOOD_HTML, html_path="<no-robots>")
    ck("未提供 robots.txt -> A1 为 None（不塌成 False）",
       p_none["features"]["a_a1_allowed"] is None)
    ck("未提供 robots.txt -> A2 为 None", p_none["features"]["a_a2_allowed"] is None)
    ck("未提供 robots.txt -> A3 为 None", p_none["features"]["a_a3_allowed"] is None)
    ck("未提供 llms.txt -> B1 为 None", p_none["features"]["b_llms_txt_wellformed"] is None)

    # ---- 4. robots 语义 ----
    p_ro = build_payload(_GOOD_HTML,
                         robots_text="User-agent: GPTBot\nDisallow: /\n"
                                     "User-agent: *\nAllow: /\n",
                         html_path="<robots>")
    ck("GPTBot 被 Disallow: / -> False",
       p_ro["features"]["a_a1_allowed"] is False)
    ck("未被点名的 UA 按 * 组放行 -> True",
       p_ro["features"]["a_a2_allowed"] is True)
    p_ro2 = build_payload(_GOOD_HTML,
                          robots_text="User-agent: GPTBot\nDisallow: /\n"
                                      "Crawl-delay: 1\nUser-agent: *\nAllow: /\n",
                          html_path="<robots2>")
    ck("Crawl-delay 终结 UA 堆叠（不把下一组错挂给前一 UA）",
       p_ro2["features"]["a_a1_allowed"] is False)
    p_ro3 = build_payload(_GOOD_HTML, robots_text="User-agent: *\nDisallow: /\n",
                          html_path="<robots3>")
    ck("* 组 Disallow: / -> 点名 UA 也被拦",
       p_ro3["features"]["a_a1_allowed"] is False)

    # ---- 5. CJK 口径纪律（上游把「词」当「字符」用的那个坑）----
    f = p_none["features"]
    ck("中文页判定为 CJK 主导", p_none["meta"]["cjk_dominant"] is True)
    ck("中文页：词基主张密度为 None", f["d_claim_density_per_100_tokens"] is None)
    ck("中文页：词基平均句长为 None", f["d_avg_sentence_tokens"] is None)
    ck("中文页：词基篇幅区间为 None", f["d_in_800_1500_tokens"] is None)
    ck("中文页：字符口径仍可用（主张密度非空）",
       f["d_claim_density_per_100_chars"] is not None)
    ck("中文页：句长字符口径非空", f["d_avg_sentence_chars"] is not None)
    ck("D2 阈值单位写死为字符",
       "字符" in p_none["feature_meta"]["d_answer_first_140"]["note"])

    # ---- 6. 小数不得当句界 ----
    sents = split_sentences("版本 3.5倍 效果。另有 2.0 版本。")
    ck("小数不被拆句（3.5倍 不产生额外句界）",
       len(sents) == 2, f"实际 {len(sents)}: {sents}")
    s2 = split_sentences("See example.com. Then more.")
    ck("域名不产生句界", len(s2) == 2, f"实际 {len(s2)}: {s2}")

    # ---- 7. 阴性对照：差页面必须被判差 ----
    p_poor = build_payload(_POOR_HTML, html_path="<poor>")
    pf = p_poor["features"]
    ck("阴性对照：无 FAQPage -> C1 False", pf["c_faqpage_qa_complete"] is False)
    ck("阴性对照：核心 schema 覆盖数 = 0", pf["c_core_schema_types_count"] == 0)
    ck("阴性对照：无 H1 -> E1 不合格", pf["e_heading_levels_ok"] is False)
    ck("阴性对照：无列表无表格 -> E2 False", pf["e_has_list_or_table"] is False)
    ck("阴性对照：付费墙标记 -> E5 不合格", pf["e_no_anti_signal"] is False)
    ck("阴性对照：CTA 计数 > 0", pf["e_cta_count"] >= 1)
    ck("阴性对照：缺署名 -> F1 False", pf["f_has_author_byline"] is False)

    # ---- 8. schema 造假必须判 False ----
    p_fake = build_payload(_FAKE_FAQ_HTML, html_path="<fake-faq>")
    ck("标了 FAQPage 但无问答 -> C1 False（schema 造假不放过）",
       p_fake["features"]["c_faqpage_qa_complete"] is False)

    # ---- 9. 正样本（用带 page_url 的那份，站外链接才可判定）----
    gf = payload["features"]
    ck("正样本：FAQPage Q&A 配对 -> C1 True", gf["c_faqpage_qa_complete"] is True)
    ck("正样本：核心 schema 覆盖 = 1（Article；FAQPage/Person 不属核心集）",
       gf["c_core_schema_types_count"] == 1,
       f"实际 {gf['c_core_schema_types_count']}")
    ck("正样本：JSON-LD 全部可解析", gf["c_schema_all_valid"] is True)
    ck("正样本：单一 H1", gf["e_h1_count"] == 1)
    ck("正样本：标题层级合格", gf["e_heading_levels_ok"] is True)
    ck("正样本：含列表或表格", gf["e_has_list_or_table"] is True)
    ck("正样本：定义式标题被识别", gf["e_has_definition_qa_heading"] is True)
    ck("正样本：有署名", gf["f_has_author_byline"] is True)
    ck("正样本：有 Person schema", gf["f_has_person_schema"] is True)
    ck("正样本：sameAs 指向维基 -> F2 True", gf["f_sameas_kg_link"] is True)
    ck("正样本：含数值句子被计数", gf["d_numeric_statement_count"] >= 1)
    ck("正文年份取完整四位（不是 (19|20) 捕获组）",
       (gf["f_max_year_in_body"] is None
        or (1000 <= gf["f_max_year_in_body"] <= 2999)),
       f"实际 {gf['f_max_year_in_body']}")
    ck("正样本：站外链接被计数（RFC 9110）", gf["d_external_link_count"] >= 1)

    # ---- 10. D2 阈值行为 ----
    long_early = "<h1>T</h1><p>" + ("前置填充文字。" * 40) + \
                 "VPN 是一种加密隧道协议。" + ("后续内容。" * 40) + "</p>"
    p_late = build_payload(long_early, html_path="<late>")
    ck("首个事实句在 140 字符外 -> answer_first_140 False",
       p_late["features"]["d_answer_first_140"] is False)
    ck("首个事实句位置被如实记录（>140）",
       (p_late["features"]["d_first_claim_end_char"] or 0) > 140)

    # ---- 11. probe_flat 类型守卫 ----
    pf_keys = p_none["probe_flat"]
    ck("probe_flat 只含 bool|None",
       all(isinstance(v, bool) or v is None for v in pf_keys.values()))
    ck("数值特征被显式排除出 probe_flat",
       "d_claim_density_per_100_chars" in p_none["probe_flat_excluded_numeric"])
    ck("被排除的数值特征确实不在 probe_flat 里",
       "d_claim_density_per_100_chars" not in pf_keys)

    # ---- 12. 确定性 ----
    a = build_payload(_GOOD_HTML, page_url="https://example.com/vpn")
    b = build_payload(_GOOD_HTML, page_url="https://example.com/vpn")
    ck("同输入两次输出逐字相同（确定性）",
       json.dumps(a, ensure_ascii=False, sort_keys=True) ==
       json.dumps(b, ensure_ascii=False, sort_keys=True))

    # ---- 13. 无网络、无 LLM ----
    import inspect
    src = inspect.getsource(extract_features)
    ck("抽取层不 import 任何网络库",
       not any(m in src for m in ("import requests", "urllib", "http.client")))
    ck("抽取层不用 LLM", "openai" not in src and "anthropic" not in src)

    # ---- 14. 不可实现项必须显式 ----
    ck("E4 记为 not_implemented", cov["E4"]["status"] == "not_implemented")
    ck("A4 记为 not_implemented", cov["A4"]["status"] == "not_implemented")
    ck("F3 记为 not_implemented", cov["F3"]["status"] == "not_implemented")
    ck("B2/B3/B4 指向 12 号探测器",
       all(cov[c]["status"] == "covered_by_probe" for c in ("B2", "B3", "B4")))

    # ---- 15. 端到端：CLI 可用 ----
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        hp = os.path.join(td, "p.html")
        with open(hp, "w", encoding="utf-8") as fh:
            fh.write(_GOOD_HTML)
        outp = os.path.join(td, "o.json")
        rc = main(["--html", hp, "--out", outp])
        ok = rc == 0 and os.path.exists(outp)
        ck("CLI 端到端可跑并落盘", ok)
        if ok:
            with open(outp, encoding="utf-8") as fh:
                disk = json.load(fh)
            ck("落盘结果含 features", bool(disk.get("features")))
            ck("落盘结果无分数键", not _scan_forbidden(disk))

    failed = [n for n, ok in checks if not ok]
    for n, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {n}")
    print(f"\n{len(checks) - len(failed)} PASS   {len(failed)} FAIL")
    return 0 if not failed else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _read(path):
    if path is None:
        return None
    if path == "-":
        return sys.stdin.read()
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="geo_page_features.py",
        description="页面级候选特征抽取器（不出总分/权重/档位）")
    p.add_argument("--html", help="页面 HTML 文件路径（'-' 读 stdin）")
    p.add_argument("--robots", default=None, help="robots.txt 文本路径")
    p.add_argument("--llms-txt", dest="llms_txt", default=None,
                   help="llms.txt 文本路径")
    p.add_argument("--page-url", dest="page_url", default=None,
                   help="本页 URL（用于区分站外链接）")
    p.add_argument("--out", default=None, help="输出 JSON 路径")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args(argv)

    if a.selftest:
        return selftest()
    if not a.html:
        print("ERROR: 需要 --html", file=sys.stderr)
        return 2

    html = _read(a.html)
    payload = build_payload(html, _read(a.robots), _read(a.llms_txt),
                            a.page_url, a.html)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"-> {os.path.normpath(a.out)}  "
              f"({len(payload['features'])} 个特征，"
              f"{len(payload['probe_flat_excluded_numeric'])} 个数值特征已排除出 probe_flat)")
        print("注意：本工具**不出分数**。特征是待验证的候选预测变量。")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
