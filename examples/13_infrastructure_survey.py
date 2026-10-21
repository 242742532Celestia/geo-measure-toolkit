#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
13_infrastructure_survey.py -- agent 可调用性基础设施现状调查

两个产出
--------
A. **字段可观测性矩阵**：逐条列出 `agent_readiness` 的计分字段，
   标注它能否从外部自动探测。这一栏直接决定 C3 命题⑤ 的构念效度检验
   在技术上可不可执行 —— **不是样本量问题，是可观测性问题。**
B. **描述性统计**：这批站点上 llms.txt / MCP / robots 点名 AI bot 的实际普及率。

为什么 B 仍然有价值
-------------------
2607.12056 与 2609.34951 都是**推方法论**，没有一份「现在到底有多少站点
真的部署了这些」的一手数据。本调查填的是这个空。

用法
----
    python examples/13_infrastructure_survey.py --facts path/to/site_facts.json
    python examples/13_infrastructure_survey.py --selftest
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

# ---------------------------------------------------------------------------
# A. 字段可观测性矩阵
#
# 逐条读自 geo_core/agent_readiness.py 的计分逻辑（2026-10-05 核对）。
#
# ⚠️ 分值口径：这里列的是**每个字段在未封顶情况下的最大贡献**，
#    所以按维度加总是 40 / 35 / 40 = 115，**大于满分 100**。
#    原因是实际实现用 `min(s, DIM_MAX)` 封顶：
#       interpretability       潜在 40 -> 封顶 30（semantic_sections 的 10 分会被砍）
#       executability          潜在 35 -> 封顶 35
#       decision_reliability   潜在 40 -> 封顶 35（末尾若干项会被砍）
#    因此**矩阵里任何一行都不能单独当成「站点在这一项能拿多少分」的保证**，
#    它只表示「若该项被满足，规则允许它贡献多少」。
#
# 等级：
#   YES    = 外部探测能拿到完整内容，可直接填
#   PARTIAL = 只能测「有没有」，拿不到内容；或只能测代理量
#   NO     = 商家内部信息，外部不可得
# ---------------------------------------------------------------------------

DIM_CAPS = {"interpretability": 30, "executability": 35, "decision_reliability": 35}

FIELDS = [
    # (维度, 字段, 分值, 可观测性, 外部可得什么, 备注)
    ("interpretability", "jsonld / schema_jsonld", 30, "PARTIAL",
     "只能测「首页有没有 <script type=application/ld+json>」",
     "12 分只要 dict 非空即可得；但 @id/sku/gtin/mpn（+8）、"
     "name+description（+6）、offers/priceSpecification（+4）共 18 分要解析内容"),
    ("interpretability", "semantic_sections", 10, "NO",
     "无", "「语义化分节」是站点自述，外部无判定标准"),

    ("executability", "mcp_server_card", 18, "YES",
     "GET /.well-known/mcp.json 可拿完整 card 内容",
     "**唯一能完整外部验证的计分项**。含 name/description/server_url/tools/typed"),
    ("executability", "webmcp_tools", 10, "NO",
     "无", "浏览器端注入，服务器端不可见"),
    ("executability", "commerce_feed", 7, "NO",
     "无", "内部商务接口"),

    ("decision_reliability", "crawl_policy.allow_retrieval_bots", 10, "PARTIAL",
     "robots.txt 里是否点名 GPTBot/ClaudeBot 等（**代理量**）",
     "⚠️ 代理 ≠ 声明。robots 点名不等于 crawl_policy 显式放行，两者不是一回事"),
    ("decision_reliability", "crawl_policy.training_retrieval_separated", 5, "PARTIAL",
     "robots.txt 里检索 bot 与训练 bot 是否分开列出（**代理量**）",
     "同上，是代理量"),
    ("decision_reliability", "freshness.last_updated", 7, "PARTIAL",
     "HTML meta 里的日期（如 article:modified_time）",
     "⚠️ 页面上有日期 ≠ 站点为 agent 声明了时效"),
    ("decision_reliability", "freshness.volatile_fields_annotated", 5, "NO",
     "无", "价格/库存类字段的标注是站点内部约定"),
    ("decision_reliability", "omission_guard.fact_completeness_checked", 8, "NO",
     "无", "业务流程，外部完全不可见"),
    ("decision_reliability", "omission_guard.missing_declared", 5, "NO",
     "无", "业务流程，外部完全不可见"),
]

SOFT_SIGNALS = [
    ("llms.txt", "GET /llms.txt", 0, "YES —— 但设计上 0 分，只进 diagnostics"),
    ("markdown endpoint", "无标准路径", 0, "NO"),
    ("content signals (ai-usage/ai-license meta)", "首页 meta 标签", 0, "PARTIAL"),
]


def observability_summary() -> dict:
    by_level = Counter(f[3].split(" ")[0] for f in FIELDS)
    potential = sum(f[2] for f in FIELDS)
    yes_pts = sum(f[2] for f in FIELDS if f[3] == "YES")
    no_pts = sum(f[2] for f in FIELDS if f[3] == "NO")
    # 各维度：潜在 vs 封顶
    per_dim = {}
    for dim, cap in DIM_CAPS.items():
        pot = sum(f[2] for f in FIELDS if f[0] == dim)
        per_dim[dim] = {"potential": pot, "cap": cap, "overflow": max(0, pot - cap)}
    return {
        "n_fields": len(FIELDS),
        "by_level": dict(by_level),
        "potential_points": potential,
        "capped_total": sum(DIM_CAPS.values()),
        "per_dim": per_dim,
        "fully_observable_points": yes_pts,
        "unobservable_points": no_pts,
        "observable_share": round(yes_pts / sum(DIM_CAPS.values()), 4),
    }


# ---------------------------------------------------------------------------
# B. 描述性统计
# ---------------------------------------------------------------------------

def tri_counts(values: list) -> dict:
    """三态计数：True / False / None(不可判定)。**不可达绝不塌成 False。**"""
    return {
        "true": sum(1 for v in values if v is True),
        "false": sum(1 for v in values if v is False),
        "unknown": sum(1 for v in values if v is None),
    }


def describe(facts: dict) -> dict:
    flat = facts.get("flat") or []
    if not flat:
        return {"verdict": "NO_DATA"}
    n = len(flat)

    def rate(key):
        c = tri_counts([r.get(key) for r in flat])
        denom = c["true"] + c["false"]
        return {
            **c,
            "rate_of_decided": (round(c["true"] / denom, 4) if denom else None),
            "n_decided": denom,
        }

    out = {
        "n_domains": n,
        "endpoint_rates": {k: rate(k) for k in (
            "llms_txt", "llms_full_txt", "has_wellknown_mcp",
            "mcp_json", "dot_mcp_json",
        )},
        "html_rates": {k: rate(k) for k in (
            "has_jsonld", "has_opengraph", "has_canonical", "has_viewport",
        )},
        "robots": {
            "present": rate("robots_present"),
            "sitemap_line": rate("robots_sitemap"),
            "n_ai_agents_named": {
                str(v): sum(1 for r in flat if r.get("robots_names_ai_agents") == v)
                for v in sorted({r.get("robots_names_ai_agents")
                                 for r in flat
                                 if r.get("robots_names_ai_agents") is not None})
            },
        },
        "reachability": {
            "homepage_ok": rate("homepage_ok"),
            "http_to_https": rate("redirects_http_to_https"),
        },
    }

    # MCP 端点：任一路径命中
    mcp_any = [bool(r.get("has_wellknown_mcp") is True or r.get("mcp_json") is True
                     or r.get("dot_mcp_json") is True) for r in flat]
    mcp_unknown = [r.get("has_wellknown_mcp") is None for r in flat]
    out["mcp_any_endpoint"] = {
        "n_with_mcp": sum(1 for v in mcp_any if v),
        "n_without_mcp": sum(1 for r in flat
                             if r.get("has_wellknown_mcp") is False
                             and r.get("mcp_json") is False
                             and r.get("dot_mcp_json") is False),
        "n_undecided": sum(1 for v in mcp_unknown if v),
        "rate_of_all_domains": round(sum(1 for v in mcp_any if v) / n, 4),
        "note": "分母含完全不可达的域 —— 这是**站点状态**的估计，不是 agent 可调用性的估计",
    }

    llms = [r.get("llms_txt") for r in flat]
    out["llms_txt_headline"] = {
        "n_present": sum(1 for v in llms if v is True),
        "n_absent": sum(1 for v in llms if v is False),
        "n_unknown": sum(1 for v in llms if v is None),
        "rate_of_decided": (
            round(sum(1 for v in llms if v is True)
                  / max(1, sum(1 for v in llms if v is not None)), 4)),
    }
    return out


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------

def render(facts: dict) -> str:
    L = []
    A = L.append
    A("# agent 可调用性基础设施现状调查（C3 命题⑤ 的可行性判定）")
    A("")
    m = facts.get("meta", {})
    A(f"- 样本：{m.get('n_domains')} 个域名，探测 UA `{m.get('ua')}`")
    A(f"- 限速：域内间隔 {m.get('min_interval_sec')}s，并发上限 {m.get('max_workers')}")
    A(f"- 完全可达：{m.get('n_fully_reachable')}/{m.get('n_domains')}")
    A(f"- 采集总耗时：{m.get('elapsed_sec')}s")
    A("")

    A("## A. 字段可观测性矩阵 ★ 这一节决定 ⑤ 可不可执行")
    A("")
    s = observability_summary()
    A(f"> **`agent_readiness` 满分 {s['capped_total']} 分里，"
      f"只有 {s['fully_observable_points']} 分（{s['observable_share']:.0%}）能从外部完整验证，"
      f"{s['unobservable_points']} 分（{1 - s['observable_share']:.0%}）是商家内部信息。**")
    A("")
    A("> ⚠️ **分值口径**：下表的分值是「若该项被满足，规则允许它贡献多少」，"
      f"按维度加总为 {s['potential_points']} 分，**大于满分 {s['capped_total']}**，"
      "因为实现用 `min(s, DIM_MAX)` 封顶：")
    A("")
    A("| 维度 | 潜在可得分 | 维度上限 | 被砍掉的溢出 |")
    A("|---|---|---|---|")
    for dim, v in s["per_dim"].items():
        A(f"| `{dim}` | {v['potential']} | {v['cap']} | {v['overflow']} |")
    A("")
    A("> 因此**任何一行都不能单独当成「该站在这项能拿多少分」的保证**。")
    A("")
    A("| 维度 | 字段 | 分值 | 可观测性 | 外部能得到什么 | 备注 |")
    A("|---|---|---|---|---|---|")
    for dim, fld, pts, lvl, ext, note in FIELDS:
        A(f"| `{dim}` | `{fld}` | {pts} | **{lvl}** | {ext} | {note} |")
    A("")
    A("**软信号（按设计 0 分，只进 diagnostics）**")
    A("")
    A("| 信号 | 探测方式 | 分值 | 可观测性 |")
    A("|---|---|---|---|")
    for name, how, pts, lvl in SOFT_SIGNALS:
        A(f"| {name} | {how} | {pts} | {lvl} |")
    A("")
    A("> 🔴 **结论：把不可观测字段填成 `False` 就是伪造。**"
      "agent_readiness 自己拒绝把「没采集过」记成「商家没做到」"
      "（`check()` 在 site 为空时返回 `applicable=false` 而非 0 分）。"
      "所以外部探测能产出的最多是**可观测子集分数**，不是它的完整分。")
    A("")

    d = describe(facts)
    if d.get("verdict") == "NO_DATA":
        A("## B. 描述性统计")
        A("")
        A("**无数据。**")
        A("")
        return "\n".join(L)

    A("## B. 描述性统计：一手现状数据")
    A("")
    A("### B1. agent 专用端点的部署率")
    A("")
    A("| 端点 | 存在 | 不存在 | 不可判定 | 占有判定者的比例 |")
    A("|---|---|---|---|---|")
    for k, v in d["endpoint_rates"].items():
        rr = f"{v['rate_of_decided']:.1%}" if v["rate_of_decided"] is not None else "—"
        A(f"| `{k}` | {v['true']} | {v['false']} | {v['unknown']} | {rr} |")
    A("")
    mcp = d["mcp_any_endpoint"]
    A(f"- **任一 MCP 端点命中：{mcp['n_with_mcp']}/{d['n_domains']} 个域名**"
      f"（占全部 {mcp['rate_of_all_domains']:.1%}）；"
      f"明确无 MCP {mcp['n_without_mcp']} 个；不可判定 {mcp['n_undecided']} 个。")
    A(f"  {mcp['note']}")
    A("")

    ll = d["llms_txt_headline"]
    A("### B2. llms.txt 部署率")
    A("")
    A(f"- 存在 **{ll['n_present']}** / 不存在 {ll['n_absent']} / 不可判定 {ll['n_unknown']}")
    A(f"- 占有判定者的比例：**{ll['rate_of_decided']:.1%}**")
    A("")
    A("> ⚠️ 这只说明「文件存不存在」，**不说明它是否影响引用**。"
      "Google AI Overviews 团队已明确表态 `llms.txt` 不影响引用选择，"
      "Perplexity / OAI-SearchBot / Googlebot 实际几乎不抓取。")
    A("")

    A("### B3. robots.txt 与 AI bot")
    A("")
    r = d["robots"]
    A(f"- robots.txt 存在：{r['present']['true']} / 不存在 {r['present']['false']}"
      f" / 不可判定 {r['present']['unknown']}")
    A(f"- 声明了 Sitemap 行：{r['sitemap_line']['true']}")
    A("- **robots.txt 中点名 AI bot 的数量分布**：")
    A("")
    A("| 点名的 AI bot 个数 | 站点数 |")
    A("|---|---|")
    for k, v in sorted(r["n_ai_agents_named"].items(), key=lambda x: int(x[0])):
        A(f"| {k} | {v} |")
    A("")

    A("### B4. 常规 SEO 结构化数据")
    A("")
    A("| 事实 | 存在 | 不存在 | 不可判定 |")
    A("|---|---|---|---|")
    for k, v in d["html_rates"].items():
        A(f"| `{k}` | {v['true']} | {v['false']} | {v['unknown']} |")
    A("")
    rc = d["reachability"]
    A(f"- 首页 200：{rc['homepage_ok']['true']} / 非 200 或失败 {rc['homepage_ok']['false']}"
      f" + {rc['homepage_ok']['unknown']} 不可判定")
    A(f"- HTTP→HTTPS 跳转：{rc['http_to_https']['true']}")
    A("")
    A("> 🔴 **本次运行的 `redirects_http_to_https` 指标不可信，已作废。**"
      "初版 `12_probe_site_facts.py` 用 `urllib.request.urlopen` 探测 `http://` 入口，"
      "而 urllib **默认自动跟随重定向**，于是拿到的是最终 200 而非第一跳 301，"
      "导致 54 个域名全部报「无跳转」—— 显然是假数据"
      "（主流站点几乎都强制 HTTPS）。"
      "已修：新增 `fetch_raw()` 禁用自动重定向，并记录 `http_first_hop_status` "
      "与 `http_to_https_location`。**该指标需复测后才有意义**，"
      "本次报告不使用它。")
    A("")

    A("## C. 这批数据能回答什么、不能回答什么")
    A("")
    A("**能**：")
    A("1. 「agent 专用端点（llms.txt / MCP）现在到底普及到什么程度」的一手数字。")
    A("2. 「robots.txt 点名放行 AI bot」的实际比例。")
    A("3. **对 ⑤ 可行性的形式化判定**（第 A 节）——这是本节最大的产出。")
    A("")
    A("**不能**：")
    A("1. **不能做任何构念效度结论。**外部只能得 "
      f"{s['fully_observable_points']}/{s['capped_total']} 分的可观测子集，"
      "且其中多数站点该子集为常数（无方差）→ 相关性无法计算。")
    A("2. **不能估计普及率。**样本是研究者凭认知枚举的，有规模/可达性/地域三重偏倚，"
      "见 `path/to/domains.txt` 头部声明。")
    A("3. **不能推断因果。**「有 llms.txt 的站点是否更容易被 agent 引用」需要真实 agent 实验，"
      "本调查完全没触及。")
    A("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------

def selftest() -> int:
    fails = []

    def chk(name, cond):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    print("== 13_infrastructure_survey selftest ==")

    s = observability_summary()
    chk("字段数 = 11", s["n_fields"] == 11)
    chk("潜在分 115 > 满分 100（封顶生效）", s["potential_points"] == 115)
    chk("满分 = 100", s["capped_total"] == 100)
    chk("interpretability 溢出 10", s["per_dim"]["interpretability"]["overflow"] == 10)
    chk("decision_reliability 溢出 5",
        s["per_dim"]["decision_reliability"]["overflow"] == 5)
    chk("executability 无溢出", s["per_dim"]["executability"]["overflow"] == 0)
    chk("完全可观测仅 18 分", s["fully_observable_points"] == 18)
    chk("可观测占比 < 25%", s["observable_share"] < 0.25)
    chk("NO 类字段占多数", s["by_level"].get("NO", 0) >= 5)

    chk("三态计数：True", tri_counts([True, True, False, None])["true"] == 2)
    chk("三态计数：False", tri_counts([True, True, False, None])["false"] == 1)
    chk("三态计数：不可达不塌成 False", tri_counts([True, True, False, None])["unknown"] == 1)

    fake = {"meta": {"n_domains": 4, "ua": "x", "min_interval_sec": 1.0,
                     "max_workers": 4, "n_fully_reachable": 4, "elapsed_sec": 1.0},
            "flat": [
                {"domain": "a.com", "llms_txt": True, "has_wellknown_mcp": False,
                 "mcp_json": False, "dot_mcp_json": False, "has_jsonld": True,
                 "robots_present": True, "robots_names_ai_agents": 2,
                 "homepage_ok": True, "redirects_http_to_https": True},
                {"domain": "b.com", "llms_txt": False, "has_wellknown_mcp": False,
                 "mcp_json": True, "dot_mcp_json": False, "has_jsonld": False,
                 "robots_present": True, "robots_names_ai_agents": 0,
                 "homepage_ok": True, "redirects_http_to_https": True},
                {"domain": "c.com", "llms_txt": None, "has_wellknown_mcp": None,
                 "mcp_json": None, "dot_mcp_json": None, "has_jsonld": None,
                 "robots_present": None, "robots_names_ai_agents": None,
                 "homepage_ok": False, "redirects_http_to_https": None},
                {"domain": "d.com", "llms_txt": False, "has_wellknown_mcp": False,
                 "mcp_json": False, "dot_mcp_json": False, "has_jsonld": True,
                 "robots_present": True, "robots_names_ai_agents": 5,
                 "homepage_ok": True, "redirects_http_to_https": True},
            ]}
    d = describe(fake)
    chk("llms.txt 存在=1 缺失=2 未知=1", d["llms_txt_headline"]["n_present"] == 1
        and d["llms_txt_headline"]["n_absent"] == 2
        and d["llms_txt_headline"]["n_unknown"] == 1)
    chk("MCP 任意端点命中 = 1（b.com）", d["mcp_any_endpoint"]["n_with_mcp"] == 1)
    chk("空输入不崩", describe({"flat": []})["verdict"] == "NO_DATA")
    md = render(fake)
    chk("渲染含可观测性矩阵", "字段可观测性矩阵" in md)
    chk("渲染含三态警告", "不可判定" in md)

    print(f"\n{len(fails)} failed" if fails else "\nall checks passed")
    for f in fails:
        print(f"  FAILED: {f}")
    return 1 if fails else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="13_infrastructure_survey.py",
        description="agent 可调用性基础设施现状调查 + ⑤ 可行性判定")
    p.add_argument("--facts", default=os.environ.get("GEO_SITE_FACTS_JSON", "").strip(),
                   help="站点事实 JSON（默认取环境变量 GEO_SITE_FACTS_JSON）")
    p.add_argument("--out", default=None)
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args(argv)

    if a.selftest:
        return selftest()

    try:
        with open(a.facts, "r", encoding="utf-8-sig") as fh:
            facts = json.load(fh)
    except FileNotFoundError:
        print(f"ERROR: 找不到 {a.facts}", file=sys.stderr)
        return 2

    md = render(facts)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(md + "\n")
        print(f"wrote {a.out}")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
