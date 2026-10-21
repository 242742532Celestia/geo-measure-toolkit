#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
geo_stat_lint.py -- GEO 论文的统计声明 lint

定位
----
不主张「某篇做错了」，而主张**错误模式可复现、可形式化、可自动检测**。
本工具读一份「统计声明卡」（JSON，作者手工填的最小信息集），
然后对每条被报出的结论跑一组形式化判据。

为什么是「手工填声明卡」而不是「自动解析 PDF」
------------------------------------------------
自动解析 PDF 的表格在实践中不可靠（多栏、脚注、跨页表、图表混排），
一旦误解析，结论就不可信 —— 而本文的全部论证建立在**数字准确**之上。
手工填卡把「机器的活」限定在**算术**上，数字的来源由作者负责、可逐条核对。
代价是填卡成本，收益是每一个判定都能追到具体字段。

六条判据
--------
L1 no-interval           报了点估计对比但无任何区间
L2 below-noise           组间差异 < 该 outcome 的已测 run-to-run 噪声
L3 rare-binary-r         二元因变量阳性数 < 15% 却报 r，且分母用了展开后的单元数
L4 cluster-as-independent 聚类结构被展开后当作独立单元计入分母
L5 uninterpretable-null  报 null 但无事前功效分析 -> null 不可读作「无效应」
L6 under-identified       报出了分布的少数统计量（mean/max），不足以唯一反解

用法
----
    python tools/geo_stat_lint.py run path/to/audit/*.json
    python tools/geo_stat_lint.py selftest
    python tools/geo_stat_lint.py explain L2

退出码：0 = 无违规；1 = 有违规；2 = 工具/输入错误（便于接 CI）。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from statpower import (  # noqa: E402
    design_effect,
    effective_n,
    mde_pearson_r,
    mde_two_proportions,
    n_required_two_proportions,
    permutation_resolution,
    r_from_p,
)

#: 阴性对照阈值：L3 里「阳性数过少导致精度虚高」的分界
RARE_BINARY_THRESHOLD = 0.15

RULES = {
    "L1": {
        "name": "no-interval",
        "severity": "major",
        "claim": "报了点估计对比但无任何区间 -> 不可判定",
    },
    "L2": {
        "name": "below-noise",
        "severity": "critical",
        "claim": "组间差异 < 该 outcome 的已测 run-to-run 噪声 -> 不可排序",
    },
    "L3": {
        "name": "rare-binary-r",
        "severity": "major",
        "claim": "二元因变量阳性数 < 15% 却报 r，且分母用了展开后的单元数",
    },
    "L4": {
        "name": "cluster-as-independent",
        "severity": "critical",
        "claim": "聚类结构被展开后当作独立单元计入分母 -> 样本量虚高",
    },
    "L5": {
        "name": "uninterpretable-null",
        "severity": "critical",
        "claim": "报 null 但无事前功效分析 -> null 不可读作「无效应」",
    },
    "L6": {
        "name": "under-identified",
        "severity": "minor",
        "claim": "只报分布的少数统计量（mean/max），不足以唯一反解",
    },
}


# ---------------------------------------------------------------------------
# 判据实现：每条返回 (violated: bool, evidence: dict, fix: str)
# ---------------------------------------------------------------------------

def rule_L1(c: dict, card: dict) -> tuple[bool, dict, str]:
    has_point = c.get("delta_pp") is not None or (
        c.get("p1") is not None and c.get("p2") is not None
    )
    has_only_p = c.get("reported_p") is not None and not has_point
    has_interval = c.get("reported_interval") not in (None, "", "none")
    has_ci = c.get("reported_ci") not in (None, "", "none")
    # 只报 p 值不给效应量，同样不可判定：一个 p=0.007 可能是 0.7 也可能是 30
    violated = (has_point or has_only_p) and not (has_interval or has_ci)
    ev = {
        "reported_point_estimate": has_point,
        "p_value_only_no_effect_size": has_only_p,
        "has_interval": has_interval,
        "has_ci": has_ci,
        "has_p_value": c.get("reported_p") is not None,
        "note": "p 值不是区间：它只告诉你『非零』，不告诉你『多大、方向多稳』",
    }
    fix = "至少给 95% 区间（比例用 Wilson，连续用 bootstrap/t 区间），并报重复次数"
    return violated, ev, fix


def rule_L2(c: dict, card: dict) -> tuple[bool, dict, str]:
    sigma = card.get("external_noise_sigma_pp")
    delta = c.get("delta_pp")
    if sigma is None or delta is None:
        return False, {"skipped": "卡片未提供 external_noise_sigma_pp 或 delta_pp"}, ""
    violated = abs(delta) < sigma
    ev = {
        "delta_pp": delta,
        "measured_run_to_run_sigma_pp": sigma,
        "ratio_delta_over_sigma": round(abs(delta) / sigma, 3),
        "verdict": "差异小于 1 个噪声标准差" if violated else "差异达到 1 个噪声标准差",
    }
    fix = f"要么在 {sigma}pp 量级上做多次重复并给误差棒，要么不排名次只报方向"
    return violated, ev, fix


def rule_L3(c: dict, card: dict) -> tuple[bool, dict, str]:
    if c.get("kind") != "correlation":
        return False, {"skipped": "非相关系数类结论"}, ""
    n_reported = c.get("reported_n")
    n_expanded = c.get("expanded_n_used")
    ev = {
        "reported_n": n_reported,
        "expanded_n_used": n_expanded,
        "analysis_unit": c.get("analysis_unit"),
    }
    violated = False
    if n_reported and n_expanded and n_expanded > n_reported:
        violated = True
        ev["n_inflation"] = round(n_expanded / n_reported, 2)
        ev["mde_underestimate_factor"] = round((n_expanded / n_reported) ** 0.5, 3)
    if c.get("reported_p") is not None and n_reported:
        implied = r_from_p(c["reported_p"], n_reported)
        mde = mde_pearson_r(n_reported)
        ev["implied_abs_r"] = round(implied, 4)
        ev["mde_r_at_80pct_power"] = round(mde, 4)
        if mde > implied:
            violated = True
            ev["null_verdict"] = (
                f"该 null 只排除了 |r| >= {mde:.3f}，"
                f"而 p 值本身只对应 |r| = {implied:.3f} —— 结论几乎是空的"
            )
    if c.get("positive_rate") is not None:
        pr = c["positive_rate"]
        ev["positive_rate"] = pr
        if pr < RARE_BINARY_THRESHOLD:
            violated = True
            ev["rare_binary"] = (
                f"二元因变量阳性率 {pr:.1%} < {RARE_BINARY_THRESHOLD:.0%}，"
                f"零膨胀严重，相关系数的有效信息量远低于名义 n"
            )
    fix = "报 n 时用独立分析单元；同时给阳性数/总数，而不是只给 n"
    return violated, ev, fix


def rule_L4(c: dict, card: dict) -> tuple[bool, dict, str]:
    cl = card.get("clustering") or {}
    n_total = c.get("expanded_n_used") or (cl.get("n_units_reported"))
    n_units = c.get("analysis_unit_n")
    ev = {
        "clustering_present": cl.get("present"),
        "n_clusters": cl.get("n_clusters"),
        "cluster_size": cl.get("cluster_size"),
        "icc_reported": cl.get("icc_reported"),
        "expanded_n_used": n_total,
        "independent_unit_n": n_units,
    }
    violated = False
    if n_total and n_units and n_total > n_units:
        violated = True
        infl = n_total / n_units
        ev["n_inflation"] = round(infl, 2)
        ev["mde_underestimate_factor"] = round(infl ** 0.5, 3)
    if cl.get("present") and not cl.get("icc_reported") and cl.get("cluster_size"):
        ev["deff_scan"] = [
            {"icc": row["icc"], "deff": row["deff"]}
            for row in __import__("statpower").design_effect_table(cl["cluster_size"])
        ]
        ev["deff_note"] = "ICC 从未测 -> 设计效应只能给区间，不能给点值；区间本身就足以否定『零修正』"
    if violated:
        ev["analysis_unit"] = c.get("analysis_unit")
    fix = "以独立分析单元（企业/域/格）为分母；或显式建模聚类（GEE / 混合效应 / cluster-robust SE）"
    return violated, ev, fix


def rule_L5(c: dict, card: dict) -> tuple[bool, dict, str]:
    if not c.get("is_null_claim"):
        return False, {"skipped": "非 null 结论"}, ""
    has_power = card.get("power_analysis_present") or c.get("power_analysis_present")
    ev = {
        "reported_p": c.get("reported_p"),
        "power_analysis_present": bool(has_power),
    }
    violated = not has_power
    if violated and c.get("reported_p") is not None and c.get("analysis_unit_n"):
        n = c["analysis_unit_n"]
        r = r_from_p(c["reported_p"], n)
        mde = mde_pearson_r(n)
        ev["implied_abs_r"] = round(r, 4)
        ev["mde_r_at_80pct_power"] = round(mde, 4)
        ev["verdict"] = (
            f"n={n} 只能检出 |r| >= {mde:.3f}；"
            f"报告的 p={c['reported_p']} 对应 |r|={r:.3f}，落在检测下限以下 "
            f"{mde / r:.1f} 倍处"
        )
    ev["interpretation"] = "无功效分析时，null 只能读作『本设计检测不到』，不能读作『无效应』"
    fix = "报出本研究在该 n 下的 MDE；null 结论必须与 MDE 成对出现"
    return violated, ev, fix


def rule_L6(c: dict, card: dict) -> tuple[bool, dict, str]:
    d = c.get("distribution_reported")
    if not d:
        return False, {"skipped": "非分布类报告"}, ""
    need = ("quantiles", "sd")
    # ⚠️ 关键：值为 null 的 key 等于**没报**。只查 key 是否存在会把
    #    "quantiles": null 误判成已报告，从而漏放最典型的违规写法。
    reported = {k: d.get(k) for k in need}
    missing = [k for k, v in reported.items() if v in (None, "", "none")]
    violated = len(missing) > 0
    ev = {
        "reported_keys": sorted(d.keys()),
        "null_valued_keys_counted_as_missing": True,
        "missing_keys": missing,
        "shape_identifiability": (
            "只给 mean 与 max 无法唯一确定分布形状，"
            "因此功效反解只能给条件结论，不能给点值"
        ),
    }
    if "n_candidates" in d and d["n_candidates"]:
        ev["permutation_resolution"] = permutation_resolution(d["n_candidates"])
    fix = "补报分位数（至少 p5/p25/p50/p75/p95）与 SD；置换检验另报候选数"
    return violated, ev, fix


RULE_FUNCS = {
    "L1": rule_L1, "L2": rule_L2, "L3": rule_L3,
    "L4": rule_L4, "L5": rule_L5, "L6": rule_L6,
}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def lint_card(card: dict) -> dict:
    findings = []
    for c in card.get("claims", []):
        cid = c.get("id", "<unnamed>")
        for rid, fn in RULE_FUNCS.items():
            violated, ev, fix = fn(c, card)
            if violated:
                findings.append({
                    "rule": rid,
                    "rule_name": RULES[rid]["name"],
                    "severity": RULES[rid]["severity"],
                    "rule_claim": RULES[rid]["claim"],
                    "claim_id": cid,
                    "claim_text": c.get("text", ""),
                    "evidence": ev,
                    "fix": fix,
                })
    sev_order = {"critical": 0, "major": 1, "minor": 2}
    findings.sort(key=lambda f: (sev_order.get(f["severity"], 9), f["claim_id"]))
    counts = {s: sum(1 for f in findings if f["severity"] == s)
              for s in ("critical", "major", "minor")}
    return {
        "paper": card.get("paper"),
        "title": card.get("title"),
        "venue": card.get("venue"),
        "n_claims": len(card.get("claims", [])),
        "n_findings": len(findings),
        "severity_counts": counts,
        "verdict": "CLEAN" if not findings else "VIOLATIONS",
        "findings": findings,
    }


def _md_table(rows: list[list], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def render_markdown(reports: list[dict]) -> str:
    lines = ["# GEO 统计声明 lint 报告", ""]
    total = sum(r["n_findings"] for r in reports)
    lines.append(
        f"扫描 {len(reports)} 份声明卡，"
        f"共 {sum(r['n_claims'] for r in reports)} 条被报结论，"
        f"触发 **{total}** 条判据。"
    )
    lines.append("")
    lines.append(_md_table(
        [[r["paper"], r["venue"], r["n_claims"],
          r["severity_counts"]["critical"], r["severity_counts"]["major"],
          r["severity_counts"]["minor"], r["verdict"]] for r in reports],
        ["论文", "状态", "结论数", "critical", "major", "minor", "判定"],
    ))
    lines.append("")
    for r in reports:
        if r["verdict"] == "CLEAN":
            lines.append(f"## {r['paper']} — CLEAN（0 违规）")
            lines.append("")
            continue
        lines.append(f"## {r['paper']} — {r['n_findings']} 条违规")
        lines.append("")
        for f in r["findings"]:
            tag = f"[{f['severity'].upper()}] {f['rule']} {f['rule_name']}"
            lines.append(f"### {tag}")
            lines.append("")
            lines.append(f"- **触发结论**：{f['claim_id']} — {f['claim_text']}")
            lines.append(f"- **判据**：{f['rule_claim']}")
            for k, v in f["evidence"].items():
                if k == "deff_scan":
                    continue
                lines.append(f"  - `{k}` = {v}")
            if "deff_scan" in f["evidence"]:
                sc = f["evidence"]["deff_scan"]
                lines.append(f"  - 设计效应扫描（ICC 未知）：{sc}")
            lines.append(f"- **修复**：{f['fix']}")
            lines.append("")
    return "\n".join(lines)


def explain(rid: str, quiet: bool = False) -> int:
    if rid not in RULES:
        if not quiet:
            print(f"unknown rule: {rid}\navailable: {', '.join(RULES)}")
        return 2
    if quiet:
        return 0
    m = RULES[rid]
    print(f"{rid}  {m['name']}  [{m['severity']}]\n")
    print(f"  判据：{m['claim']}\n")
    helps = {
        "L1": "对应案例：2603.09296 表 2（11 臂排完整名次，全文零区间）",
        "L2": "对应案例：2603.09296 表 2 中 9 个臂（0.74–5.86pp < 6.77pp 噪声）",
        "L3": "对应案例：2601.00912（per-query 784 口径 vs 112 个独立产品）",
        "L4": "对应案例：2607.12056（15 格 × 10 次 = 300 按独立单元计入）",
        "L5": "对应案例：2601.00912 的 p=0.256（零功效分析下报 null）",
        "L6": "对应案例：2606.04362（placebo 分布只给 mean 0.24 / max 0.65）",
    }
    if rid in helps:
        print(f"  已收录实例：{helps[rid]}")
    return 0


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------

def selftest(verbose: bool = True) -> int:
    fails = []

    def chk(name, cond):
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    print("== geo_stat_lint selftest ==")

    # 1. 干净卡片必须零违规
    clean = {
        "paper": "T-clean", "venue": "test",
        "external_noise_sigma_pp": 6.77,
        "power_analysis_present": True,
        "clustering": {"present": False},
        "claims": [{
            "id": "c1", "text": "A vs B", "delta_pp": 20.0,
            "reported_interval": "[12.1, 27.9]", "reported_p": 0.001,
        }],
    }
    r = lint_card(clean)
    chk("干净卡片零违规", r["verdict"] == "CLEAN")

    # 2. 同一张卡，去掉区间 -> 触发 L1
    dirty = json.loads(json.dumps(clean))
    dirty["claims"][0]["reported_interval"] = None
    r2 = lint_card(dirty)
    chk("去区间触发 L1", any(f["rule"] == "L1" for f in r2["findings"]))

    # 3. 差异 < 噪声 -> 触发 L2
    small = json.loads(json.dumps(clean))
    small["claims"][0]["delta_pp"] = 5.86
    small["claims"][0]["reported_interval"] = None
    r3 = lint_card(small)
    chk("5.86pp < 6.77pp 触发 L2", any(f["rule"] == "L2" for f in r3["findings"]))

    # 4. 分母膨胀 -> 触发 L3 + L4
    infl = json.loads(json.dumps(clean))
    infl["claims"][0] = {
        "id": "corr", "text": "r", "kind": "correlation",
        "reported_p": 0.256, "reported_n": 112, "expanded_n_used": 784,
        "analysis_unit": "product", "analysis_unit_n": 112, "reported_interval": None,
    }
    r4 = lint_card(infl)
    ids4 = {f["rule"] for f in r4["findings"]}
    chk("分母膨胀触发 L3", "L3" in ids4)
    chk("分母膨胀触发 L4", "L4" in ids4)
    l3 = next(f for f in r4["findings"] if f["rule"] == "L3")
    chk("L3 证据含 mde_r=0.258",
        abs(l3["evidence"].get("mde_r_at_80pct_power", 0) - 0.258) < 0.002)

    # 5. 聚类未测 ICC -> 给出扫描表
    clus = json.loads(json.dumps(clean))
    clus["clustering"] = {"present": True, "n_clusters": 15, "cluster_size": 10,
                          "icc_reported": None}
    clus["claims"][0] = {
        "id": "chi", "text": "300 次", "reported_interval": None,
        "expanded_n_used": 300, "analysis_unit_n": 15,
    }
    r5 = lint_card(clus)
    l4f = next((f for f in r5["findings"] if f["rule"] == "L4"), None)
    chk("聚类触发 L4", l4f is not None)
    chk("L4 给出 ICC 扫描表", l4f is not None and "deff_scan" in l4f["evidence"])
    chk("DEFF(10, ICC=0.5)=5.5 在扫描表里",
        l4f is not None and 5.5 in [row["deff"] for row in l4f["evidence"]["deff_scan"]])

    # 6. null 无功效 -> 触发 L5
    nullc = json.loads(json.dumps(clean))
    nullc["power_analysis_present"] = False
    nullc["claims"][0] = {
        "id": "null", "text": "无相关", "kind": "correlation",
        "is_null_claim": True, "reported_p": 0.256,
        "analysis_unit_n": 112, "reported_interval": None,
    }
    r6 = lint_card(nullc)
    chk("null 无功效触发 L5", any(f["rule"] == "L5" for f in r6["findings"]))
    l5 = next(f for f in r6["findings"] if f["rule"] == "L5")
    chk("L5 证据含 MDE 口径", "verdict" in l5["evidence"])

    # 7. 只给 mean/max -> 触发 L6
    dist = json.loads(json.dumps(clean))
    dist["claims"][0] = {
        "id": "pl", "text": "placebo", "reported_interval": None,
        "distribution_reported": {"mean": 0.24, "max": 0.65, "n_candidates": 26},
    }
    r7 = lint_card(dist)
    chk("只给 mean/max 触发 L6", any(f["rule"] == "L6" for f in r7["findings"]))
    l6 = next(f for f in r7["findings"] if f["rule"] == "L6")
    chk("L6 算出 26 候选最小 p=0.0385",
        abs(l6["evidence"]["permutation_resolution"]["min_p_conservative"] - 1 / 26) < 1e-6)

    # 7b. 回归防护：key 存在但值为 null 不得被当成「已报告」
    nullval = json.loads(json.dumps(clean))
    nullval["claims"][0] = {
        "id": "pl2", "text": "placebo", "reported_interval": None,
        "distribution_reported": {"mean": 0.24, "max": 0.65, "n_candidates": 26,
                                  "quantiles": None, "sd": None},
    }
    r7b = lint_card(nullval)
    chk("quantiles:null / sd:null 仍触发 L6（防空值冒充）",
        any(f["rule"] == "L6" for f in r7b["findings"]))
    l6b = next(f for f in r7b["findings"] if f["rule"] == "L6")
    chk("L6 正确列出两个缺失键", set(l6b["evidence"]["missing_keys"]) == {"quantiles", "sd"})

    # 7c. 真正报全了分位数与 SD -> 不触发 L6
    full = json.loads(json.dumps(clean))
    full["claims"][0] = {
        "id": "pl3", "text": "placebo", "reported_interval": None,
        "distribution_reported": {"mean": 0.24, "sd": 0.18, "n_candidates": 26,
                                  "quantiles": [0.05, 0.25, 0.5, 0.75, 0.95]},
    }
    r7c = lint_card(full)
    chk("分位数与 SD 都报全则不触发 L6",
        not any(f["rule"] == "L6" for f in r7c["findings"]))

    # 7d. 回归防护：只报 p 值不给效应量也必须触发 L1
    ponly = json.loads(json.dumps(clean))
    ponly["claims"][0] = {
        "id": "ponly", "text": "只报 p", "reported_p": 0.007, "reported_interval": None,
    }
    r7d = lint_card(ponly)
    chk("只报 p 值不给效应量触发 L1", any(f["rule"] == "L1" for f in r7d["findings"]))
    l1d = next(f for f in r7d["findings"] if f["rule"] == "L1")
    chk("L1 标记 p_value_only", l1d["evidence"].get("p_value_only_no_effect_size") is True)

    # 8. 严重度排序：critical 在前
    r8 = lint_card(infl)
    sev = [f["severity"] for f in r8["findings"]]
    chk("critical 排在 major 前", sev == sorted(sev, key=lambda s: {"critical": 0, "major": 1, "minor": 2}[s]))

    # 9. explain 对每条规则都能输出（静默模式）
    for rid in RULES:
        chk(f"explain {rid}", explain(rid, quiet=True) == 0)
    chk("explain 未知规则返回 2", explain("L99", quiet=True) == 2)

    print(f"\n{len(fails)} failed" if fails else "\nall checks passed")
    for f in fails:
        print(f"  FAILED: {f}")
    return 1 if fails else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="geo_stat_lint.py",
        description="GEO 论文统计声明 lint：把功效缺陷变成可自动检测的判据",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("run", help="对声明卡跑 lint")
    s.add_argument("cards", nargs="+", help="JSON 路径，支持 glob")
    s.add_argument("--json", action="store_true", help="输出 JSON 而非 markdown")
    s.add_argument("--out", default=None, help="写入文件")

    sub.add_parser("selftest", help="运行内置自检")

    s = sub.add_parser("explain", help="解释某条判据")
    s.add_argument("rule", help=", ".join(RULES))

    a = p.parse_args(argv)

    if a.cmd == "selftest":
        return selftest()
    if a.cmd == "explain":
        return explain(a.rule)

    paths: list[str] = []
    for pattern in a.cards:
        if any(ch in pattern for ch in "*?["):
            paths.extend(sorted(glob.glob(pattern)))
        else:
            paths.append(pattern)

    reports, errors = [], []
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8-sig") as fh:
                reports.append(lint_card(json.load(fh)))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path}: {exc}")

    if errors:
        for e in errors:
            print(f"ERROR {e}", file=sys.stderr)
        if not reports:
            return 2

    payload = (json.dumps(reports, ensure_ascii=False, indent=2) if a.json
               else render_markdown(reports))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(payload + "\n")
        print(f"wrote {a.out}")
    else:
        print(payload)

    return 1 if any(r["n_findings"] for r in reports) else 0


if __name__ == "__main__":
    sys.exit(main())
