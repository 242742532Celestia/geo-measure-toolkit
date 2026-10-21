#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
11_construct_validity.py -- agent_readiness 构念效度分析

背景
----
C3 命题⑤：`_geo_core/agent_readiness` 测的到底是「agent 可调用性」这个构念，
还是「我设计它时脑子里想的那个东西」？构念效度必须用**独立判据**回答。

🔴 先说一个必须避开的陷阱（同义反复）
----------------------------------------
`agent_readiness.check()` 的输入是**人工填写的站点声明**。
如果用「按 agent_readiness 的计分规则填声明」的数据去验证 agent_readiness，
得到的相关系数**恒等于 1，且不携带任何信息** ——
这正是本文 C3 命题①（2604.25707 附录 B）要审计的错误形态。
**本工具的设计前提就是：判据必须独立于计分规则。**

效度证据的四种（按可自动验证程度排序）
----------------------------------------
| 类型 | 判据来源 | 能否自动 |
|---|---|---|
| 判别效度 | GEO 主分（`deliverable_100`）—— 两者应当**不**高相关 | ✅ 本项目有 100 SKU |
| 收敛效度 | 独立探测的客观事实维度 | ✅ 需要站点样本 |
| 内部一致性 | 三个子维度两两相关 + Cronbach α | ✅ 需要站点样本 |
| 已知组效度 | 已知「有 MCP」与「无 MCP」的站点分组 | ✅ 需要站点样本 |

⚠️ **硬信号（internal MCP card / tool schema / 商务 feed / 时效声明 / 遗漏防护）
是商家内部信息，外部无法自动探测**。这部分只能靠双人独立审计 + 报一致性，
不能靠自动探测器替代。本工具会把「可自动验证」与「只能人工审计」严格分开。

用法
----
    # 1) 只看判别效度（不需要站点数据，本项目已有 100 SKU）
    python examples/11_construct_validity.py --discriminant

    # 2) 站点样本到位后跑全套
    python examples/11_construct_validity.py --sites site_facts.json

    # 3) 自检（合成数据）
    python examples/11_construct_validity.py --selftest
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
# 🔴 2026-10-21 开源化：包外参考目录改为**可选**的环境变量 `GEO_CORE_PATH`。
#    设了就加进 sys.path；没设就不加 —— 相关检查会**跳过并出声**，不是缺陷。
_geo_core = os.environ.get("GEO_CORE_PATH", "").strip()
if _geo_core and os.path.isdir(_geo_core):
    sys.path.insert(0, _geo_core)

from statpower import r_from_p  # noqa: E402

#: 判别效度的预设门槛（预注册，改之前先改这里并记日期）
DISCRIMANT_MAX_ABS_R = 0.60

#: 收敛效度的预设门槛
CONVERGENCE_MIN_ABS_R = 0.40


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------

def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def stdev(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    n = len(xs)
    if n < 2:
        return float("nan")
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def pearson(xs, ys):
    """返回 (r, n)。n<3 或任一方无方差时 r 为 nan。"""
    pairs = [(a, b) for a, b in zip(xs, ys)
             if a is not None and b is not None
             and not (isinstance(a, float) and math.isnan(a))
             and not (isinstance(b, float) and math.isnan(b))]
    n = len(pairs)
    if n < 3:
        return float("nan"), n
    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    sa, sb = stdev(a), stdev(b)
    if not sa or not sb or math.isnan(sa) or math.isnan(sb):
        return float("nan"), n
    ma, mb = mean(a), mean(b)
    num = sum((x - ma) * (y - mb) for x, y in pairs)
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return (num / den if den else float("nan")), n


def spearman(xs, ys):
    """秩相关（带并列取平均秩）。"""
    pairs = [(a, b) for a, b in zip(xs, ys)
             if a is not None and b is not None
             and not (isinstance(a, float) and math.isnan(a))]
    n = len(pairs)
    if n < 3:
        return float("nan"), n
    ra, rb = _rank([p[0] for p in pairs]), _rank([p[1] for p in pairs])
    return pearson(ra, rb)


def _rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def cronbach_alpha(rows: list[list[float]]) -> float:
    """Cronbach's α。rows = 每个对象一行，每行是各维度分数。"""
    rows = [[v for v in r if v is not None and not (isinstance(v, float) and math.isnan(v))]
            for r in rows]
    k = min((len(r) for r in rows if r), default=0)
    rows = [r[:k] for r in rows if len(r) >= k]
    n = len(rows)
    if n < 3 or k < 2:
        return float("nan")
    item_vars = sum(stdev([r[j] for r in rows]) ** 2 for j in range(k))
    tot = [sum(r) for r in rows]
    tot_var = stdev(tot) ** 2
    if not tot_var:
        return float("nan")
    a = (k / (k - 1)) * (1 - item_vars / tot_var)
    return a


# ---------------------------------------------------------------------------
# 效度分析
# ---------------------------------------------------------------------------

def discriminant_validity(geo_scores, agent_scores, labels=None) -> dict:
    """判别效度：agent 可调用性与 GEO 主分应当**不**高相关。

    论证：两者测的是不同构念。GEO 主分测「内容对生成引擎的可见性」，
    agent 副轴测「agent 能否调用该站点」。若二者 r > 0.6，
    说明副轴没有带来新信息，它就是 GEO 主分的换皮。
    """
    r, n = pearson(geo_scores, agent_scores)
    rs, n2 = spearman(geo_scores, agent_scores)
    mde = None
    if n >= 4 and not math.isnan(r):
        from statpower import mde_pearson_r
        mde = round(mde_pearson_r(n), 4)
    passed = (not math.isnan(r)) and abs(r) <= DISCRIMANT_MAX_ABS_R
    return {
        "pearson_r": None if math.isnan(r) else round(r, 4),
        "spearman_rho": None if math.isnan(rs) else round(rs, 4),
        "n": n,
        "mde_r_at_80pct_power": mde,
        "threshold": DISCRIMANT_MAX_ABS_R,
        "verdict": "PASS（两者不相关 → 是不同构念）" if passed
                   else ("FAIL（高相关 → 副轴可能是 GEO 主分的换皮）"
                         if not math.isnan(r) else "INCONCLUSIVE（样本不足）"),
        "note": (
            "⚠️ 这里的 null 是**要论证的命题**（两者不相关），"
            "所以必须报 MDE：n=" + str(n) + " 时只能检出 |r| >= " + str(mde) +
            "。若实测 |r| 低于该下限，正确表述是「本设计检测不到相关」，"
            "不是「证明不相关」。"
        ),
    }


def convergent_validity(sites: list[dict]) -> dict:
    """收敛效度：agent_readiness 分数与**独立探测的客观事实**维度相关。

    sites 每一项需含：
      - `agent_readiness_score`: 该站点的副轴分（0-100）
      - `probe`: 独立探测得到的客观事实 dict（见 11_probe_site_facts.py 的字段）
    ⚠️ 探测规则必须**独立于** agent_readiness 的计分规则，否则是同义反复。
    """
    if not sites:
        return {"verdict": "NO_DATA"}
    scores = [s.get("agent_readiness_score") for s in sites]
    probe_keys = sorted({k for s in sites for k in (s.get("probe") or {})})
    rows = []
    for k in probe_keys:
        vals = [(s.get("probe") or {}).get(k) for s in sites]
        if all(v is None for v in vals):
            continue
        r, n = pearson(scores, [None if v is None else (1.0 if v else 0.0) for v in vals])
        rows.append({
            "probe_fact": k,
            "r": None if math.isnan(r) else round(r, 4),
            "n": n,
            "hit": (not math.isnan(r)) and abs(r) >= CONVERGENCE_MIN_ABS_R,
        })
    hits = [x for x in rows if x["hit"]]
    return {
        "n_sites": len(sites),
        "n_facts": len(rows),
        "n_convergent": len(hits),
        "threshold": CONVERGENCE_MIN_ABS_R,
        "correlations": rows,
        "verdict": ("PASS（至少一项客观事实与副轴显著相关）" if hits
                    else "FAIL（无任何客观事实与副轴相关）"),
        "caveat": (
            "这是**探索性**结果：探测事实是事后选的，多重比较未校正。"
            "论文里报之前必须预注册要检验哪几条，且明确写出「这是收敛证据，不是因果证据」。"
        ),
    }


def internal_consistency(sites: list[dict]) -> dict:
    """内部一致性：三个子维度两两相关 + Cronbach α。

    ⚠️ α 只在**维度同质**时有意义。agent_readiness 的三维
    （interpretability / executability / decision reliability）
    在概念上**并不同质** —— 它们是三个可独立缺失的能力。
    所以 α 低**不一定是坏事**，这一点必须写进方法里，不能拿 α 当验收标准。
    """
    if not sites:
        return {"verdict": "NO_DATA"}
    dims = ["interpretability", "executability", "decision_reliability"]
    mat = [[d for d in dims if s.get(d) is not None] for s in sites]
    if any(len(m) != len(dims) for m in mat):
        return {"verdict": "INCOMPLETE", "note": "部分站点缺子维度分数"}
    rows = [[s[d] for d in dims] for s in sites]
    pairs = []
    for i in range(len(dims)):
        for j in range(i + 1, len(dims)):
            r, n = pearson([row[i] for row in rows], [row[j] for row in rows])
            pairs.append({
                "pair": f"{dims[i]} × {dims[j]}",
                "r": None if math.isnan(r) else round(r, 4),
                "n": n,
            })
    return {
        "n_sites": len(rows),
        "dimension_correlations": pairs,
        "cronbach_alpha": None if math.isnan(cronbach_alpha(rows))
                           else round(cronbach_alpha(rows), 3),
        "verdict": "COMPUTED",
        "caveat": (
            "⚠️ **不要把 α 当验收标准。**三维在概念上不同质，"
            "α 低只说明它们不是同一个东西测的 —— 对一个多维构念这是正常的。"
            "真正该报的是三对相关系数本身，以及它们解释了哪些差异。"
        ),
    }


def known_groups_validity(sites: list[dict]) -> dict:
    """已知组效度：已知「有 MCP」与「无 MCP」的站点，副轴分应有差别。"""
    if not sites:
        return {"verdict": "NO_DATA"}
    with_mcp = [s["agent_readiness_score"] for s in sites if (s.get("probe") or {}).get("has_wellknown_mcp")]
    without = [s["agent_readiness_score"] for s in sites if not (s.get("probe") or {}).get("has_wellknown_mcp")]
    caveat = (
        "⚠️ 站点不是随机分配的（Kish/析因设计）→ 这只是**分组均值差**，"
        "不是处理效应。**不得写成「有 MCP 让分提高 X 分」。**"
    )
    if len(with_mcp) < 3 or len(without) < 3:
        return {"verdict": "INSUFFICIENT",
                "n_with": len(with_mcp), "n_without": len(without),
                "note": "任一组少于 3 个站点 —— 已知组效度要求两组各 ≥3",
                "caveat": caveat}
    a, b = mean(with_mcp), mean(without)
    sa, sb = stdev(with_mcp), stdev(without)
    se = math.sqrt(sa * sa / len(with_mcp) + sb * sb / len(without))
    t = (a - b) / se if se else float("nan")
    df = len(with_mcp) + len(without) - 2
    return {
        "n_with_mcp": len(with_mcp), "n_without_mcp": len(without),
        "mean_with_mcp": round(a, 2), "mean_without_mcp": round(b, 2),
        "diff": round(a - b, 2),
        "welch_t": None if math.isnan(t) else round(t, 3),
        "df": df,
        "verdict": "COMPUTED",
        "caveat": caveat,
    }


def run_all(sites: list[dict]) -> dict:
    return {
        "convergent": convergent_validity(sites),
        "internal": internal_consistency(sites),
        "known_groups": known_groups_validity(sites),
    }


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------

def selftest() -> int:
    fails = []

    def chk(name, cond):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    print("== 11_construct_validity selftest ==")

    chk("pearson 完全正相关 = 1", abs(pearson([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])[0] - 1.0) < 1e-9)
    chk("pearson 完全负相关 = -1", abs(pearson([1, 2, 3, 4, 5], [10, 8, 6, 4, 2])[0] + 1.0) < 1e-9)
    chk("pearson 无方差 -> nan", math.isnan(pearson([1, 1, 1, 1, 1], [1, 2, 3, 4, 5])[0]))
    r, n = pearson([1, 2], [1, 2])
    chk("n<3 -> nan 且 n 正确", math.isnan(r) and n == 2)
    r3, n3 = pearson([1, 2, 3], [1, 2, 3])
    chk("n=3 正常计算", abs(r3 - 1.0) < 1e-9 and n3 == 3)

    chk("spearman 单调非线性仍 = 1",
        abs(spearman([1, 2, 3, 4, 5], [1, 4, 9, 16, 25])[0] - 1.0) < 1e-9)
    chk("spearman 并列秩取平均",
        all(abs(a - b) < 1e-9
            for a, b in zip(_rank([10, 20, 20, 30]), [1.0, 2.5, 2.5, 4.0])))

    rows = [[10, 10, 10], [20, 20, 20], [30, 30, 30], [40, 40, 40], [50, 50, 50]]
    chk("完美同质 -> alpha = 1", abs(cronbach_alpha(rows) - 1.0) < 1e-6)
    chk("完全无关 -> alpha <= 0", cronbach_alpha([[1, 9, 5], [2, 8, 4], [3, 7, 3]]) <= 0.001)

    d = discriminant_validity([1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1])
    chk("判别效度：完全负相关 -> FAIL", d["verdict"].startswith("FAIL"))
    # 真正无关的数据：正交化的两组
    d2 = discriminant_validity([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                               [3, -1, 4, 1, 5, 9, 2, 6, 5, 3])
    chk("判别效度：低相关 -> PASS", d2["verdict"].startswith("PASS"))
    chk("判别效度带 MDE", d2["mde_r_at_80pct_power"] is not None)
    d3 = discriminant_validity([1, 2], [1, 2])
    chk("判别效度：n 太小 -> INCONCLUSIVE", d3["verdict"].startswith("INCONCLUSIVE"))

    sites = [
        {"agent_readiness_score": 90, "interpretability": 30, "executability": 35,
         "decision_reliability": 25, "probe": {"has_wellknown_mcp": True, "llms_txt": True}},
        {"agent_readiness_score": 80, "interpretability": 28, "executability": 30,
         "decision_reliability": 22, "probe": {"has_wellknown_mcp": True, "llms_txt": False}},
        {"agent_readiness_score": 70, "interpretability": 25, "executability": 26,
         "decision_reliability": 19, "probe": {"has_wellknown_mcp": True, "llms_txt": False}},
        {"agent_readiness_score": 40, "interpretability": 10, "executability": 18,
         "decision_reliability": 12, "probe": {"has_wellknown_mcp": False, "llms_txt": False}},
        {"agent_readiness_score": 30, "interpretability": 8, "executability": 12,
         "decision_reliability": 10, "probe": {"has_wellknown_mcp": False, "llms_txt": False}},
        {"agent_readiness_score": 60, "interpretability": 18, "executability": 24,
         "decision_reliability": 18, "probe": {"has_wellknown_mcp": False, "llms_txt": True}},
    ]
    res = run_all(sites)
    chk("收敛效度已计算", res["convergent"]["verdict"].startswith(("PASS", "FAIL")))
    chk("内部一致性已计算", res["internal"]["verdict"] == "COMPUTED")
    chk("内部一致性带 α 警告", "不要把 α 当验收标准" in res["internal"]["caveat"])
    chk("已知组已计算", res["known_groups"]["verdict"] == "COMPUTED")
    chk("已知组带因果禁令", "不得写成" in res["known_groups"]["caveat"])
    chk("空输入不崩", run_all([])["convergent"]["verdict"] == "NO_DATA")
    chk("收敛效度带多重比较警告", "探索性" in res["convergent"]["caveat"])

    # 6. 反同义反复：判据与计分规则解耦的硬约束
    taut = [{"agent_readiness_score": 90, "interpretability": 30, "executability": 35,
             "decision_reliability": 25, "probe": {"has_wellknown_mcp": True}},
            {"agent_readiness_score": 90, "interpretability": 30, "executability": 35,
             "decision_reliability": 25, "probe": {"has_wellknown_mcp": True}},
            {"agent_readiness_score": 90, "interpretability": 30, "executability": 35,
             "decision_reliability": 25, "probe": {"has_wellknown_mcp": True}}]
    rt = convergent_validity(taut)
    chk("全同分数据 -> 收敛效度给不出 r（不是伪 PASS）",
        all(x["r"] is None for x in rt["correlations"]))

    chk("r_from_p 可用", abs(r_from_p(0.05, 100)) > 0)

    print(f"\n{len(fails)} failed" if fails else "\nall checks passed")
    for f in fails:
        print(f"  FAILED: {f}")
    return 1 if fails else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="11_construct_validity.py",
        description="agent_readiness 构念效度分析（判据独立于计分规则）")
    p.add_argument("--sites", default=None, help="站点事实 JSON 列表")
    p.add_argument("--discriminant", action="store_true",
                   help="只跑判别效度（用 100 SKU 的 GEO 分）")
    p.add_argument("--remeasure-json", default=None,
                   help="判别效度用的重测 JSON（含 items[].before.deliverable_100）。"
                        "默认取环境变量 GEO_REMEASURE_JSON；没有就跳过这条分析")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    if a.selftest:
        return selftest()

    out = {}
    if a.sites:
        with open(a.sites, "r", encoding="utf-8-sig") as fh:
            out["site_based"] = run_all(json.load(fh))
    if a.discriminant:
        # 🔴 2026-10-21 开源化：原来硬编码了作者机器上的绝对路径。
        #    现在走 `--remeasure-json` / 环境变量 `GEO_REMEASURE_JSON`；
        #    没有就**跳过**这条分析并出声（证据不在场 ≠ 功能缺失）。
        src = a.remeasure_json or os.environ.get("GEO_REMEASURE_JSON", "").strip()
        if src and os.path.exists(src):
            with open(src, "r", encoding="utf-8-sig") as fh:
                data = json.load(fh)
            items = data.get("items") or []
            geo = [it.get("before", {}).get("deliverable_100") for it in items]
            out["discriminant"] = {
                "source": src,
                "note": "副轴在 100 SKU 上 applicable=false（无站点基础设施），"
                        "因此判别效度只能用**外部站点样本**做，不能用这批 SKU。",
                "n_items": len(items),
                "geo_score_range": [min([x for x in geo if x is not None], default=None),
                                   max([x for x in geo if x is not None], default=None)],
            }

    payload = json.dumps(out, ensure_ascii=False, indent=2)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(payload + "\n")
        print(f"wrote {a.out}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
