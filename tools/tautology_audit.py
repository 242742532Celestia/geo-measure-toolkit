#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tautology_audit.py -- 同义反复审计器（C3 命题①）

一句话
------
**检查一篇论文的因果实验里，处理臂是否机械地改变了主因变量自己的组成部分。**
若是，则「处理臂 > 对照」有一部分是**定义使然**，不是发现。

形式化
------
设主因变量是一个加权复合：

    Y = Σᵢ wᵢ · fᵢ(x),        Σᵢ wᵢ = 1

设处理臂 T 机械改变的成分集合为 A(T)（例如「仅加证据块」必然抬高段落覆盖率）。
定义**泄漏度**

    L(T, Y) = Σ_{j ∈ A(T)} wⱼ  /  Σᵢ wᵢ

L 越大，Y 在处理臂之间的差异越有可能是**构造性的**（trivially true），
越不能当作「机制起作用」的证据。

分档（预设，改之前先改这里并记日期）
----------------------------------------
    L = 0        干净。主因变量可作因果 estimand
    0 ≤ L < 0.30 轻微。必须讨论，不能默认当作发现
    0.30 ≤ L < 0.60 部分泄漏。主因变量须重排或做成分位移校正
    L ≥ 0.60     严重。**该复合禁止作为主因变量**

🔴 为什么要分开「分解」与「泄漏」
-------------------------------
把某个成分 f_k **单独**当因变量，测的是「这个成分本身变没变」——那是**分解**，合法。
把 Y 当因变量、而 T 机械操纵 f_k，测的是「一个部分由构造决定的量变没变」——那是**泄漏**。
**两者的区别只在于 f_k 是不是因变量本身。** 这个区分是本审计的全部判据。

用法
----
    python tools/tautology_audit.py --spec path/to/tautology_spec.json
    python tools/tautology_audit.py --selftest
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List

# 泄漏度分档（预注册）。左闭右开，边界值一律归入更严的一档。
#   L = 0          -> CLEAN
#   0 < L < 0.30   -> MINOR
#   0.30 ≤ L < 0.60-> PARTIAL
#   L ≥ 0.60       -> SEVERE
BANDS = [
    (0.60, "SEVERE", "严重。该复合禁止作为主因变量"),
    (0.30, "PARTIAL", "部分泄漏。主因变量须重排或做成分位移校正"),
    (0.0, "MINOR", "轻微。必须讨论，不能默认当作发现"),
]


def band(L: float) -> Dict[str, str]:
    if L <= 0.0:
        return {"band": "CLEAN", "advice": "干净。主因变量可作因果 estimand"}
    if L < 0.30:
        return {"band": "MINOR", "advice": "轻微。必须讨论，不能默认当作发现"}
    if L < 0.60:
        return {"band": "PARTIAL", "advice": "部分泄漏。主因变量须重排或做成分位移校正"}
    return {"band": "SEVERE", "advice": "严重。该复合禁止作为主因变量"}


# ---------------------------------------------------------------------------
# 核心
# ---------------------------------------------------------------------------

def leak(outcome: Dict[str, Any], arm: Dict[str, Any]) -> Dict[str, Any]:
    """单个处理臂相对某个 outcome 的泄漏度。

    outcome = {"name":..., "components": {成分名: 权重}, "note":...}
    arm     = {"name":..., "mechanically_affects": [成分名], ...}

    只统计 components 中真实存在且权重可解析的成分；
    arm 声明了 outcome 里没有的成分 -> 记为 warning 而不是静默忽略。
    """
    comps: Dict[str, float] = outcome.get("components") or {}
    if not comps:
        return {"error": "outcome 没有 components"}

    total = sum(float(v) for v in comps.values())
    if total <= 0:
        return {"error": "权重和为 0"}

    hits, unknown, wsum = [], [], 0.0
    for c in arm.get("mechanically_affects", []) or []:
        if c in comps:
            hits.append(c)
            wsum += float(comps[c])
        else:
            unknown.append(c)

    L = wsum / total
    b = band(L)
    return {
        "outcome": outcome.get("name"),
        "arm": arm.get("name"),
        "L": round(L, 4),
        "L_pct": round(L * 100, 2),
        "hit_components": hits,
        "hit_weight": round(wsum, 4),
        "unknown_components": unknown,
        "band": b["band"],
        "advice": b["advice"],
        "verdict": (
            "该臂与主因变量部分同源 —— 「此臂优于对照」有一部分是定义使然，"
            "不能整体读作机制证据"
            if L > 0 else
            "该臂不机械改变主因变量的任何成分 —— 干净"
        ),
    }


def audit(spec: Dict[str, Any]) -> Dict[str, Any]:
    outcome = spec["outcome"]
    arms = spec.get("arms", [])
    rows = [leak(outcome, a) for a in arms]
    leaks = [r["L"] for r in rows if "L" in r]
    worst = max(leaks) if leaks else 0.0
    names = [r["arm"] for r in rows if r.get("L") == worst and worst > 0]

    # 逃生口：审计另一个候选 outcome
    escapes = []
    for alt in spec.get("alternative_outcomes", []) or []:
        rows_alt = [leak(alt, a) for a in arms]
        la = [r["L"] for r in rows_alt if "L" in r]
        escapes.append({
            "outcome": alt.get("name"),
            "max_L": round(max(la), 4) if la else 0.0,
            "max_L_pct": round(max(la) * 100, 2) if la else 0.0,
            "band": band(max(la) if la else 0.0)["band"],
            "rationale": alt.get("note"),
        })

    return {
        "case": spec.get("case"),
        "source": spec.get("source"),
        "outcome_audited": outcome.get("name"),
        "outcome_weights": outcome.get("components"),
        "n_arms": len(arms),
        "rows": rows,
        "max_leak": round(worst, 4),
        "max_leak_pct": round(worst * 100, 2),
        "worst_arms": names,
        "overall_band": band(worst)["band"],
        "alternatives": escapes,
        "verdict": spec.get("expected_verdict", ""),
    }


# ---------------------------------------------------------------------------
# 横向调查：L 判据在已核实的一批论文里适用面有多大？
#
# ⚠️ 本表全部数字来自 `path/to/notes/` 里**已全文核实**的记录，不是推断。
#    每一行的 outcome 形态都能在对应笔记里逐条对上。
# ---------------------------------------------------------------------------

SURVEY = [
    {"paper": "2604.25707", "venue": "preprint", "arm": "五臂受控改写（附录 B）",
     "outcome": "influence_score (Eq 2)", "form": "weighted_composite",
     "applicable": True,
     "max_L": 0.65, "band": "SEVERE",
     "note": "唯一用加权复合作主因变量且有处理臂的一篇"},
    {"paper": "2603.09296", "venue": "preprint", "arm": "11 臂策略对比（表 2）",
     "outcome": "citation rate", "form": "binary_event",
     "applicable": False, "max_L": None, "band": "N/A",
     "note": "二元事件不是加权复合 -> **L 不适用**（它的风险在别处：L2 噪声）"},
    {"paper": "2607.12056", "venue": "ICEME 2026 接收", "arm": "agent-ready vs baseline",
     "outcome": "clear recommendation (PASS)", "form": "binary_event",
     "applicable": False, "max_L": None, "band": "N/A",
     "note": "二元事件 -> L 不适用（风险是 L1 无区间 + L4 聚类当独立）"},
    {"paper": "2601.00912", "venue": "M.Tech 学位论文", "arm": "**无处理臂**（观察性）",
     "outcome": "GEO 分数（6 维 0-100 平均）", "form": "unweighted_mean",
     "applicable": False, "max_L": None, "band": "N/A",
     "note": "无处理臂 -> 无 A(T) 可算；且 6 维是等权平均而非加权复合"},
    {"paper": "2606.04362", "venue": "preprint", "arm": "**无处理臂**（ITS 自然实验）",
     "outcome": "ITS 水平断点 β2", "form": "regression_coef",
     "applicable": False, "max_L": None, "band": "N/A",
     "note": "回归系数不是复合量 -> L 不适用（风险是 L5 零功效分析）"},
]


def render_survey() -> str:
    L = []
    A = L.append
    A("# 同义反复审计：L 判据的横向适用性")
    A("")
    A("| 论文 | 状态 | 主因变量 | outcome 形态 | L 适用? | 最大 L | 分档 |")
    A("|---|---|---|---|---|---|---|")
    for r in SURVEY:
        Ls = f"**{r['max_L']:.0%}**" if r["max_L"] is not None else "—"
        A(f"| `{r['paper']}` | {r['venue']} | {r['outcome']} | `{r['form']}` | "
          f"{'✅ 是' if r['applicable'] else '— 否'} | {Ls} | "
          f"{'**`%s`**' % r['band'] if r['applicable'] else 'N/A'} |")
    A("")
    n_app = sum(1 for r in SURVEY if r["applicable"])
    A(f"## 结论（这比单案例的结果更重要）")
    A("")
    A(f"**5 篇已全文核实的论文里，只有 {n_app} 篇适用 L 判据。**")
    A("")
    A("原因不是判据太严，而是**大部分论文的 outcome 根本不是加权复合**：")
    A("")
    A("- `binary_event`（2 篇）：二元事件只有一个「有没有」，不含可被操纵的成分 → L 无定义")
    A("- **无处理臂**（2 篇）：`A(T) = ∅` → L ≡ 0，但没有东西可审计（这两篇的风险在别处）")
    A("- `regression_coef`：回归系数不是复合量")
    A("")
    A("> 🔴 **所以诚实的定位是：L 判据覆盖面很窄，但那一篇恰好是风险最高的一篇。**")
    A("> ")
    A("> 这个风险是**由 2604.25707 自己的指标构造引入的** —— 它是这批文献里")
    A("> 唯一构建了加权复合指标（Eq 2）的人。别人用二元事件，结构上就躲开了这个坑。")
    A("> ")
    A("> **这削弱了「这是领域普遍问题」的叙述，增强了「这是该文特有的问题」的叙述。**")
    A("> 论文里必须这样写，不能把它包装成对全领域的指控。")
    A("")
    A("## 对每篇的实际风险（各归各的判据）")
    A("")
    A("| 论文 | 它的真实风险 | 归属判据 |")
    A("|---|---|---|")
    A("| `2604.25707` | 附录 B 的两个核心臂 L=65% | **L（本判据）** |")
    A("| `2603.09296` | 9/11 臂间差 < 已测噪声，零区间 | L2 + L1 |")
    A("| `2607.12056` | 300 次当独立、全文无 CI | L4 + L1 |")
    A("| `2601.00912` | 零功效分析下报 null，口径混用 | L5 + L3 |")
    A("| `2606.04362` | 无 MDE，placebo 分布不可识别 | L5 + L6 |")
    A("")
    A("> **五条不同判据、五篇不同论文、五个不同错法。**")
    A("> 这比「同义反复是普遍问题」更难被反驳，因为它承认了错误的**多样性**。")
    A("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# 内置阳性/阴性对照（2604.25707 附录 B）
# ---------------------------------------------------------------------------

EQ2 = {
    "name": "influence_score (Eq 2)",
    "components": {
        "min(ref_count/3,1)": 0.20,
        "1-first_position_ratio": 0.15,
        "paragraph_coverage_ratio": 0.20,
        "tfidf_cosine": 0.25,
        "ngram_overlap": 0.20,
    },
    "note": "absorption 复合量；内容项（paragraph/tfidf/ngram）合计 0.65",
}

ARMS_B = [
    {"name": "original page", "mechanically_affects": []},
    {"name": "structured headings only",
     "mechanically_affects": ["paragraph_coverage_ratio", "ngram_overlap"]},
    {"name": "evidence blocks only",
     "mechanically_affects": ["paragraph_coverage_ratio", "tfidf_cosine", "ngram_overlap"]},
    {"name": "semantic-intent alignment only", "mechanically_affects": ["tfidf_cosine"]},
    {"name": "full evidence-container rewrite",
     "mechanically_affects": ["paragraph_coverage_ratio", "tfidf_cosine", "ngram_overlap"]},
]

BUILTIN = {
    "case": "2604.25707 附录 B 的五臂受控改写设计",
    "source": "arXiv:2604.25707v2 附录 B（笔记 15 §4）",
    "outcome": EQ2,
    "arms": ARMS_B,
    "alternative_outcomes": [
        {"name": "citation probability",
         "components": {},
         "note": "**完全在 Eq 2 之外**（属于 selection 层）-> L 恒为 0"},
        {"name": "first_position_ratio",
         "components": {"first_position_ratio": 1.0},
         "note": "是 Eq 2 的成分，但**单独**作因变量是分解不是泄漏"},
    ],
    "expected_verdict": "SEVERE：证据块臂与完整改写臂的 L = 0.65",
}


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------

def render(rep: Dict[str, Any]) -> str:
    L = []
    A = L.append
    A(f"# 同义反复审计：{rep['case']}")
    A("")
    A(f"- 来源：{rep.get('source', '—')}")
    A(f"- 受审 outcome：**{rep['outcome_audited']}**")
    A(f"- 权重：{rep['outcome_weights']}")
    A("")
    A("## 各臂泄漏度")
    A("")
    A("| 处理臂 | 机械影响的成分 | 权重和 | **L** | 分档 |")
    A("|---|---|---|---|---|")
    for r in rep["rows"]:
        A(f"| `{r['arm']}` | {'、'.join(r['hit_components']) or '—'} | "
          f"{r['hit_weight']} | **{r['L_pct']:.1f}%** | `{r['band']}` |")
    A("")
    A(f"- **最大泄漏度 L = {rep['max_leak_pct']:.1f}%** → `{rep['overall_band']}`")
    if rep["worst_arms"]:
        A(f"- 最严重的臂：{', '.join('`' + a + '`' for a in rep['worst_arms'])}")
    A("")
    if rep.get("alternatives"):
        A("## 逃生口：换成哪个 outcome 才干净")
        A("")
        A("| 候选 outcome | 最大 L | 分档 | 理由 |")
        A("|---|---|---|---|")
        for e in rep["alternatives"]:
            A(f"| `{e['outcome']}` | **{e['max_L_pct']:.1f}%** | `{e['band']}` | {e['rationale']} |")
        A("")
    A("## 判读")
    A("")
    A(f"{rep.get('verdict', '')}")
    A("")
    A("> **注意分档不等于指控。** 同一篇论文若把 `influence_score` 只当**描述性共同结果**、"
      "把主因变量换成 Eq 2 之外的量，那么它的设计就是干净的 —— "
      "**同一份数据，取决于怎么定义 estimand，结论完全不同。**")
    A("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------

def selftest() -> int:
    fails = []

    def chk(name, cond, extra=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {extra}")
        if not cond:
            fails.append(name)

    print("== tautology_audit selftest ==")

    # 1. 分档边界
    chk("L=0 -> CLEAN", band(0.0)["band"] == "CLEAN")
    chk("L=0.25 -> MINOR", band(0.25)["band"] == "MINOR")
    chk("L=0.30 -> PARTIAL", band(0.30)["band"] == "PARTIAL")
    chk("L=0.59 -> PARTIAL", band(0.59)["band"] == "PARTIAL")
    chk("L=0.60 -> SEVERE", band(0.60)["band"] == "SEVERE")
    chk("L=0.65 -> SEVERE", band(0.65)["band"] == "SEVERE")
    chk("L=1.0 -> SEVERE", band(1.0)["band"] == "SEVERE")

    # 2. 基本算术
    oc = {"name": "y", "components": {"a": 0.5, "b": 0.3, "c": 0.2}}
    r0 = leak(oc, {"name": "t0", "mechanically_affects": []})
    chk("空臂 L=0", r0["L"] == 0.0 and r0["band"] == "CLEAN")
    r1 = leak(oc, {"name": "t1", "mechanically_affects": ["a"]})
    chk("命中 a -> L=0.5", abs(r1["L"] - 0.5) < 1e-9)
    r2 = leak(oc, {"name": "t2", "mechanically_affects": ["a", "b"]})
    chk("命中 a,b -> L=0.8", abs(r2["L"] - 0.8) < 1e-9)
    r3 = leak(oc, {"name": "t3", "mechanically_affects": ["zzz"]})
    chk("未知成分记 warning 不静默", r3["unknown_components"] == ["zzz"] and r3["L"] == 0.0)

    # 3. 空 components -> citation probability 的情形
    r4 = leak({"name": "cp", "components": {}},
              {"name": "t", "mechanically_affects": ["a"]})
    chk("空 components 返回 error 而非崩溃", "error" in r4)

    # 4. 阳性对照：2604.25707 附录 B
    rep = audit(BUILTIN)
    by = {r["arm"]: r for r in rep["rows"]}
    chk("original page L=0", by["original page"]["L"] == 0.0)
    chk("structured headings L=0.40", abs(by["structured headings only"]["L"] - 0.40) < 1e-9,
        f"got {by['structured headings only']['L']}")
    chk("evidence blocks L=0.65", abs(by["evidence blocks only"]["L"] - 0.65) < 1e-9,
        f"got {by['evidence blocks only']['L']}")
    chk("semantic-intent L=0.25", abs(by["semantic-intent alignment only"]["L"] - 0.25) < 1e-9)
    chk("full rewrite L=0.65", abs(by["full evidence-container rewrite"]["L"] - 0.65) < 1e-9)
    chk("整体 SEVERE", rep["overall_band"] == "SEVERE")

    # 5. 逃生口
    alts = {a["outcome"]: a for a in rep["alternatives"]}
    chk("citation probability L=0 -> CLEAN",
        alts["citation probability"]["max_L"] == 0.0
        and alts["citation probability"]["band"] == "CLEAN")
    chk("first_position_ratio 作为单独因变量 CLEAN（分解非泄漏）",
        alts["first_position_ratio"]["band"] == "CLEAN")

    # 6. 阴性对照：把 outcome 换成在 Eq 2 之外的东西，全部臂 L=0
    clean_spec = dict(BUILTIN, outcome={"name": "citation_probability",
                                         "components": {"cited": 1.0}})
    rep2 = audit(clean_spec)
    chk("阴性对照：全部臂 L=0 -> CLEAN",
        rep2["max_leak"] == 0.0 and rep2["overall_band"] == "CLEAN")

    # 7. 渲染不崩
    md = render(rep)
    chk("渲染含泄漏度表", "各臂泄漏度" in md)
    chk("渲染含逃生口", "逃生口" in md)
    chk("渲染声明分档不等于指控", "分档不等于指控" in md)

    # 8. 横向调查
    chk("SURVEY 覆盖 5 篇", len(SURVEY) == 5)
    chk("只有 1 篇适用 L",
        sum(1 for r in SURVEY if r["applicable"]) == 1)
    chk("唯一适用者是 2604.25707",
        [r["paper"] for r in SURVEY if r["applicable"]] == ["2604.25707"])
    chk("两篇是无处理臂",
        sum(1 for r in SURVEY if "无处理臂" in r["arm"]) == 2)
    chk("survey 渲染含诚实降调",
        "削弱了「这是领域普遍问题」的叙述" in render_survey())
    chk("survey 渲染含五判据归属",
        render_survey().count("L2") >= 1 and render_survey().count("L6") >= 1)

    print(f"\n{len(fails)} failed" if fails else "\nall checks passed")
    for f in fails:
        print(f"  FAILED: {f}")
    return 1 if fails else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="tautology_audit.py",
        description="同义反复审计：处理臂是否机械改变了主因变量的组成部分")
    p.add_argument("--spec", default=None, help="审计声明 JSON")
    p.add_argument("--builtin", action="store_true",
                   help="跑内置的 2604.25707 附录 B 阳性对照")
    p.add_argument("--survey", action="store_true",
                   help="输出 L 判据在已核实论文里的横向适用性")
    p.add_argument("--out", default=None)
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args(argv)

    if a.selftest:
        return selftest()

    if a.survey:
        md = render_survey()
        if a.out:
            with open(a.out, "w", encoding="utf-8") as fh:
                fh.write(md + "\n")
            print(f"wrote {a.out}")
        else:
            print(md)
        return 0

    if a.spec:
        try:
            with open(a.spec, "r", encoding="utf-8-sig") as fh:
                spec = json.load(fh)
        except FileNotFoundError:
            print(f"ERROR: 找不到 {a.spec}", file=sys.stderr)
            return 2
    elif a.builtin:
        spec = BUILTIN
    else:
        spec = BUILTIN

    rep = audit(spec)
    md = render(rep)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(md + "\n")
        with open(a.out.replace(".md", ".json"), "w", encoding="utf-8") as fh:
            json.dump(rep, fh, ensure_ascii=False, indent=2)
        print(f"wrote {a.out}")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
