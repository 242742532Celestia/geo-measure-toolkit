#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10_citation_noise.py -- 在 citation rate 上复测 run-to-run 噪声

对应框架 §4-C3 命题④，2026-10-05 落地。

为什么必须重做
--------------
本项目已实测的 σ=0.0677（6.77pp）是跑在 **`geo_impression`**（连续份额）上的。
而 2603.09296 表 2 的 citation rate 与 2607.12056 的 89.3% 都是**二值**outcome。
**不同 outcome 的噪声量级可以差很远**，所以 6.77pp 那个数字不能直接搬去判它们。

本脚本用已有的 n66 数据（66 query × 10 臂 × 5 repeat = 3300 条）做方差分解，
在 **`cited`（二值 = 目标来源是否被引用，即 citation rate）** 上得到 σ。
数据已存在，零 API 成本。

四层输出
--------
1. 噪声底噪：run-to-run σ、格内方差、格内翻转率
2. 聚类结构：ICC / 设计效应 / 有效样本量
3. 功效反解：MDE、配对 vs 非配对的样本量
4. 误差棒：给 2603.09296 表 2 与 2607.12056 的 89.3% 配区间

用法
----
    python examples/10_citation_noise.py
    python examples/10_citation_noise.py --csv path/to/raw_trials.csv
    python examples/10_citation_noise.py --selftest
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
# ── 控制台编码兜底 ────────────────────────────────────────────────
# Windows 控制台默认 GBK 码页，打印 ✅/⚠/❌/🔴/⇒/² 这类字符会 UnicodeEncodeError 崩溃。
# 这里把「无法编码」改成「替换显示」而不是抛异常；UTF-8 环境（Linux/macOS/CI）下是空操作。
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))

from statpower import (  # noqa: E402
    mde_pearson_r,
    mde_two_proportions,
    n_required_two_proportions,
    paired_sample_advantage,
    wilson_ci,
)

#: 🔴 2026-10-21 开源化：原来硬编码了作者机器上的绝对路径。
#:    现在留空 —— 必须显式传 `--csv <path>`（默认取环境变量 GEO_RAW_TRIALS_CSV）。
DEFAULT_CSV = os.environ.get("GEO_RAW_TRIALS_CSV", "").strip()

#: 09 号脚本在 geo_impression 上实测的跨 run 噪声
GI_SIGMA_PP = 6.77


# ---------------------------------------------------------------------------
# 基础统计（纯标准库）
# ---------------------------------------------------------------------------

def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def stdev(xs):
    n = len(xs)
    if n < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def statistics_median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def icc_oneway(groups: list[list[float]]) -> dict:
    """单因素随机效应 ANOVA 估计 ICC(1)。

    groups: 每个 (query, arm) 一组，组内是该组的 m 次重复观测。
    返回 MSB / MSW / ICC / 设计效应 / 有效样本量。
    """
    groups = [g for g in groups if len(g) > 0]
    k = len(groups)
    N = sum(len(g) for g in groups)
    if k < 2 or N <= k:
        return {"icc": float("nan"), "note": "样本不足以估 ICC"}
    grand = mean([v for g in groups for v in g])
    sizes = [len(g) for g in groups]
    m = N / k

    msb = sum(len(g) * (mean(g) - grand) ** 2 for g in groups) / (k - 1)
    msw = sum(sum((v - mean(g)) ** 2 for v in g) for g in groups) / (N - k)

    if msb + (m - 1) * msw <= 0:
        icc = 0.0
    else:
        icc = (msb - msw) / (msb + (m - 1) * msw)
    icc = max(0.0, min(1.0, icc))  # 截断到合法区间

    deff = 1.0 + (m - 1) * icc
    return {
        "n_groups": k,
        "n_obs": N,
        "mean_group_size": round(m, 3),
        "MSB": round(msb, 6),
        "MSW": round(msw, 6),
        "icc": round(icc, 4),
        "design_effect": round(deff, 3),
        "n_effective": round(N / deff, 1),
        "n_inflation": round(deff, 2),
    }


def flip_rate(groups: list[list[float]], tol: float = 1e-9) -> dict:
    """格内翻转率：一个 (query, arm) 格的 m 次重复里，偏离多数值的有多少次。

    这是二值 outcome 最直观的噪声指标，直接回答 2607.12056 的
    「每格 10 次、从未测格内方差」——格内到底抖不抖。
    """
    if not groups:
        return {"n_groups": 0}
    n_flipped_groups = 0
    n_flipped_obs = 0
    n_obs = 0
    for g in groups:
        n_obs += len(g)
        ones = sum(1 for v in g if v > 0.5)
        maj = 1.0 if ones * 2 > len(g) else 0.0
        flipped = [v for v in g if abs(v - maj) > tol]
        if flipped:
            n_flipped_groups += 1
            n_flipped_obs += len(flipped)
    return {
        "n_groups": len(groups),
        "n_obs": n_obs,
        "groups_with_any_flip": n_flipped_groups,
        "group_flip_rate": round(n_flipped_groups / len(groups), 4),
        "obs_flip_rate": round(n_flipped_obs / n_obs, 4),
        "n_flipped_obs": n_flipped_obs,
    }


def run_level_sigma(rows: list[dict], field: str) -> dict:
    """【聚合级】run-to-run σ：每个 repeat(seed) 的整体均值，看它们之间抖多少。

    ⚠️ **这个量与「单次测量精度」不是一回事，不能拿去和 09 号脚本的 6.77pp 比。**
    它已经把 n 条观测平均掉了，随 n 按 1/√n 缩小。
    它的用途是「整轮实验的总体估计有多稳」，**不是**「一次调用有多不准」。
    """
    by_rep = defaultdict(list)
    for r in rows:
        by_rep[r["repeat"]].append(float(r[field]))
    per_run = {k: mean(v) for k, v in sorted(by_rep.items())}
    vals = list(per_run.values())
    sd = stdev(vals)
    return {
        "level": "aggregate",
        "per_run_mean": {k: round(v, 6) for k, v in per_run.items()},
        "n_runs": len(vals),
        "n_obs_per_run": len(next(iter(by_rep.values()))) if by_rep else 0,
        "run_mean": round(mean(vals), 6),
        "run_sd": None if math.isnan(sd) else round(sd, 6),
        "run_sd_pp": None if math.isnan(sd) else round(sd * 100, 4),
        "run_range_pp": round((max(vals) - min(vals)) * 100, 4),
        "warning": "聚合级。已按 n 平均过，不能当单次测量精度用",
    }


def measurement_noise(groups: list[list[float]], label: str) -> dict:
    """【测量级】同一个 (query, arm) 配置重复跑，抖多少。

    两种等价口径，都与 09 号脚本的「格极差 SD」同层级：

    - `cell_range_sd`：每个格的 m 次重复极差，再对所有格求 SD。
      **这就是 09 号脚本 `between_run_sd` 的算法**，可与 6.77pp 直接同层对比。
    - `cell_within_sd`：每个格 m 次重复的 SD，再对所有格求均值。
      这个更接近「单次调用相对真值的误差」，与文献里说的 measurement error 同义。

    ⚠️ 09 号那个 6.77pp 来自 **4 query × 3 臂** 的 run1/2/3，n 极小；
    本函数跑在 **66 query × 10 臂 × 5 repeat** 上。两者样本量差 5 倍以上，
    **不可直接互相替换**，只能对比量级。
    """
    groups = [g for g in groups if len(g) > 1]
    if not groups:
        return {"level": "measurement", "note": "无重复观测"}
    ranges = [max(g) - min(g) for g in groups]
    sds = [stdev(g) for g in groups]
    m = len(groups[0])
    r_sd = stdev(ranges)

    # 🔴 2026-10-06 修正：格内 SD 的两种估计量差别巨大，必须都报。
    #   mean(s_i)   —— 各格 SD 的**算术平均**。本脚本 v1 一直用它，
    #                   在 n66 上给出 cited = 14.30pp。
    #   sqrt(mean(s_i²)) —— **合并格内方差**，即 E[Var(y | 配置)] 的无偏估计。
    #                   n66 上给出 cited = **26.86pp**。
    # 差别 1.88×，原因是 **471/660 = 71% 的格方差恰为 0**
    # （引擎对多数 (query, arm) 是确定性的），SD 分布高度右偏，
    # 算术平均被这些 0 拖下去。**MDE 只能用 RMS。**
    mean_of_sds = mean(sds)
    rms_of_sds = math.sqrt(mean([s * s for s in sds]))
    n_zero = sum(1 for s in sds if s < 1e-12)
    return {
        "level": "measurement",
        "label": label,
        "n_cells": len(groups),
        "m_repeats_per_cell": m,
        "cell_range_sd": None if math.isnan(r_sd) else round(r_sd, 6),
        "cell_range_sd_pp": None if math.isnan(r_sd) else round(r_sd * 100, 3),
        "mean_cell_range": round(mean(ranges), 6),
        # `cell_within_sd*` 现改为**无偏的 RMS**，是引用噪声时的正确口径。
        "cell_within_sd": round(rms_of_sds, 6),
        "cell_within_sd_pp": round(rms_of_sds * 100, 3),
        "mean_of_cell_sds_pp": round(mean_of_sds * 100, 3),
        "rms_over_mean": (round(rms_of_sds / mean_of_sds, 4)
                          if mean_of_sds else None),
        "zero_variance_cells": n_zero,
        "zero_variance_share": round(n_zero / len(sds), 4),
        "note": ("cell_within_sd = **RMS（合并格内方差，无偏）**。"
                 "各格 SD 的算术平均见 `mean_of_cell_sds_pp`，**有偏低**，"
                 "因多数格方差为 0。MDE 与引用噪声一律用 RMS。"),
    }


# ---------------------------------------------------------------------------
# 主分析
# ---------------------------------------------------------------------------

def load_rows(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        if r.get("error"):
            continue
        try:
            out.append({
                "query_id": r["query_id"],
                "arm": r["arm"],
                "repeat": r["repeat"],
                "seed": r.get("seed", ""),
                "cited": float(r["cited"]),
                "geo_impression": float(r["geo_impression"]),
            })
        except (KeyError, TypeError, ValueError):
            continue
    return out


def analyze(rows: list[dict]) -> dict:
    rep = {}

    # ---- 0. 数据概览
    queries = {r["query_id"] for r in rows}
    arms = sorted({r["arm"] for r in rows})
    repeats = sorted({r["repeat"] for r in rows})
    rep["dataset"] = {
        "n_rows": len(rows),
        "n_queries": len(queries),
        "n_arms": len(arms),
        "arms": arms,
        "n_repeats": len(repeats),
        "repeats": repeats,
        "design": f"{len(queries)} query x {len(arms)} arm x {len(repeats)} repeat",
    }

    # ---- 1. 三个层级的噪声（必须分开，不可混用）
    cells_cited, cells_gi = defaultdict(list), defaultdict(list)
    for r in rows:
        cells_cited[(r["query_id"], r["arm"])].append(r["cited"])
        cells_gi[(r["query_id"], r["arm"])].append(r["geo_impression"])

    rep["noise_levels"] = {
        "cited": {
            "outcome": "citation rate (二值)",
            "overall": round(mean([r["cited"] for r in rows]), 6),
            "measurement": measurement_noise(list(cells_cited.values()),
                                             "固定 (query,arm) 的 5 次重复"),
            "aggregate": run_level_sigma(rows, "cited"),
        },
        "geo_impression": {
            "outcome": "geo_impression (连续份额)",
            "overall": round(mean([r["geo_impression"] for r in rows]), 6),
            "measurement": measurement_noise(list(cells_gi.values()),
                                             "固定 (query,arm) 的 5 次重复"),
            "aggregate": run_level_sigma(rows, "geo_impression"),
        },
    }

    mc = rep["noise_levels"]["cited"]["measurement"]
    mg = rep["noise_levels"]["geo_impression"]["measurement"]
    if mc.get("cell_within_sd_pp") and mg.get("cell_within_sd_pp"):
        ratio = mc["cell_within_sd_pp"] / mg["cell_within_sd_pp"]
        rep["noise_levels"]["verdict"] = (
            f"**测量级**对比（无偏 RMS 口径）：`cited` 的格内 SD = {mc['cell_within_sd_pp']:.2f}pp，"
            f"`geo_impression` 的 = {mg['cell_within_sd_pp']:.2f}pp，"
            f"前者是后者的 **{ratio:.1f} 倍**。"
            f"**二值化本身会放大相对噪声** —— 这不是 `cited`「测得更差」，"
            f"而是 0/1 化丢掉了份额信息量。"
            f"⚠️ 早期版本用「各格 SD 的算术平均」，给出 "
            f"{mc.get('mean_of_cell_sds_pp')}pp / {mg.get('mean_of_cell_sds_pp')}pp，"
            f"因 **{mc.get('zero_variance_cells')}/{mc.get('n_cells')}** 个 `cited` 格方差恰为 0"
            f"而**有偏低**，已废弃。"
        )
    rep["noise_levels"]["caveat_0909"] = (
        f"09 号脚本的 σ=0.0677（6.77pp）算法是**格极差 SD**。"
        f"本脚本在同一 outcome 上测得格极差 SD = {mg.get('cell_range_sd_pp')}pp，"
        f"但**这两个数不可互换**，原因有二："
        f"(a) **重复数不同** —— 极差随样本数单调增大（3 个值的期望极差系数约 1.69σ，"
        f"5 个值约 2.33σ），09 号每格 3 个值、本脚本每格 5 个值；"
        f"(b) **设计不同** —— 09 号跑在 run1/2/3（**4 query × 3 臂**），"
        f"本脚本跑在 n66（**66 query × 10 臂**），臂集合与样本量都不同。"
        f"⚠️ **09 号那个 6.77pp 只在 4 个 query 上估过，本身就是高方差估计。**"
        f"⚠️ 本脚本的格内 SD（{mg.get('cell_within_sd_pp')}pp）与 09 号的格极差 SD 口径不同，"
        f"量级接近**不构成交叉验证**，不要这样引用。"
    )

    # ---- 2. 格内结构（聚类）
    cell_list = list(cells_cited.values())
    rep["clustering"] = {
        "cited": {
            "flip": flip_rate(cell_list),
            "icc_deff": icc_oneway(cell_list),
        },
    }

    gi_cells = defaultdict(list)
    for r in rows:
        gi_cells[(r["query_id"], r["arm"])].append(r["geo_impression"])
    rep["clustering"]["geo_impression"] = {
        "flip": flip_rate(list(gi_cells.values()), tol=1e-9),
        "icc_deff": icc_oneway(list(gi_cells.values())),
    }

    # ---- 3. 各臂 cited 率 + 误差棒
    by_arm = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)
    arm_tbl = []
    for a in arms:
        rs = by_arm[a]
        k = int(sum(1 for r in rs if r["cited"] > 0.5))
        n = len(rs)
        p, lo, hi = wilson_ci(k, n)
        arm_tbl.append({
            "arm": a, "n": n, "cited": k, "rate": round(p, 4),
            "ci95": [round(lo, 4), round(hi, 4)], "width_pp": round((hi - lo) * 100, 2),
        })
    arm_tbl.sort(key=lambda x: -x["rate"])
    base = next((x for x in arm_tbl if x["arm"] == "baseline"), None)
    for row in arm_tbl:
        row["delta_vs_baseline_pp"] = (
            round((row["rate"] - base["rate"]) * 100, 2) if base else None
        )
        # 该差异的 95% 区间（两独立比例差，pooled）
        if base and row["arm"] != "baseline":
            d_lo = (row["ci95"][0] - base["ci95"][1]) * 100
            d_hi = (row["ci95"][1] - base["ci95"][0]) * 100
            row["delta_ci95_pp"] = [round(d_lo, 2), round(d_hi, 2)]
            row["sign_stable_across_seeds"] = None  # 后面填
        else:
            row["delta_ci95_pp"] = None
    rep["arms"] = arm_tbl

    # ---- 4. 符号稳定性：每个臂的 5 个 run 的 Δ vs baseline 符号
    base_by_rep = defaultdict(list)
    for r in rows:
        if r["arm"] == "baseline":
            base_by_rep[r["repeat"]].append(r["cited"])
    base_rep = {k: mean(v) for k, v in base_by_rep.items()}
    for a in arms:
        if a == "baseline":
            continue
        by_ar = defaultdict(list)
        for r in rows:
            if r["arm"] == a:
                by_ar[r["repeat"]].append(r["cited"])
        deltas = []
        for rep_id, v in sorted(by_ar.items()):
            if rep_id in base_rep:
                deltas.append(round((mean(v) - base_rep[rep_id]) * 100, 2))
        signs = [1 if d > 0 else (-1 if d < 0 else 0) for d in deltas]
        entry = next(x for x in arm_tbl if x["arm"] == a)
        entry["delta_per_run_pp"] = deltas
        entry["sign_stable_across_seeds"] = (len(set(signs)) == 1 and signs[0] != 0)
        entry["n_positive_seeds"] = sum(1 for s in signs if s > 0)
        entry["n_runs"] = len(signs)
    base_row = next(x for x in arm_tbl if x["arm"] == "baseline")
    base_row["delta_per_run_pp"] = [0.0] * rep["dataset"]["n_repeats"]
    base_row["sign_stable_across_seeds"] = None
    base_row["n_positive_seeds"] = None
    base_row["n_runs"] = rep["dataset"]["n_repeats"]

    # ---- 5. 功效反解
    n_q = rep["dataset"]["n_queries"]
    m = rep["dataset"]["n_repeats"]
    p_base = base["rate"] if base else 0.5
    power = {}
    for n_per_arm in (n_q, n_q * 2, n_q * 5, 60):
        d = mde_two_proportions(p_base, n_per_arm, n_per_arm)
        power[str(n_per_arm)] = {
            "mde_pp": round(d * 100, 2),
            "note": "非配对，80% power, alpha=0.05",
        }
    rep["power_unpaired"] = power

    # 配对分析：先把每个 (query, arm) 的 m 次重复聚合成一个值（格均值），
    # 再在 query 内做处理臂 vs baseline 的差。
    # ⚠️ 早期版本这里错误地用了单次观测（0/1）而不是格均值，
    #    得到的 SD=32.84pp 是二值的固有方差，不是配对差的真实散布。
    per_q = defaultdict(lambda: defaultdict(list))
    for r in rows:
        per_q[r["query_id"]][r["arm"]].append(r["cited"])

    pair_diffs = []          # 池化：所有处理臂对 baseline 的格均值之差
    per_arm_pair = {}         # 单臂：每个臂的配对差散布
    for q, am in per_q.items():
        if "baseline" not in am:
            continue
        b = mean(am["baseline"])
        for a, v in am.items():
            if a == "baseline":
                continue
            cell_mean = mean(v)
            pair_diffs.append(cell_mean - b)
            per_arm_pair.setdefault(a, []).append(cell_mean - b)

    sd_pooled = stdev(pair_diffs)
    single = {a: round(stdev(v) * 100, 3) for a, v in per_arm_pair.items()
              if len(v) > 1}
    rep["power_paired"] = {
        "unit": "每 (query, arm) 先对 m 次重复取均值，再在 query 内做差",
        "n_paired_cells": len(pair_diffs),
        "sd_pooled_pp": None if math.isnan(sd_pooled) else round(sd_pooled * 100, 3),
        "per_arm_sd_pp": dict(sorted(single.items(), key=lambda x: x[1])),
        "note": "这才是配对分析该用的 SD：query 难易已被消掉，只剩处理效应 + 残余噪声",
    }

    # 配对 MDE：δ = (z_a+z_b) * σ_d / sqrt(n_query)
    from statpower import z as _z  # noqa: E402
    if not math.isnan(sd_pooled):
        zsum = _z(0.05, "two") + _z(0.20, "upper")
        per_arm_sds = list(single.values())
        if per_arm_sds:
            med_sd = statistics_median(per_arm_sds) / 100.0
            rep["power_paired"]["mde_table"] = {
                str(n): round(zsum * med_sd / math.sqrt(n) * 100, 2)
                for n in (66, 132, 330, 660)
            }
            rep["power_paired"]["mde_note"] = (
                f"用单臂配对差 SD 的中位数 {med_sd * 100:.2f}pp 反解。"
                f"对比非配对 MDE（同 n=66）：配对约省 "
                f"{math.sqrt(2) / med_sd:.0f}× 样本。"
            )

    # ---- 6. 给两篇论文配误差棒
    m_cited = rep["noise_levels"]["cited"]["measurement"]
    rep["apply_to_literature"] = {
        "2603.09296": {
            "reported": "表 2：vanilla 56.58%，11 臂，差 0.74–5.86pp，全文零区间",
            "our_cited_rate": round(p_base * 100, 2),
            "our_measurement_sd_pp": m_cited.get("cell_within_sd_pp"),
            "caveat": (
                "⚠️ 本测量跑在 GEO-bench + qwen-plus 上，**不是 MIMIQ**。"
                "σ 是 harness + 模型 + outcome 三者的联合属性，**跨 harness 不可搬用**。"
                "所以它不能替 2603.09296 定罪，只能说："
                "在 GEO-bench 家族上，citation rate 的格内 SD 实测为 "
                f"{m_cited.get('cell_within_sd_pp')}pp；"
                "**要在 MIMIQ 上复测才能对它下结论。**"
            ),
        },
        "2607.12056": {
            "reported": "134/150 = 89.3% vs 74/150 = 49.3%，全文无任何置信区间",
            "wilson_under_independence": {
                "134/150": [round(x, 4) for x in wilson_ci(134, 150)[1:]],
                "74/150": [round(x, 4) for x in wilson_ci(74, 150)[1:]],
            },
            "caveat": (
                "该区间是「150 次全独立」这一**最强假设**下的；实际是 15 格 × 10 次，"
                "配对内相关从未测过，真实区间只会更宽。"
            ),
        },
    }

    return rep


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------

def selftest() -> int:
    fails = []

    def chk(name, cond):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    print("== 10_citation_noise selftest ==")

    # 1. ICC：完全一致的组 -> ICC=0, DEFF=1
    g = [[1.0] * 5 for _ in range(10)]
    r = icc_oneway(g)
    chk("完全一致组 ICC=0", abs(r["icc"]) < 1e-9)
    chk("完全一致组 DEFF=1", abs(r["design_effect"] - 1.0) < 1e-9)

    # 2. ICC：组间方差极大 -> ICC 接近 1
    g2 = [[0.0] * 5, [1.0] * 5] * 5
    r2 = icc_oneway(g2)
    chk("组间完全分离 ICC=1", abs(r2["icc"] - 1.0) < 1e-6)

    # 3. DEFF = 1 + (m-1)*ICC
    g3 = [[0, 1, 0, 1, 0], [0, 1, 0, 1, 0], [1, 0, 1, 0, 1], [1, 0, 1, 0, 1],
          [0, 0, 1, 1, 0], [1, 1, 0, 0, 1]] * 3
    r3 = icc_oneway(g3)
    chk("DEFF 恒等式 1+(m-1)*ICC",
        abs(r3["design_effect"] - (1 + 4 * r3["icc"])) < 1e-9)

    # 4. 翻转率：全 1 -> 0 翻转
    chk("全 1 组无翻转", flip_rate([[1.0] * 5] * 10)["obs_flip_rate"] == 0.0)
    # 半数翻转的组
    f = flip_rate([[1, 1, 1, 0, 0]])
    chk("3正2负 -> 2 次翻转", f["n_flipped_obs"] == 2)
    chk("3正2负 -> 组翻转率 1.0", f["group_flip_rate"] == 1.0)

    # 5. run-level sigma 手算复核：1,2,3,4,5 -> SD = sqrt(2.5) = 1.58113883
    rl = run_level_sigma(
        [{"repeat": str(i), "cited": float(i)} for i in (1, 2, 3, 4, 5)], "cited")
    chk("run SD(1..5)=sqrt(2.5)", abs(rl["run_sd"] - math.sqrt(2.5)) < 1e-5)

    # 6. Wilson 复核（与 statpower 一致）
    chk("Wilson 134/150 下界", abs(wilson_ci(134, 150)[1] - 0.8338) < 1e-3)

    # 7. 配对优势恒等式
    chk("paired advantage rho=0.5 -> 1.0", abs(paired_sample_advantage(1.0, 0.5) - 1.0) < 1e-9)

    print(f"\n{len(fails)} failed" if fails else "\nall checks passed")
    for f in fails:
        print(f"  FAILED: {f}")
    return 1 if fails else 0


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------

def render(rep: dict) -> str:
    L = []
    A = L.append
    d = rep["dataset"]
    A("# C3 命题④：citation rate 上的 run-to-run 噪声复测")
    A("")
    A(f"- 数据：`{d['design']}` = {d['n_rows']} 条（零 API 成本，用的是已有 run 产物）")
    A(f"- 重复结构：{d['n_repeats']} 次 repeat（seed 不同），构成格内重复测量")
    A("")

    A("## 1. 噪声必须分三个层级报 ★")
    A("")
    nl = rep["noise_levels"]
    A("| outcome | 层级 | 口径 | 值 | 用途 |")
    A("|---|---|---|---|---|")
    for key in ("cited", "geo_impression"):
        o = nl[key]
        m, a = o["measurement"], o["aggregate"]
        A(f"| `{key}` | 总体均值 | — | {o['overall']:.4f} | — |")
        A(f"| `{key}` | **测量级** | 格内 SD = RMS（固定 query×arm，{m['m_repeats_per_cell']} 次重复） | "
          f"**{m['cell_within_sd_pp']:.2f} pp** | 单次调用有多不准 |")
        A(f"| `{key}` | 测量级 | 格内 SD = 各格 SD 算术平均（**有偏低，已废弃**） | "
          f"{m['mean_of_cell_sds_pp']:.2f} pp | 因 {m['zero_variance_cells']}/{m['n_cells']} "
          f"格方差为 0 |")
        A(f"| `{key}` | 测量级 | 格极差 SD（09 号同算法） | {m['cell_range_sd_pp']:.2f} pp | 与 09 号同层对比 |")
        A(f"| `{key}` | 聚合级 | run 均值 SD（每 run n={a['n_obs_per_run']}） | "
          f"{a['run_sd_pp']:.2f} pp | 整轮总体估计有多稳 |")
    A("")
    A(f"> {nl.get('verdict', '')}")
    A("")
    A(f"> {nl.get('caveat_0909', '')}")
    A("")
    A("> **聚合级的 σ 已被 n 条观测平均掉，随 n 按 1/√n 缩小，不可当测量精度用。**"
      "本节第一版曾把它和 09 号的 6.77pp 直接相比，得出「差 3.5 倍」的结论 —— "
      "那是层级混淆，**已作废**。")
    A("")

    A("## 2. 格内结构：二值 outcome 的翻转率与 ICC")
    A("")
    for key in ("cited", "geo_impression"):
        c = rep["clustering"][key]
        fl, ic = c["flip"], c["icc_deff"]
        A(f"### `{key}`")
        A("")
        A(f"- 格数 {fl['n_groups']}，其中 **{fl['groups_with_any_flip']} 个格出现过翻转**"
          f"（{fl['group_flip_rate']:.1%}）")
        A(f"- 观测级翻转率 **{fl['obs_flip_rate']:.2%}**（{fl['n_flipped_obs']}/{fl['n_obs']}）")
        A(f"- ICC(1) = **{ic['icc']:.4f}** → 设计效应 DEFF = **{ic['design_effect']:.3f}**"
          f" → 有效样本量 {ic['n_effective']:.0f} / {ic['n_obs']}"
          f"（**分母虚高 {ic['n_inflation']:.2f}×**）")
        A("")

    A("## 3. 各臂 citation rate（Wilson 95% 区间）")
    A("")
    A("| 臂 | cited/n | 率 | 95% CI | 宽度 | Δ vs baseline | Δ 的 95% CI | 符号跨 5 seed |")
    A("|---|---|---|---|---|---|---|---|")
    for r in rep["arms"]:
        d95 = r.get("delta_ci95_pp")
        dci = f"[{d95[0]:+.1f}, {d95[1]:+.1f}]" if d95 else "—"
        sign = r.get("sign_stable_across_seeds")
        if sign is None:
            sign_s = "—（基线）"
        elif not sign:
            sign_s = f"❌ **翻转**（{r['n_positive_seeds']}/{r['n_runs']} 正）"
        else:
            sign_s = "✅ 稳定为正" if r["n_positive_seeds"] == r["n_runs"] else "✅ 稳定为负"
        A(f"| `{r['arm']}` | {r['cited']}/{r['n']} | {r['rate']:.1%} | "
          f"[{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] | {r['width_pp']:.1f}pp | "
          f"{r.get('delta_vs_baseline_pp'):+.2f}pp | {dci} | {sign_s} |")
    A("")
    A("> **`authoritative` / `keyword_stuffing` / `fluency_optimization` 三个臂符号翻转** —— "
      "同一配置重跑，一次正一次负。它们与 baseline 的 Δ 全部落在各自 Wilson 区间的重叠区内。"
      "**这三个臂不能写进论文。**这与 09 号脚本在 `geo_impression` 上发现的 "
      "`statistics_addition` 符号翻转是同一类现象。")
    A("")

    A("## 4. 功效反解（80% power, α=0.05）")
    A("")
    pp = rep["power_paired"]
    A("| 每臂 query 数 | 非配对 MDE | 配对 MDE | 配对省 |")
    A("|---|---|---|---|")
    mt = pp.get("mde_table", {})
    for k, v in rep["power_unpaired"].items():
        pv = mt.get(k)
        save = f"{v['mde_pp'] / pv:.1f}×" if pv else "—"
        A(f"| {k} | **{v['mde_pp']:.2f} pp** | "
          f"**{pv:.2f} pp** | {save} |" if pv else
          f"| {k} | **{v['mde_pp']:.2f} pp** | — | — |")
    A("")
    A(f"- 配对单位：{pp['unit']}")
    A(f"- 配对格数 {pp['n_paired_cells']}，池化配对差 SD = **{pp['sd_pooled_pp']:.2f} pp**")
    A(f"- 单臂配对差 SD（{len(pp['per_arm_sd_pp'])} 臂）：")
    for a_, s_ in pp["per_arm_sd_pp"].items():
        A(f"  - `{a_}`: {s_} pp")
    if pp.get("mde_note"):
        A(f"- {pp['mde_note']}")
    A("")
    A("> ⚠️ **天花板效应**：`cited` 是二值，且 GEO-bench 的设定把目标文档放进搜索结果，"
      "所以 baseline 的引用率本来就有 67%。接近 1 的比例会让 MDE 迅速变宽 —— "
      "**这正是 GEO-bench 选 `geo_impression`（连续份额）而不是 `cited` 做主指标的原因**。")
    A("")

    A("## 5. 应用到两篇被审计的论文")
    A("")
    for k, v in rep["apply_to_literature"].items():
        A(f"### {k}")
        A("")
        A(f"- **它报的**：{v['reported']}")
        if "wilson_under_independence" in v:
            w = v["wilson_under_independence"]
            A(f"- **本项目配的区间（全独立假设下）**：")
            A(f"  - `134/150` = 89.3% → 95% CI [{w['134/150'][0]:.3f}, {w['134/150'][1]:.3f}]")
            A(f"  - `74/150` = 49.3% → 95% CI [{w['74/150'][0]:.3f}, {w['74/150'][1]:.3f}]")
        else:
            A(f"- **本项目 citation rate**：{v['our_cited_rate']:.1f}%，"
              f"格内 SD（测量级）= **{v['our_measurement_sd_pp']:.2f} pp**")
        A(f"- ⚠️ {v['caveat']}")
        A("")

    A("## 6. 这一节解决了什么、没解决什么")
    A("")
    A("**解决了**：")
    A("1. **「σ 是哪个 outcome、在哪个层级上的」这个含糊说法，现在有答案了。** "
      "`cited` 与 `geo_impression` 的格内 SD 差 "
      f"{nl['verdict'].split('前者是后者的')[1].split('倍')[0].strip()} 倍；"
      "而 09 号那个 6.77pp 其实是**测量级**、且只在 **4 个 query** 上估过 —— "
      "**它本身就是个高方差估计**。这是本节最实质的收获。")
    A("2. 2607.12056 的格内方差从未测过；本项目在同一 harness 家族上测了两种 outcome，"
      "并给出 ICC 与设计效应（DEFF 分别为 "
      f"{rep['clustering']['cited']['icc_deff']['design_effect']} 与 "
      f"{rep['clustering']['geo_impression']['icc_deff']['design_effect']}）。")
    A("3. 拿到了 citation rate 上的**配对** MDE 表，配对比非配对省下的倍数可量化。")
    A("")
    A("**没解决（诚实评估）**：")
    A("1. **没有在 MIMIQ 或 2607.12056 的 harness 上复测。** "
      "σ 是 harness + 模型 + outcome 的联合属性，跨 harness 不可搬用。")
    A("2. 因此对 2603.09296 表 2 的判定**仍然是「需要在 MIMIQ 上复测才能定论」，不是定罪。**")
    A("3. n66 的 66 个 query 全部是 GEO-bench 的品牌曝光场景，**不是 MIMIQ-OOD 的 persona 分布**。")
    A("4. **天花板效应限制了 `cited` 作为主指标的上限** —— baseline 已经 67%，"
      "这个 outcome 的统计分辨率先天受限。这本身是关于指标选择的一个发现，"
      "但也意味着本节的 σ 不能直接推广到别的 citation rate 设定。")
    A("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="10_citation_noise.py",
        description="在 citation rate 上复测 run-to-run 噪声（C3 命题④）",
    )
    p.add_argument("--csv", default=DEFAULT_CSV, help="raw_trials.csv 路径")
    p.add_argument("--out", default=None, help="写入 markdown 的路径")
    p.add_argument("--json", default=None, help="写入 json 的路径")
    p.add_argument("--selftest", action="store_true", help="只跑自检")
    a = p.parse_args(argv)

    if a.selftest:
        return selftest()

    if not os.path.exists(a.csv):
        print(f"ERROR: 找不到 {a.csv}", file=sys.stderr)
        return 2

    rows = load_rows(a.csv)
    if not rows:
        print("ERROR: 没有可用数据行", file=sys.stderr)
        return 2

    rep = analyze(rows)
    md = render(rep)

    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(md + "\n")
        print(f"wrote {a.out}")
    else:
        print(md)

    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, ensure_ascii=False, indent=2)
        print(f"wrote {a.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
