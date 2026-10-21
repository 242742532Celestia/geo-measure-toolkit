#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
statpower.py -- GEO 测量研究的统计功效反解内核

为什么需要它
------------
本项目已实测：官方 GEO-bench + qwen-plus，3 次重复，geo_impression 跨 run
sigma = 0.0677（6.77 个百分点），占总方差 49.7%；baseline 从 0.0564 漂到 0.0919。
这个数字（本文件 `NOISE_SIGMA_PP`）加上下面这组函数，构成 C3「功效反解」命题的
全部工具基础：给定一篇论文报告的数字，反解出它真实能检出多小的效应。

核心用途（反解方向是本文的核心主张）
----------------------------------------
    常规做法：先假定效应 -> 算需要多少样本 -> 检验是否显著
    本文件做法：先看它报了什么 -> 反解它的最小可检出效应（MDE）-> 判断该结论是否可判定

纯标准库实现（statistics.NormalDist），不依赖 scipy/numpy。
Python >= 3.8。

CLI
---
    python tools/statpower.py selftest
    python tools/statpower.py mde-prop --p1 0.56 --n1 50 --n2 50
    python tools/statpower.py mde-r --n 112
    python tools/statpower.py deff --clusters 15 --cluster-size 10 --icc 0.5
    python tools/statpower.py perm-resolution --candidates 26
    python tools/statpower.py ci --k 134 --n 150
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from statistics import NormalDist
# ── 控制台编码兜底 ────────────────────────────────────────────────
# Windows 控制台默认 GBK 码页，打印 ✅/⚠/❌/🔴/⇒/² 这类字符会 UnicodeEncodeError 崩溃。
# 这里把「无法编码」改成「替换显示」而不是抛异常；UTF-8 环境（Linux/macOS/CI）下是空操作。
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

_N = NormalDist()

# ---------------------------------------------------------------------------
# 本项目一手实测的噪声常量（唯一需要外部输入的部分）
# 来源：examples/09_measure_noise.py，官方 GEO-bench + qwen-plus，3 次重复
# ---------------------------------------------------------------------------

NOISE_SIGMA = 0.0677          # geo_impression 跨 run 标准差（比例尺度）
NOISE_SIGMA_PP = 6.77         # 同上，百分点
NOISE_VARIANCE_SHARE = 0.497  # 占总方差 49.7%

#: 效应量低于此值（百分点）时，本项目实测表明 n=4~8 什么都测不出来
MDE_INSIGNIFICANT_PP = 6.77


# ---------------------------------------------------------------------------
# 基础：正态近似下的两比例 power
# ---------------------------------------------------------------------------

def z(alpha: float, side: str = "two") -> float:
    """临界值。side: 'two' | 'one' | 'upper' | 'lower'"""
    if side == "two":
        return _N.inv_cdf(1.0 - alpha / 2.0)
    if side == "one":
        return _N.inv_cdf(1.0 - alpha)
    if side == "upper":
        return _N.inv_cdf(1.0 - alpha)
    if side == "lower":
        return -_N.inv_cdf(1.0 - alpha)
    raise ValueError(f"unknown side: {side}")


def power_two_proportions(
    p1: float, p2: float, n1: int, n2: int, alpha: float = 0.05,
    pooled: bool = True,
) -> float:
    """两比例对比的 power（正态近似）。

    pooled=True（默认，且**必须**用它做样本量计算）
        对应标准的 two-proportion z test：H0 下 p1=p2，标准误用合并比例 p̄。
        这是社区通用做法，n_required 必须用这一支，否则会**低估**所需样本量。

    pooled=False
        用备择下的真实方差（unpooled）。对给定的真实 p1/p2 这是更贴近的
        检出概率描述，但反解 n 时偏乐观。仅作诊断对照，不用于定样本量。

    ⚠️ 无论哪一支，若 p1 本身是估计值，都还应额外承担估计误差；
       本函数是**乐观下界**。
    """
    if n1 < 2 or n2 < 2:
        return float("nan")
    d = abs(p1 - p2)
    if pooled:
        pp = (p1 * n1 + p2 * n2) / (n1 + n2)
        se = math.sqrt(pp * (1 - pp) * (1.0 / n1 + 1.0 / n2))
    else:
        se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    if se <= 0:
        return 1.0 if d > 0 else float("alpha")
    zc = z(alpha, "two")
    return _N.cdf(d / se - zc) + _N.cdf(-d / se - zc)


def power_two_proportions_unpooled(
    p1: float, p2: float, n1: int, n2: int, alpha: float = 0.05
) -> float:
    return power_two_proportions(p1, p2, n1, n2, alpha, pooled=False)


def power_two_proportions_equal_n(
    p1: float, p2: float, n_per_arm: int, alpha: float = 0.05
) -> float:
    return power_two_proportions(p1, p2, n_per_arm, n_per_arm, alpha)


# ---------------------------------------------------------------------------
# MDE 求解：二分法
# ---------------------------------------------------------------------------

def _bisect(f, lo: float, hi: float, tol: float = 1e-6, iters: int = 200) -> float:
    """求 f(x) = 0 的根，f 在 (lo, hi) 上单调。"""
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        return float("nan")
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        fm = f(mid)
        if abs(fm) < tol or (hi - lo) < 1e-9:
            return mid
        if flo * fm <= 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return (lo + hi) / 2.0


def mde_two_proportions(
    p1: float,
    n1: int,
    n2: int | None = None,
    target_power: float = 0.80,
    alpha: float = 0.05,
) -> float:
    """给定基线比例与样本量，反解最小可检出差异（比例尺度，非百分点）。

    做法：在 p2 方向上二分，使 power(p1, p2) == target_power。
    返回 delta = p1 - p2（正数）。
    """
    n2 = n1 if n2 is None else n2
    if p1 <= 0 or p1 >= 1 or n1 < 2 or n2 < 2:
        return float("nan")

    # 差异下界
    def f(d: float) -> float:
        p2 = p1 - d
        if p2 <= 0:
            return 1.0
        return power_two_proportions(p1, p2, n1, n2, alpha) - target_power

    hi = min(p1, 1.0 - 1e-9)
    if f(hi) < 0:
        return float("nan")  # 即使 p2=0 也达不到目标功效
    return _bisect(f, 0.0, hi)


def n_required_two_proportions(
    p1: float,
    p2: float,
    alpha: float = 0.05,
    target_power: float = 0.80,
) -> int:
    """给定真实效应，反解每臂所需样本量（向上取整）。"""
    if p1 == p2:
        return 1 << 30
    lo, hi = 2, 4096
    while hi < (1 << 22):
        if power_two_proportions_equal_n(p1, p2, hi, alpha) >= target_power:
            break
        lo, hi = hi, hi * 2
    else:
        return 1 << 30
    # 二分细化
    while lo < hi:
        mid = (lo + hi) // 2
        if power_two_proportions_equal_n(p1, p2, mid, alpha) >= target_power:
            hi = mid
        else:
            lo = mid + 1
    return int(lo)


# ---------------------------------------------------------------------------
# Pearson r
# ---------------------------------------------------------------------------

def mde_pearson_r(n: int, alpha: float = 0.05, target_power: float = 0.80) -> float:
    """给定样本量反解最小可检出相关系数。

    推导：t = r*sqrt(df)/sqrt(1-r^2), df = n-2
    power 达标要求 t >= z_{1-a/2} + z_{1-b}
    => r = t / sqrt(df + t^2)
    """
    if n < 4:
        return float("nan")
    df = n - 2
    t = z(alpha, "two") + z(1.0 - target_power, "upper")
    return t / math.sqrt(df + t * t)


def r_from_p(p: float, n: int) -> float:
    """从报告的 p 值反解实际相关系数（|r|）。"""
    if n < 3 or p <= 0 or p >= 1:
        return float("nan")
    df = n - 2
    t = _N.inv_cdf(1.0 - p / 2.0)          # 双尾
    return t / math.sqrt(df + t * t)


def n_required_pearson_r(
    r: float, alpha: float = 0.05, target_power: float = 0.80
) -> int:
    """给定目标效应量反解所需 n。"""
    if r <= 0:
        return 1 << 30
    lo, hi = 4, 4096
    while mde_pearson_r(hi, alpha, target_power) > r:
        lo, hi = hi, hi * 2
        if hi > (1 << 22):
            return 1 << 30
    while lo < hi:
        mid = (lo + hi) // 2
        if mde_pearson_r(mid, alpha, target_power) <= r:
            hi = mid
        else:
            lo = mid + 1
    return int(lo)


def rho_implied_by_mde(
    mde: float, sigma: float, n: int, alpha: float = 0.05, target_power: float = 0.80
) -> float:
    """反解：某次**配对**实测得到 MDE，对应的配对内相关 rho 是多少。

    配对设计下  MDE = (z_a+z_b) * sigma_d / sqrt(n),  sigma_d = sigma*sqrt(2(1-rho))
    => rho = 1 - ( MDE*sqrt(n) / ((z_a+z_b)*sigma) )^2 / 2

    用途：把「实测 1.5–3pp 需约 50–70 query」这个经验结论翻译成
    **可检验的假设**，而不是当成既定常数引用。
    """
    denom = (z(alpha, "two") + z(1.0 - target_power, "upper")) * sigma
    if denom <= 0 or n < 2:
        return float("nan")
    return 1.0 - (mde * math.sqrt(n) / denom) ** 2 / 2.0


# ---------------------------------------------------------------------------
# 连续变量（两均值差）
# ---------------------------------------------------------------------------

def mde_continuous(
    sigma: float,
    n_per_arm: int,
    alpha: float = 0.05,
    target_power: float = 0.80,
) -> float:
    """非配对两均值差的最小可检出效应（同单位）。"""
    if n_per_arm < 2 or sigma <= 0:
        return float("nan")
    return (z(alpha, "two") + z(1.0 - target_power, "upper")) * sigma * math.sqrt(2.0 / n_per_arm)


def mde_paired(
    sigma_d: float, n_pairs: int, alpha: float = 0.05, target_power: float = 0.80
) -> float:
    """配对差值的最小可检出效应。sigma_d 是**配对差**的标准差。

    与非配对的关系：若两组同方差 sigma、配对相关 rho，则
        sigma_d = sigma * sqrt(2(1-rho))
    配对省下的样本量比 = 2(1-rho)。
    """
    if n_pairs < 2 or sigma_d <= 0:
        return float("nan")
    return (z(alpha, "two") + z(1.0 - target_power, "upper")) * sigma_d / math.sqrt(n_pairs)


def paired_sample_advantage(sigma: float, rho: float) -> float:
    """配对比非配对省多少倍样本（等功效、等 margin 下 n_unpaired / n_paired）。"""
    rho = max(-1.0, min(1.0, rho))
    if rho >= 1.0:
        return float("inf")
    return 2.0 * (1.0 - rho)


# ---------------------------------------------------------------------------
# 聚类 / 设计效应
# ---------------------------------------------------------------------------

def design_effect(cluster_size: int, icc: float) -> float:
    """Kish 设计效应：DEFF = 1 + (m-1) * ICC。"""
    if cluster_size < 1:
        return float("nan")
    return 1.0 + (cluster_size - 1) * max(0.0, icc)


def effective_n(n_total: int, deff: float) -> float:
    return n_total / deff if deff > 0 else float("nan")


def design_effect_table(cluster_size: int, iccs=(0.0, 0.1, 0.3, 0.5, 0.7)) -> list[dict]:
    return [
        {"icc": c, "deff": round(design_effect(cluster_size, c), 3)} for c in iccs
    ]


# ---------------------------------------------------------------------------
# 区间估计
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float, float]:
    """Wilson score 区间（二项比例）。返回 (点估计, 下界, 上界)。

    为什么用 Wilson 而不是 Wald：n*p 或 n*(1-p) 接近 5 时 Wald 区间会越出 [0,1]，
    且在极端比例上宽度严重失真。GEO 论文的阳性率常在 3%~50% 区间，正是这个坑。
    """
    if n <= 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    zc = z(alpha, "two")
    denom = 1.0 + zc * zc / n
    center = (p + zc * zc / (2 * n)) / denom
    half = (zc / denom) * math.sqrt(p * (1 - p) / n + zc * zc / (4 * n * n))
    return p, max(0.0, center - half), min(1.0, center + half)


def wilson_width_pp(k: int, n: int, alpha: float = 0.05) -> float:
    _, lo, hi = wilson_ci(k, n, alpha)
    return (hi - lo) * 100.0


# ---------------------------------------------------------------------------
# 置换检验分辨率
# ---------------------------------------------------------------------------

def permutation_resolution(n_candidates: int) -> dict:
    """置换检验的最小可达 p 值。

    n 个候选断点 -> 最小 p = 1/(n+1)（含自身），保守写法 1/n。
    这是**离散化下限**，与效应量大小无关：候选不够多，p 永远到不了 0.01。
    """
    n = int(n_candidates)
    if n < 1:
        return {"candidates": n, "min_p_exact": None, "min_p_conservative": None}
    return {
        "candidates": n,
        "min_p_exact": 1.0 / (n + 1),
        "min_p_conservative": 1.0 / n,
        "can_reach_0.05": (1.0 / n) <= 0.05,
        "candidates_needed_for_0.05": 20,
        "candidates_needed_for_0.01": 100,
    }


def permutation_p(observed: float, placebo_values: list[float]) -> dict:
    """给定观测值与 placebo 分布，算经验 p。

    这是**唯一**能回答「它的 placebo 检验实际能走多远」的函数：
    p 的分辨率由 len(placebo) 决定，不是由效应量决定。
    """
    n = len(placebo_values)
    if n == 0:
        return {"n": 0, "p": None, "note": "empty placebo distribution"}
    ge = sum(1 for v in placebo_values if abs(v) >= abs(observed))
    p = ge / (n + 1)
    return {
        "n": n,
        "n_ge_observed": ge,
        "p": p,
        "min_p_exact": 1.0 / (n + 1),
        "min_p_conservative": 1.0 / n,
        "can_reach_0.05": (1.0 / n) <= 0.05,
    }


# ---------------------------------------------------------------------------
# 条件反解：给定报告的 p，反解「真实效应被高估/低估了多少」
# ---------------------------------------------------------------------------

def p_ratio_inflation(reported_n: int, true_n: int) -> float:
    """分母膨胀比：把 cluster 展开当独立单元会把 n 虚高多少倍。"""
    return reported_n / true_n if true_n > 0 else float("nan")


def mde_inflation(reported_n: int, true_n: int, alpha: float = 0.05,
                  target_power: float = 0.80) -> dict:
    """连续变量的 MDE 因样本量虚高而被低估多少倍。"""
    r = p_ratio_inflation(reported_n, true_n)
    return {
        "reported_n": reported_n,
        "true_n": true_n,
        "n_inflation": round(r, 2),
        "mde_underestimate_factor": round(math.sqrt(r), 3),
        "note": "MDE 与 1/sqrt(n) 成正比 -> 分母虚高 sqrt(n_inflation) 倍，MDE 偏小同样倍数",
    }


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------

def selftest(verbose: bool = True) -> int:
    fails = []

    def chk(name, got, lo, hi, tol=1e-6):
        ok = (got is not None) and not (isinstance(got, float) and math.isnan(got)) \
            and (lo - tol) <= got <= (hi + tol)
        if verbose:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {got}")
        if not ok:
            fails.append(name)

    print("== statpower selftest ==")

    # 1. Wilson：k=134/150 = 89.3% 应落在 [0.836, 0.934]
    p, lo, hi = wilson_ci(134, 150)
    chk("wilson 134/150 点估计", round(p, 4), 0.893, 0.893, 1e-3)
    chk("wilson 134/150 下界", round(lo, 3), 0.836, 0.836, 2e-3)
    chk("wilson 134/150 上界", round(hi, 3), 0.934, 0.934, 2e-3)

    p2, lo2, hi2 = wilson_ci(74, 150)
    chk("wilson 74/150 点估计", round(p2, 4), 0.493, 0.493, 1e-3)
    chk("wilson 74/150 下界", round(lo2, 3), 0.414, 0.414, 2e-3)
    chk("wilson 74/150 上界", round(hi2, 3), 0.573, 0.573, 2e-3)

    # 2. 89.3% vs 49.3% 在 n=150/臂 下的 MDE 应远小于 40pp（可检出）
    mde = mde_two_proportions(0.893, 150, 150)
    chk("MDE(0.893, n=150) < 0.40（该差异本身可检出）", round(mde, 3), 0.0, 0.40)

    # 3. 2607.12056 表 1 的 40pp 差异所需 n 应约 21/臂（pooled 公式 20.9）
    n_need = n_required_two_proportions(0.893, 0.493)
    chk("0.893 vs 0.493 所需 n/臂 (pooled)", n_need, 20, 22)
    # 3b. 回归防护：unpooled 会给出 ~17，偏低。定样本量必须用 pooled。
    n_unpooled = int(round(
        sum(1 for n in range(2, 200)
            if power_two_proportions_unpooled(0.893, 0.493, n, n) >= 0.80 and
            power_two_proportions_unpooled(0.893, 0.493, n - 1, n - 1) < 0.80)
    ))
    chk("unpooled 偏低（回归防护，应 < pooled 的 21）", 1.0 if n_unpooled < n_need else 0.0,
        1.0, 1.0)
    chk("pooled 公式手算复核 n=20.87", round(
        2 * (1.959964 + 0.841621) ** 2 * 0.693 * 0.307 / 0.40 ** 2, 2), 20.87, 20.87, 5e-3)

    # 4. 2601.00912：p=0.256, n=112 -> |r| 约 0.108；MDE(112) = 0.258
    #    ⚠️ 0.258 而非 0.19 —— 2026-10-05 修正了框架与笔记 18 中的旧值 0.19
    r_est = r_from_p(0.256, 112)
    chk("r_from_p(0.256, 112)", round(r_est, 3), 0.107, 0.109)
    r_mde = mde_pearson_r(112)
    chk("MDE_r(n=112) = 0.258（框架旧值 0.19 是错的）", round(r_mde, 3), 0.257, 0.259)
    chk("MDE_r(112) / |r_actual| >= 2 倍（null 几乎无信息量）",
        round(r_mde / r_est, 2), 2.3, 2.5)
    chk("达到 0.19 需 n =", n_required_pearson_r(0.19), 210, 230)

    # 5. 配对优势：rho=0.5 -> 2(1-0.5)=1.0x ... rho=0.8 -> 0.4x
    chk("paired advantage rho=0.0", paired_sample_advantage(1.0, 0.0), 2.0, 2.0)
    chk("paired advantage rho=0.8", round(paired_sample_advantage(1.0, 0.8), 3), 0.4, 0.4)

    # 6. 设计效应：15 格 × 10 次, ICC=0.5 -> DEFF = 1+9*0.5 = 5.5
    chk("DEFF(10, 0.5)", design_effect(10, 0.5), 5.5, 5.5)
    chk("DEFF(10, 0.0)", design_effect(10, 0.0), 1.0, 1.0)
    chk("n_eff(300, 5.5)", round(effective_n(300, 5.5), 2), 54.55, 54.55, 1e-2)

    # 7. 置换分辨率：26 个候选 -> 最小 p 1/26 = 0.038（可到 0.05）；100 个 -> 0.01
    r26 = permutation_resolution(26)
    chk("26 候选 min p", round(r26["min_p_conservative"], 4), 0.0385, 0.0385, 1e-3)
    r100 = permutation_resolution(100)
    chk("100 候选 min p", round(r100["min_p_conservative"], 4), 0.01, 0.01, 1e-4)
    r19 = permutation_resolution(19)
    chk("19 候选无法到 0.05", 0.0 if not r19["can_reach_0.05"] else 1.0, 0.0, 0.0)

    # 8. MDE 膨胀：784 vs 112 -> 7x n -> MDE 偏小 2.646 倍
    infl = mde_inflation(784, 112)
    chk("n 虚高倍数 784/112", infl["n_inflation"], 7.0, 7.0)
    chk("MDE 低估倍数", round(infl["mde_underestimate_factor"], 3), 2.646, 2.646, 1e-2)

    # 9. 连续变量 MDE 自洽：δ = 2.8016 * sigma * sqrt(2/n)
    m1 = mde_continuous(1.0, 25)
    chk("MDE 连续 sigma=1, n=25/arm = 2.8016*sqrt(2/25)", round(m1, 4), 0.7924, 0.7924, 1e-3)
    chk("MDE(本项目 sigma, n=50/臂) 非配对", round(mde_continuous(NOISE_SIGMA, 50) * 100, 2),
        3.79, 3.79, 1e-2)
    chk("MDE(本项目 sigma, n=70/臂) 非配对", round(mde_continuous(NOISE_SIGMA, 70) * 100, 2),
        3.21, 3.21, 1e-2)
    # 9b. 配对反解：本项目实测「1.5–3pp 需 50–70 query」是非配对做不到的
    #     （非配对 @50 只能测 3.79pp），隐含配对内相关 rho 必须很高。
    r_lo = rho_implied_by_mde(0.03, NOISE_SIGMA, 50)   # 50 query 测到 3pp
    r_hi = rho_implied_by_mde(0.015, NOISE_SIGMA, 70)  # 70 query 测到 1.5pp
    chk("隐含 rho(50q->3pp) 在 (0,1)", 1.0 if 0.0 < r_lo < 1.0 else 0.0, 1.0, 1.0)
    chk("隐含 rho(70q->1.5pp) 在 (0,1)", 1.0 if 0.0 < r_hi < 1.0 else 0.0, 1.0, 1.0)
    chk("实测 1.5-3pp 区间只可能来自配对设计（非配对 @50 = 3.79pp）",
        1.0 if r_lo > 0.3 else 0.0, 1.0, 1.0)
    # 9c. 配对优势公式与实测「省 4-11 倍」一致：rho=0.375 -> 1.25x; rho=0.6 -> 0.8x
    chk("paired advantage rho=0.375", round(paired_sample_advantage(1.0, 0.375), 3), 1.25, 1.25)

    # 10. 本项目常量自洽
    chk("NOISE_SIGMA_PP = 100 * NOISE_SIGMA", NOISE_SIGMA_PP, 6.77, 6.77, 1e-2)
    chk("NOISE_SIGMA 在 (0,1)", NOISE_SIGMA, 0.0, 1.0)

    print(f"\n{len(fails)} failed" if fails else "\nall checks passed")
    if fails:
        for f in fails:
            print(f"  FAILED: {f}")
        return 1
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _fmt(d: dict) -> str:
    return json.dumps(d, ensure_ascii=False, indent=2)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="statpower.py",
        description="GEO 测量研究的统计功效反解内核（纯标准库）",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("selftest", help="运行内置自检")

    s = sub.add_parser("mde-prop", help="两比例对比的最小可检出差异")
    s.add_argument("--p1", type=float, required=True, help="基线比例，如 0.56")
    s.add_argument("--n1", type=int, required=True)
    s.add_argument("--n2", type=int, default=None)
    s.add_argument("--power", type=float, default=0.80)
    s.add_argument("--alpha", type=float, default=0.05)

    s = sub.add_parser("n-for-prop", help="给定效应量反解每臂所需 n")
    s.add_argument("--p1", type=float, required=True)
    s.add_argument("--p2", type=float, required=True)
    s.add_argument("--power", type=float, default=0.80)
    s.add_argument("--alpha", type=float, default=0.05)

    s = sub.add_parser("mde-r", help="给定 n 反解最小可检出相关系数")
    s.add_argument("--n", type=int, required=True)
    s.add_argument("--power", type=float, default=0.80)
    s.add_argument("--alpha", type=float, default=0.05)

    s = sub.add_parser("r-from-p", help="从报告的 p 值反解实际 |r|")
    s.add_argument("--p", type=float, required=True)
    s.add_argument("--n", type=int, required=True)

    s = sub.add_parser("deff", help="设计效应表")
    s.add_argument("--clusters", type=int, default=15, help="聚类数（格数）")
    s.add_argument("--cluster-size", type=int, default=10, help="每簇样本数")
    s.add_argument("--icc", type=float, default=None, help="组内相关系数；不给则输出 ICC 扫描表")
    s.add_argument("--n-total", type=int, default=None, help="展开后的总单元数")

    s = sub.add_parser("perm-resolution", help="置换检验的最小可达 p")
    s.add_argument("--candidates", type=int, required=True)

    s = sub.add_parser("ci", help="Wilson 区间")
    s.add_argument("--k", type=int, required=True, help="成功次数")
    s.add_argument("--n", type=int, required=True)
    s.add_argument("--alpha", type=float, default=0.05)

    a = p.parse_args(argv)

    if a.cmd == "selftest":
        return selftest()

    if a.cmd == "mde-prop":
        d = mde_two_proportions(a.p1, a.n1, a.n2, a.power, a.alpha)
        return print(_fmt({
            "p1": a.p1, "n1": a.n1, "n2": a.n2 or a.n1,
            "target_power": a.power, "alpha": a.alpha,
            "mde": round(d, 6), "mde_pp": round(d * 100, 3),
            "project_noise_sigma_pp": NOISE_SIGMA_PP,
            "mde_vs_noise_ratio": round(d * 100 / NOISE_SIGMA_PP, 3),
        })) or 0

    if a.cmd == "n-for-prop":
        n = n_required_two_proportions(a.p1, a.p2, a.alpha, a.power)
        return print(_fmt({
            "p1": a.p1, "p2": a.p2, "delta_pp": round(abs(a.p1 - a.p2) * 100, 3),
            "n_per_arm": n, "target_power": a.power, "alpha": a.alpha,
        })) or 0

    if a.cmd == "mde-r":
        r = mde_pearson_r(a.n, a.alpha, a.power)
        return print(_fmt({
            "n": a.n, "target_power": a.power, "alpha": a.alpha,
            "mde_r": round(r, 4),
            "note": "n 减半时 MDE 只恶化 sqrt(2) 倍；n 增到 4 倍 MDE 才减半",
        })) or 0

    if a.cmd == "r-from-p":
        r = r_from_p(a.p, a.n)
        mde = mde_pearson_r(a.n)
        return print(_fmt({
            "reported_p": a.p, "n": a.n,
            "implied_abs_r": round(r, 4),
            "mde_r_at_80pct_power": round(mde, 4),
            "verdict": "报告的 null 未排除 |r| < %.3f 的效应" % round(mde, 4),
        })) or 0

    if a.cmd == "deff":
        n_total = a.n_total if a.n_total else a.clusters * a.cluster_size
        if a.icc is not None:
            deff = design_effect(a.cluster_size, a.icc)
            return print(_fmt({
                "clusters": a.clusters, "cluster_size": a.cluster_size,
                "icc": a.icc, "deff": round(deff, 4),
                "n_total_as_counted": n_total,
                "n_effective": round(effective_n(n_total, deff), 2),
                "n_inflation": round(n_total / effective_n(n_total, deff), 2),
            })) or 0
        return print(_fmt({
            "clusters": a.clusters, "cluster_size": a.cluster_size,
            "n_total_as_counted": n_total,
            "table": design_effect_table(a.cluster_size),
            "note": "ICC 从未测 -> DEFF 只能给区间，不能给点值",
        })) or 0

    if a.cmd == "perm-resolution":
        return print(_fmt(permutation_resolution(a.candidates))) or 0

    if a.cmd == "ci":
        pt, lo, hi = wilson_ci(a.k, a.n, a.alpha)
        return print(_fmt({
            "k": a.k, "n": a.n, "p_hat": round(pt, 4),
            "ci_low": round(lo, 4), "ci_high": round(hi, 4),
            "width_pp": round((hi - lo) * 100, 2),
            "walford_warning": "阳性率 <5% 或 >95% 时 Wald 区间会越界或严重失真",
        })) or 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
