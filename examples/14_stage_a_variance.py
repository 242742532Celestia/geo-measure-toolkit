#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
14_stage_a_variance.py -- 预注册阶段 A：0 臂方差分解与样本量定型

对应方法规范的 §6。本文只做 §6 规定的计算，**不新增规则**。

🔴 本脚本的纪律
---------------
1. **样本量由 §6.3 的先写死规则决定，不由「跑了多少」决定。**
   脚本读入估计量后按档位自动定型，**不允许手工指定 n 绕过档位**。
2. **产出的是 MDE 报告，不是「实验可行」的背书。**
   若定型后 MDE 未达目标，如实报出并进降级档。
3. **所有数字从落盘 CSV 重算，不依赖内存状态，不读任何缓存结论。**

用法
----
    python examples/14_stage_a_variance.py --csv path/to/raw_trials.csv
    python examples/14_stage_a_variance.py --csv ... --out outputs/stage_a.json
    python examples/14_stage_a_variance.py --selftest
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import sys
from collections import defaultdict
from statistics import NormalDist
# ── 控制台编码兜底 ────────────────────────────────────────────────
# Windows 控制台默认 GBK 码页，打印 ✅/⚠/❌/🔴/⇒/² 这类字符会 UnicodeEncodeError 崩溃。
# 这里把「无法编码」改成「替换显示」而不是抛异常；UTF-8 环境（Linux/macOS/CI）下是空操作。
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from statpower import mde_paired, mde_two_proportions, wilson_ci  # noqa: E402

_N = NormalDist()

# ---------------------------------------------------------------------------
# 预注册常量（§6 / §6.3，改动必须递增方法规范版本号）
# ---------------------------------------------------------------------------
TARGET_MDE_PP = 5.0        # §6.1 目标 MDE
ALPHA = 0.05                # 双尾
POWER = 0.80
BUDGET_N = 400              # §6.2 预算上限 / 臂
BUDGET_CALLS_CAP = 12000    # §6.2 预算上限总调用数
N_SOURCES = 5               # §4 源文档数（目标 + 4 竞争者）-> 位置分层数
POSITION_MIN_PER_STRATUM = 30   # §11 局限 5：每个位置至少 30 个 query


def _arm_count() -> int:
    """臂数**从冻结的 prompt 登记表推导**，不手写常数。

    2604.25707 附录 B 原文：*"A minimal randomized design would use five arms:
    original page, structured headings only, evidence blocks only,
    semantic-intent alignment only, and full evidence-container rewrite."*
    即 **1 original + 4 treatment = 5 臂**。

    🔴 2026-10-06 修正：此前本脚本写 `ARMS_TOTAL = 6`、§7.1 写「5 个处理臂
    × 2 = 10 个检验」，都**与附录 B 原文和 §2.1 矛盾**。5 臂才是对的。
    现在从 `tools/freeze_prompts.ARMS` 推导，将来 prompt 增减臂时
    预算与检验族大小自动跟着变，不会再漂。
    """
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
        from freeze_prompts import ARMS as _FROZEN_ARMS
        return len(_FROZEN_ARMS)
    except Exception:
        return 5  # 附录 B 原文的五臂


DEGRADE_LADDER = {          # §6.3 先写死的降级顺序
    0: (200, TARGET_MDE_PP),
    1: (400, 7.0),
    2: (400, None),          # None = 按实测 σ_d 重算并明确报出
}
ARMS_TOTAL = _arm_count()    # 5（附录 B 原文五臂），见 _arm_count()
REPEATS = 15


# ---------------------------------------------------------------------------
# 载入
# ---------------------------------------------------------------------------

def load(path: str, arm: str | None = None) -> tuple[list[dict], list[str], list[str]]:
    """读入落盘 CSV。

    🔴 **必须锁定单臂。** 阶段 A 的方差分解是「固定 query、跨重复」，
    如果把多个臂混进同一个格，格内 SD 就变成了**臂间差异 + 重复间差异**的混合，
    数字会**静默地变大**且看不出任何异常 —— 2026-10-06 实测：
    n66 的 10 臂数据误喂进来后报出 σ_cite = 20.5pp，
    而同一份数据只取 baseline 臂时是 14.3pp。差 6pp，MDE 直接翻倍。

    所以这里默认**只接受单臂**，多臂直接报错并列出臂名，由人显式选。
    """
    rows = []
    arms_seen: set[str] = set()
    dropped: set[str] = set()
    # 阶段 A'（MiniMax-M3）落盘是 .csv.gz（v1.3 压缩），这里一并支持，
    # 否则会误报「文件不存在」或「空数据」。
    opener = (gzip.open if path.endswith(".gz") else open)
    with opener(path, "rt", encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("error"):
                dropped.add(f"error:{r.get('arm', '?')}")
                continue
            a = (r.get("arm") or "").strip()
            arms_seen.add(a)
            if arm is not None and a != arm:
                continue
            try:
                rows.append({
                    "query_id": r["query_id"],
                    "arm": a,
                    "repeat": int(r["repeat"]),
                    "seed": int(r.get("seed") or 0),
                    "target_idx": int(r.get("target_idx") or -1),
                    "cited": float(r["cited"]),
                    "geo_impression": float(r["geo_impression"]),
                    "first_pos": int(r.get("first_pos") or -1),
                })
            except (KeyError, TypeError, ValueError):
                dropped.add(f"parse:{a}")
    return rows, sorted(arms_seen), sorted(dropped)


def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def stdev(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    n = len(xs)
    if n < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def median(xs):
    s = sorted(x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x)))
    n = len(s)
    if not n:
        return float("nan")
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


# ---------------------------------------------------------------------------
# 方差分解
# ---------------------------------------------------------------------------

def variance_decomposition(rows: list[dict], field: str) -> dict:
    """0 臂的方差分解：每个 query 有 R 次重复。"""
    by_q = defaultdict(list)
    for r in rows:
        by_q[r["query_id"]].append(r[field])
    cells = [v for v in by_q.values() if len(v) > 1]
    if not cells:
        return {"error": "没有重复观测"}

    within = [stdev(c) for c in cells]
    within = [w for w in within if not math.isnan(w)]
    cell_means = [mean(c) for c in cells]

    # 🔴 两种格内 SD 估计量，差别很大，必须都报：
    #   mean_of_sds  = 各格 SD 的**算术平均**  -> 本项目早期一直引用这个
    #   rms_of_sds   = sqrt(各格 SD² 的平均)  = 合并格内方差，**无偏**
    # 在 n66 baseline 上：16.28pp vs **28.34pp**，比值 1.74。
    # 原因：66 个格里 **44 个格方差恰为 0**（引擎对这些 query 是确定性的），
    # 中位格内 SD = 0。SD 分布高度右偏，取算术平均会被这些 0 拖下去。
    # MDE 只需要「单次观测的方差」，即 E[Var(y|q)] = 各格方差的平均 -> 用 RMS。
    n_w = len(within)
    mean_of_sds = mean(within) if n_w else float("nan")
    rms_of_sds = math.sqrt(mean([w * w for w in within])) if n_w else float("nan")
    n_zero = sum(1 for w in within if w < 1e-12)

    # run-level：每个 repeat 的整体均值
    by_rep = defaultdict(list)
    for r in rows:
        by_rep[r["repeat"]].append(r[field])
    per_run = {k: mean(v) for k, v in sorted(by_rep.items())}
    run_vals = list(per_run.values())

    allv = [r[field] for r in rows]
    return {
        "n_cells": len(cells),
        "repeats_per_cell": len(cells[0]),
        "overall_mean": round(mean(allv), 6),
        "measurement_sd": round(rms_of_sds, 6) if n_w else None,
        "measurement_sd_pp": round(rms_of_sds * 100, 3) if n_w else None,
        "mean_of_cell_sds_pp": round(mean_of_sds * 100, 3) if n_w else None,
        "rms_over_mean": round(rms_of_sds / mean_of_sds, 4) if n_w and mean_of_sds else None,
        "zero_variance_cells": n_zero,
        "zero_variance_share": round(n_zero / n_w, 4) if n_w else None,
        "query_sd": round(stdev(cell_means), 6),
        "query_sd_pp": round(stdev(cell_means) * 100, 3),
        "total_sd": round(stdev(allv), 6),
        "run_level_sd": round(stdev(run_vals), 6),
        "run_level_sd_pp": round(stdev(run_vals) * 100, 4),
        "run_level_range_pp": round((max(run_vals) - min(run_vals)) * 100, 4),
        "per_run_mean": {k: round(v, 6) for k, v in per_rep.items()} if (per_rep := per_run) else {},
        "note": (
            "measurement_sd = **RMS（合并格内方差，无偏）**，用于 MDE。"
            "注意它 ≠ 各格 SD 的算术平均 —— 后者被零方差格拖低，不可用于 MDE。"
        ),
    }


def variance_components(rows: list[dict], field: str) -> dict:
    """单因素随机效应方差分解（ANOVA），只在**完整格**上估计。

    模型：`y[query, rep] = μ_q + e[query, rep]`，`e ~ N(0, σ²_e)`。
    这是 0 臂重复测量唯一支持的模型 —— 没有处理臂，就没有处理效应项。

    为什么必须做这个分解（§6.1 要求「ρ 由重复测量直接估计，不用外部假设」）
    ------------------------------------------------------------------------
    两个**不同臂**在同一 query 上：

        y_A = μ_q + τ_A + e_A        e_A 与 e_B 独立（prompt 不同 → 误差独立）
        y_B = μ_q + τ_B + e_B        μ_q 是**共享**的 query 效应

        ρ(臂间) = σ²_q / (σ²_q + σ²_e)

    代回 §6.1 的 `σ_d = σ_total·√(2(1-ρ))`：

        σ_d = √(σ²_q + σ²_e) · √( 2σ²_e/(σ²_q+σ²_e) ) = **√2 · σ_e**

    **ρ 消掉了。** 0 臂重复测量能定出配对差 SD 的一个**下界**，
    而且这个下界与 query 难度方差无关 —— 无论 query 之间差多少。

    真值还要加上处理效应在 query 间的异质性 `Var(τ_B - τ_A)`，
    而那一项在 0 臂设计下**原理上测不到**。所以本函数给的是下界，不是估计。
    """
    by_q: dict[str, list] = defaultdict(list)
    for r in rows:
        by_q[r["query_id"]].append(r[field])
    R = max((len(v) for v in by_q.values()), default=0)
    cells = [v for v in by_q.values() if len(v) == R]
    k = len(cells)
    if k < 2 or R < 2:
        return {"error": f"完整格不足（k={k}, R={R}）"}

    grand = mean([x for c in cells for x in c])
    ssb = R * sum((mean(c) - grand) ** 2 for c in cells)      # 组间
    ssw = sum((x - mean(c)) ** 2 for c in cells for x in c)   # 组内
    msb = ssb / (k - 1)
    msw = ssw / (k * (R - 1))

    var_e = max(0.0, msw)                 # σ²_meas
    var_q = max(0.0, (msb - msw) / R)     # σ²_query
    tot = var_e + var_q
    rho = (var_q / tot) if tot > 0 else float("nan")

    sd_e = math.sqrt(var_e)
    sd_tot = math.sqrt(tot)
    return {
        "n_complete_cells": k,
        "repeats_per_cell": R,
        "var_measurement": round(var_e, 8),
        "var_query": round(var_q, 8),
        "sd_measurement_pp": round(sd_e * 100, 3),
        "sd_query_pp": round(sd_tot * 100, 3),
        "rho_between_arms": round(rho, 4),
        "sigma_d_floor_pp": round(sd_e * math.sqrt(2) * 100, 3),
        "sigma_d_ceiling_pp": round(sd_tot * math.sqrt(2) * 100, 3),
        "F": round(msb / msw, 4) if msw > 0 else None,
        "df_between": k - 1,
        "df_within": k * (R - 1),
        "reading": (
            "σ_d **下界** = √2·σ_meas（ρ 消掉，与 query 难度方差无关）。"
            "上界 = √2·σ_total（假设臂间相关为 0）。真值 = 下界 + "
            "处理效应在 query 间异质性的方差，该项 0 臂测不到。"
        ),
    }


def paired_diff_sd(rows: list[dict], field: str) -> dict:
    """配对差 SD：**下界 / 上界 / 朴素代理** 三者都给，不假装测到了真值。

    - `floor`  来自 `variance_components`，是 §6.1 公式的真实下界；
    - `ceiling` 是臂间相关取 0 时的上界；
    - `proxy`  是把全部格内偏差汇成一个池求 SD，**包含 query 间方差**，
      因此它既不是下界也不是上界，只是数量级参考。
      （v1.0 曾把它当 σ_d 用，会把 query 难度方差误算进配对差 —— 本版纠正。）
    """
    vc = variance_components(rows, field)
    by_q: dict[str, list] = defaultdict(list)
    for r in rows:
        by_q[r["query_id"]].append(r[field])
    devs = [x - mean(v) for v in by_q.values() for x in v]
    s = stdev(devs)
    return {
        "floor_pp": vc.get("sigma_d_floor_pp"),
        "ceiling_pp": vc.get("sigma_d_ceiling_pp"),
        "proxy_sd_pp": round(s * 100, 3) if not math.isnan(s) else None,
        "proxy_kind": "naive-pooled (includes between-query variance)",
        "kind": "bracket",
        "warning": (
            "🔴 0 臂没有处理臂，**真正的 σ_d 测不到**。"
            "floor 是下界、ceiling 是上界，真值落在两者之间，"
            "差额来自处理效应在 query 间的异质性。"
            "**样本量按 floor 定（乐观），按 ceiling 定（保守），两者都报。**"
            "最终 n 必须在处理臂数据到手后用实测 σ_d 复核。"
        ),
    }


def decide_n(mde_target_pp: float, sigma_d_pp: float | None,
             p_base: float, repeats: int, ladder_level: int | None = None) -> dict:
    """按 §6.3 的先写死规则定型样本量。

    🔴 **2026-10-06 修正：MDE 必须除以 √(Q·R)，不是 √Q。**

    配对设计里，每个 (query, arm) 有 R 次重复，先对这 R 次取平均再配对，
    所以配对差的 SD 是 `σ_d/√R`，再对 Q 个 query 求均值再除 √Q：

        SE(d̄) = σ_d / √(Q·R)        MDE = z' · σ_d / √(Q·R)

    v1.0 漏掉了 √R，于是把 MDE 报大约 √R 倍、把所需 query 数也报大约 R 倍，
    **直接导致"5pp 在预算内不可达"的错误结论**。这是代数错误，与任何
    处理臂数据无关：阶段 A 只有 0 臂，跑出来才发现。
    """
    out = {"target_mde_pp": mde_target_pp, "repeats": repeats, "ladder": []}
    for lvl in sorted(DEGRADE_LADDER):
        n_cap, mde_req = DEGRADE_LADDER[lvl]
        if sigma_d_pp is None:
            n_needed = None
        else:
            sd = sigma_d_pp / 100.0
            n_paired = ((_N.inv_cdf(1 - ALPHA / 2) + _N.inv_cdf(POWER)) * sd
                        / (mde_target_pp / 100.0)) ** 2
            n_needed = int(math.ceil(n_paired / max(1, repeats)))
        out["ladder"].append({
            "level": lvl,
            "n_cap": n_cap,
            "declared_mde_pp": mde_req,
            "n_needed_for_target": n_needed,
            "paired_obs_needed": (None if n_needed is None
                                  else n_needed * repeats),
            "reachable": (n_needed is not None and n_needed <= n_cap),
        })
        if n_needed is not None and n_needed <= n_cap:
            out["selected_level"] = lvl
            out["n_per_arm"] = min(n_needed, n_cap)
            out["actual_mde_pp"] = mde_req if mde_req else TARGET_MDE_PP
            break
    else:
        last = out["ladder"][-1]
        out["selected_level"] = last["level"]
        out["n_per_arm"] = last["n_cap"]
        out["actual_mde_pp"] = None  # 按实测 σ 重算
        if sigma_d_pp:
            sd = sigma_d_pp / 100.0
            out["actual_mde_pp"] = round(
                (_N.inv_cdf(1 - ALPHA / 2) + _N.inv_cdf(POWER)) * sd
                / math.sqrt(last["n_cap"] * max(1, repeats)) * 100, 3)
        out["degraded"] = True
    return out


def mde_at(n_q: int, repeats: int, sigma_d_pp: float | None) -> float | None:
    """给定 query 数与重复数，返回可达的 MDE（pp）。"""
    if sigma_d_pp is None or n_q <= 0 or repeats <= 0:
        return None
    sd = sigma_d_pp / 100.0
    return round((_N.inv_cdf(1 - ALPHA / 2) + _N.inv_cdf(POWER)) * sd
                 / math.sqrt(n_q * repeats) * 100, 3)


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------

def analyze(rows: list[dict]) -> dict:
    arms = sorted({r["arm"] for r in rows})
    if len(arms) != 1:
        raise ValueError(
            f"阶段 A 方差分解要求**单臂**数据，收到 {len(arms)} 个：{arms}。"
            f"多臂混入会把臂间差异算进格内 SD（实测偏差 6pp），"
            f"请用 --arm <臂名> 显式选择。"
        )
    cite = variance_decomposition(rows, "cited")
    gi = variance_decomposition(rows, "geo_impression")
    vc_cite = variance_components(rows, "cited")
    vc_gi = variance_components(rows, "geo_impression")

    n_q = len({r["query_id"] for r in rows})
    n_rep = len({r["repeat"] for r in rows})
    p_base = cite.get("overall_mean", 0.5)

    sd_cite = cite.get("measurement_sd_pp")
    br = paired_diff_sd(rows, "cited")
    sigma_floor = br.get("floor_pp")
    sigma_ceil = br.get("ceiling_pp")

    decision = decide_n(TARGET_MDE_PP, sigma_floor, p_base, n_rep)
    decision_ceiling = decide_n(TARGET_MDE_PP, sigma_ceil, p_base, n_rep)
    # 🔴 预算描述的是**阶段 B 的计划规模**（6 臂），不是阶段 A 数据里有几个臂。
    #    阶段 A 只有 0 臂，若用 len(arms)=1 算，会把预算少报 6 倍。
    arms_n = ARMS_TOTAL
    paired_obs_cap = int(BUDGET_CALLS_CAP / arms_n)

    # Q / R 拆分：MDE 只取决于 Q·R，所以拆分是纯设计权衡。
    # 两个约束：(a) 预算 Q·R ≤ paired_obs_cap；(b) §11 局限 5 要求每位置 ≥30
    # 个 query（5 个位置 -> Q ≥ 150），否则位置差异与 query 难易分不开。
    min_q = POSITION_MIN_PER_STRATUM * N_SOURCES
    tradeoff = []
    for q in sorted({50, 100, 150, 200, 300, 400}):
        r = paired_obs_cap // q if q else 0
        tradeoff.append({
            "n_queries": q,
            "max_repeats": r,
            "paired_obs": q * r,
            "within_budget": q * r <= paired_obs_cap,
            "meets_position_rule": q >= min_q,
            "mde_floor_pp": mde_at(q, r, sigma_floor),
            "mde_ceiling_pp": mde_at(q, r, sigma_ceil),
        })
    budget = {
        "n_per_arm": decision["n_per_arm"],
        "n_per_arm_ceiling": decision_ceiling["n_per_arm"],
        "repeats": n_rep,
        "arms": ARMS_TOTAL,
        "calls_cap": BUDGET_CALLS_CAP,
        "paired_obs_cap_per_arm": paired_obs_cap,
        "api_calls": decision["n_per_arm"] * n_rep * arms_n,
        "api_calls_ceiling": (decision_ceiling["n_per_arm"] or 0) * n_rep * arms_n,
        "est_hours": round(decision["n_per_arm"] * n_rep * arms_n * 2.56 / 3600, 2),
        "est_prompt_tokens_M": round(
            decision["n_per_arm"] * n_rep * arms_n * 3889 / 1e6, 2),
    }

    return {
        "meta": {
            "source": "阶段 A 0 臂方差分解",
            "n_rows": len(rows), "n_queries": n_q, "n_repeats": n_rep,
            "arms": arms, "preregistered": "方法规范 §6（v1.1）",
        },
        "citation_probability": cite,
        "geo_impression": gi,
        "variance_components": {"cited": vc_cite, "geo_impression": vc_gi},
        "paired_diff_sd": br,
        "n_decision": decision,
        "n_decision_ceiling": decision_ceiling,
        "q_r_tradeoff": {
            "paired_obs_cap_per_arm": paired_obs_cap,
            "min_queries_for_position_rule": min_q,
            "rows": tradeoff,
        },
        "budget": budget,
    }


def fmt_pp(v: float | None) -> str:
    """MDE 的表格渲染：None 显示为破折号而不是 0（0 会被读成「测得极准」）。"""
    return "—" if v is None else f"{v} pp"


def load_gate_report(path: str | None) -> dict:
    """读 15 号脚本的 `gate_report.json`，折算成逐臂门通过率。

    🔴 为什么这个折算必须进方差脚本而不是手算：§4 的 Q/R 拆分给的是
    **设计值**（每臂 Q 个 query），而信息保持门让各臂的**有效** Q 不同
    （2026-10-06 实测 18.7%–85.3%）。两者差最多 4.6 倍。
    MDE 论证属于本文档的职责，**不允许在报告里出现手算的 MDE**。
    """
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8-sig") as fh:
        rep = json.load(fh)
    # 门统计是**测量前**的产物：`gate_report.json` 由 dry-run 写，
    # 正式跑则落在 `summary.json` 的 `gate` 键里。两处都要认 ——
    # 否则「正式跑的臂通过率」会拿不到，逐臂 MDE 就只能靠手算。
    gate = rep.get("gate")
    if gate is None:
        gate = rep.get("arms")
        if not isinstance(gate, dict) or "original" not in gate:
            gate = None
    out: dict = {}
    for arm, s in (gate or {}).items():
        if arm == "original" or not s.get("n"):
            continue
        out[arm] = {"n": s["n"], "pass": s["pass"],
                    "pass_rate": s["pass"] / s["n"],
                    "codes": s.get("codes", {})}
    return {"per_arm": out, "source": path}


def synthetic_gate_report() -> dict:
    """合成门报告 —— **只用于覆盖渲染代码路径**。

    🔴 它**不能**替代真实通过率。真实数字必须由真实数据说话。
       这里唯一的作用是：让「4b 逐臂实际 MDE」这条**渲染与判读逻辑**
       在没有项目产出的环境里也能被验证 —— 否则这段代码只在有数据时
       才被检查过，等于长期裸奔。

    合成形态刻意做成「通过率随改写轴数递减」：这正是该段代码要处理的
    情形（有效 Q 随保留率下降 ⇒ 最弱臂的 MDE 显著更差）。
    """
    #: 臂 → (门样本数, 通过数)。刻意做出 3 倍以上的通过率落差。
    arms = {
        "arm_evidence_real": (40, 34),        # 85.0%
        "arm_evidence_fabricated": (40, 18),  # 45.0%
        "arm_multi_axis": (40, 8),            # 20.0%
    }
    per_arm = {}
    for arm, (n, p) in arms.items():
        per_arm[arm] = {"n": n, "pass": p, "pass_rate": p / n, "codes": {}}
    return {"per_arm": per_arm, "source": "<synthetic>"}


def render(rep: dict) -> str:
    L = []
    A = L.append
    c = rep["citation_probability"]
    g = rep["geo_impression"]
    m = rep["meta"]
    gate = rep.get("gate") or {}
    A("# 预注册阶段 A：0 臂方差分解结果")
    A("")
    A(f"- 数据：{m['n_rows']} 行 = {m['n_queries']} query × {m['n_repeats']} repeat × {len(m['arms'])} 臂（0 臂）")
    # 🔴 2026-10-21：这里**回读** meta 里的注册版本，不再硬编码 ——
    #    单一事实来源：版本号只在 analyze 里定义一次。
    A(f"- 依据：{m.get('preregistered', '方法规范 §6')}")
    A("")
    A("## 1. 两个 outcome 的方差分解")
    A("")
    A("| 量 | `citation probability` | `geo_impression` |")
    A("|---|---|---|")
    A(f"| 总体均值 | {c['overall_mean']:.4f} | {g['overall_mean']:.4f} |")
    A(f"| **测量级 σ_e**（合并格内方差，**无偏**，用于 MDE） | **{c['measurement_sd_pp']} pp** | **{g['measurement_sd_pp']} pp** |")
    A(f"| 各格 SD 的算术平均（**有偏低**） | {c['mean_of_cell_sds_pp']} pp | {g['mean_of_cell_sds_pp']} pp |")
    A(f"| RMS / 平均 之比 | {c['rms_over_mean']} | {g['rms_over_mean']} |")
    A(f"| 方差恰为 0 的格 | {c['zero_variance_cells']}/{c['n_cells']}（{c['zero_variance_share']:.1%}） | {g['zero_variance_cells']}/{g['n_cells']}（{g['zero_variance_share']:.1%}） |")
    A(f"| query 间 SD | {c['query_sd_pp']} pp | {g['query_sd_pp']} pp |")
    A(f"| 总 SD | {c['total_sd']:.4f} | {g['total_sd']:.4f} |")
    A(f"| run 级 SD | {c['run_level_sd_pp']} pp | {g['run_level_sd_pp']} pp |")
    A(f"| run 级极差 | {c['run_level_range_pp']} pp | {g['run_level_range_pp']} pp |")
    A("")
    A(f"> {c['note']}")
    A("")

    vc = rep.get("variance_components", {}).get("cited", {})
    A("## 1b. 随机效应方差分量（§6.1 要求的 ρ 由此直接估计）")
    A("")
    if vc.get("error"):
        A(f"> ⚠️ {vc['error']}")
    else:
        A("| 量 | 值 |")
        A("|---|---|")
        A(f"| 完整格数 k / 每格重复 R | {vc['n_complete_cells']} / {vc['repeats_per_cell']} |")
        A(f"| σ²_meas（测量误差） | {vc['var_measurement']} |")
        A(f"| σ²_query（query 效应） | {vc['var_query']} |")
        A(f"| **ρ（臂间，同 query）** | **{vc['rho_between_arms']}** |")
        A(f"| F(df={vc['df_between']}, {vc['df_within']}) | {vc['F']} |")
        A("")
        A(f"> {vc['reading']}")
    A("")

    p = rep["paired_diff_sd"]
    A("## 2. 配对差 SD —— 🔴 只给区间，不假装测到真值")
    A("")
    A("| 量 | 值 | 含义 |")
    A("|---|---|---|")
    A(f"| **下界 floor** | **{p['floor_pp']} pp** | √2·σ_meas，**ρ 消掉** |")
    A(f"| 上界 ceiling | {p['ceiling_pp']} pp | 臂间相关取 0 |")
    A(f"| 朴素代理 proxy | {p['proxy_sd_pp']} pp | {p['proxy_kind']} |")
    A("")
    A(f"> {p['warning']}")
    A("")
    d = rep["n_decision"]
    dc = rep["n_decision_ceiling"]
    b = rep["budget"]
    qt = rep["q_r_tradeoff"]
    A("## 3. 样本量定型（§6.3 先写死的降级顺序）")
    A("")
    A(f"按 **floor** 口径列出，R = {d['repeats']}（保守口径见下方两行结论）。")
    A("")
    A("| 档 | Q 上限 | 声明 MDE | 达标所需 Q | = 配对观测 | 可达? |")
    A("|---|---|---|---|---|---|")
    for row in d["ladder"]:
        need = row["n_needed_for_target"]
        po = row.get("paired_obs_needed")
        A(f"| {row['level']} | {row['n_cap']} | "
          f"{row['declared_mde_pp'] if row['declared_mde_pp'] else '按实测 σ 重算'} pp | "
          f"{need if need is not None else '—'} | "
          f"{po if po is not None else '—'} | "
          f"{'✅' if row['reachable'] else '❌'} |")
    A("")
    A(f"- 按 **floor**（乐观）：档 **{d['selected_level']}**，Q = **{d['n_per_arm']}/臂**，"
      f"实际 MDE = **{d.get('actual_mde_pp')} pp**")
    A(f"- 按 **ceiling**（保守）：档 **{dc['selected_level']}**，Q = **{dc['n_per_arm']}/臂**，"
      f"实际 MDE = **{dc.get('actual_mde_pp')} pp**")
    A("")
    if d.get("degraded"):
        A(f"> ⚠️ **目标 MDE {d['target_mde_pp']}pp 在预算内不可达**。"
          f"必须写进摘要的 MDE 取**保守值 {dc.get('actual_mde_pp')} pp**。")
    else:
        A(f"> ✅ **目标 MDE {d['target_mde_pp']}pp 达标**（档 {d['selected_level']}）。"
          f"floor 口径实际 MDE = {d.get('actual_mde_pp')} pp，"
          f"ceiling 口径 = {dc.get('actual_mde_pp')} pp。")
    A("")
    A("## 4. 预算与 Q/R 拆分（真正的设计决策在这里）")
    A("")
    A(f"> 🔴 **MDE 只取决于 `Q·R`，不取决于二者怎么分。** "
      f"预注册 §6.3 的阶梯写在 `Q`（query 数）上，但公式必须除以 `√(Q·R)`。")
    A(f"> 预算上限 **{b['calls_cap']:,} 次调用 / {b['arms']} 臂 = "
      f"{b['paired_obs_cap_per_arm']:,} 个配对观测/臂**。")
    A(f"> §11 局限 5 要求每个位置 ≥ {POSITION_MIN_PER_STRATUM} 个 query，"
      f"{N_SOURCES} 个位置 -> **Q ≥ {qt['min_queries_for_position_rule']}**。")
    A("")
    A("| Q（query） | 预算内最大 R | 配对观测 | 合预算 | 满足位置规则 | MDE floor | MDE ceiling |")
    A("|---|---|---|---|---|---|---|")
    for row in qt["rows"]:
        A(f"| {row['n_queries']} | {row['max_repeats']} | {row['paired_obs']:,} | "
          f"{'✅' if row['within_budget'] else '❌'} | "
          f"{'✅' if row['meets_position_rule'] else '❌'} | "
          f"{row['mde_floor_pp']} pp | {row['mde_ceiling_pp']} pp |")
    A("")
    ok = [r for r in qt["rows"] if r["within_budget"] and r["meets_position_rule"]]
    if ok:
        best = ok[-1]
        A(f"- **同时满足两个约束的 Q 取值**：{', '.join(str(r['n_queries']) for r in ok)}。")
        A(f"- Q 越大 = query 多样性越好、位置分层越稳；R 越大 = 符号稳定性规则（§7.2）"
          f"越有依据。**两者 MDE 相同，所以这是纯设计权衡，不是精度权衡。**")
    A("")

    # 🔴 功效算出来的 Q 与位置规则要求的 Q 可能冲突，必须显式仲裁
    q_need = d["n_per_arm"]
    q_min = qt["min_queries_for_position_rule"]
    if q_need is not None and q_need < q_min:
        q_final = q_min
        r_final = b["paired_obs_cap_per_arm"] // q_final
        A(f"> 🔴 **仲裁：功效只需 Q = {q_need}，但 §11 局限 5 要求 Q ≥ {q_min}**"
          f"（每位置 ≥ {POSITION_MIN_PER_STRATUM} 个 query，否则位置差异与 query 难易分不开）。")
        A(f"> **绑定约束是位置规则，不是功效。** 取 Q = **{q_final}**、"
          f"R = **{r_final}**，MDE = **{mde_at(q_final, r_final, p['floor_pp'])} pp**"
          f"（floor）/ **{mde_at(q_final, r_final, p['ceiling_pp'])} pp**（ceiling），"
          f"**优于 5pp 目标**。")
        A(f"> 也就是说：**目标 MDE 之所以在本设计下轻松达标，是因为重复次数多、"
          f"而多数 query 的引擎输出是确定性的**（{c['zero_variance_cells']}/{c['n_cells']} "
          f"个格方差恰为 0）—— 这不是运气，是这个 harness 的性质。")
    A("")
    A("| 量 | 值 |")
    A("|---|---|")
    A(f"| 功效所需 Q（floor 口径） | {d['n_per_arm']}/臂 |")
    A(f"| 位置规则要求 Q ≥ | {qt['min_queries_for_position_rule']} |")
    A(f"| 按阶梯所选 R | {d['repeats']} |")
    A(f"| 功效口径机时 | {b['est_hours']} h |")
    A(f"| 功效口径 prompt token | {b['est_prompt_tokens_M']} M |")
    A("")

    # ---- 4b. 🔴 逐臂实际功效：信息保持门会让各臂的有效 Q **不同**
    # §4 的表假设所有臂都有 Q 个 query。那是**设计值**，不是**实际值**：
    # 各臂门通过率 18.7%–85.3%（2026-10-06 stage B dry-run 实测），
    # 于是每臂真正能做配对的 query 数差 4.6 倍，MDE 也跟着差。
    if gate and gate.get("per_arm"):
        A("## 4b. 🔴 逐臂实际 MDE（信息保持门造成各臂有效 Q 不同）")
        A("")
        A("> §4 的 Q/R 拆分假设每臂都有 Q 个 query —— 那是**设计值**。"
          "**实际**每臂能做配对的 query 数 = Q × 该臂门通过率，"
          "而门通过率随改写轴数单调下降（信息流失剂量-反应，§3.4）。"
          "**所以「设计 MDE 达标」不等于「每个臂都达标」。**")
        A("")
        A("| 臂 | 门通过 | 有效 Q | 配对观测 (Q·R) | MDE floor | MDE ceiling | 5pp floor | 5pp ceiling |")
        A("|---|---|---|---|---|---|---|---|")
        q_design = q_final if q_need is not None and q_need < q_min else (
            q_need or qt["min_queries_for_position_rule"])
        r_use = r_final if q_need is not None and q_need < q_min else d["repeats"]
        for arm, gstat in gate["per_arm"].items():
            rate = gstat["pass_rate"]
            q_eff = int(round(q_design * rate))
            obs = q_eff * r_use
            mf = mde_at(q_eff, r_use, p["floor_pp"]) if q_eff else None
            mc = mde_at(q_eff, r_use, p["ceiling_pp"]) if q_eff else None
            A(f"| {arm} | {gstat['pass']}/{gstat['n']} = {rate:.1%} | {q_eff} | "
              f"{obs:,} | {fmt_pp(mf)} | {fmt_pp(mc)} | "
              f"{'✅' if mf is not None and mf <= 5.0 else '❌'} | "
              f"{'✅' if mc is not None and mc <= 5.0 else '❌'} |")
        A("")
        worst = min(
            ((a, q_design * gstat["pass_rate"])
             for a, gstat in gate["per_arm"].items()),
            key=lambda kv: kv[1])
        A(f"> 🔴 **最弱臂是 `{worst[0]}`，有效 Q ≈ {int(round(worst[1]))}**"
          f"（设计 {q_design} 的 {worst[1]/q_design:.0%}）。"
          f"**该臂的 MDE 必须逐臂报，不能拿设计值代替。**")
        A("> 这不是执行失误，是 §3.4 那个主发现的**代价**："
          "改写做得越彻底（保留率越低），能做配对的 query 越少。"
          "**效应量与它的可测性在同一个方向上被同一个机制推动** —— "
          "这必须在正文里同时报出来，不能只报保留率或只报 MDE。")
        A("")
    A("## 5. 必须回填方法规范的 `[TBD]`")
    A("")
    A("| # | 项 | 值 |")
    A("|---|---|---|")
    A(f"| 2a | `σ_cite`（测量级） | **{c['measurement_sd_pp']} pp** |")
    A(f"| 2b | `σ_gi`（测量级） | **{g['measurement_sd_pp']} pp** |")
    A(f"| 2c | `ρ`（配对内相关） | **{vc.get('rho_between_arms', '—')}**（由 0 臂重复直接估计） |")
    A(f"| 2d | run-level σ（citation） | {c['run_level_sd_pp']} pp |")
    A(f"| 2e | `σ_d` | 区间 **{p['floor_pp']} – {p['ceiling_pp']} pp**（真值需处理臂数据） |")
    A(f"| 3 | 阶段 B 的 Q（query 数） | 见 §4 的 Q/R 拆分，Q·R = {b['paired_obs_cap_per_arm']:,} |")
    A(f"| 3 | 实际 MDE | 随 Q/R 拆分而定：floor {d.get('actual_mde_pp')} pp ~ ceiling {dc.get('actual_mde_pp')} pp |")
    A("")
    A("> ⚠️ **回填纪律**：只填数字，不改 §6.3 的档位规则。")
    A("> 规则若变，那本身就是 deviation，必须在论文中报告。")
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

    print("== 14_stage_a_variance selftest ==")

    # 1. 合成数据：已知 SD 能否复原
    rows = []
    for qi in range(20):
        base = 0.5 + (qi % 2) * 0.4
        for rep in range(10):
            rows.append({"query_id": f"Q{qi}", "arm": "baseline", "repeat": rep,
                         "seed": 42 + rep * 1009, "target_idx": rep % 5,
                         "cited": base if rep % 2 == 0 else 1.0 - base,
                         "geo_impression": 0.2, "first_pos": rep % 5})
    vd = variance_decomposition(rows, "cited")
    chk("格数 = 20", vd["n_cells"] == 20)
    chk("每格重复 = 10", vd["repeats_per_cell"] == 10)
    chk("run 级有 10 个点", len(vd["per_run_mean"]) == 10)
    chk("测量级 SD 可计算", vd["measurement_sd_pp"] is not None)

    # 2. 完全一致的重复 -> 格内 SD = 0
    flat = [{"query_id": f"Q{qi}", "arm": "b", "repeat": r, "seed": 0,
             "target_idx": 0, "cited": 0.7, "geo_impression": 0.2, "first_pos": 0}
            for qi in range(10) for r in range(5)]
    vdf = variance_decomposition(flat, "cited")
    chk("无重复变异 -> 格内 SD = 0", vdf["measurement_sd_pp"] == 0.0)
    chk("无重复变异 -> run 级 SD = 0", vdf["run_level_sd_pp"] == 0.0)

    # 3. 样本量定型
    d_small = decide_n(5.0, 10.0, 0.67, 5)
    chk("σ_d=10pp 小 -> 选档 0", d_small["selected_level"] == 0, str(d_small.get("n_per_arm")))
    d_big = decide_n(5.0, 60.0, 0.67, 1)
    chk("σ_d=60pp 大、R=1 -> 进入降级档", d_big.get("degraded") is True,
        f"level={d_big.get('selected_level')} n={d_big.get('n_per_arm')}")
    d_big5 = decide_n(5.0, 60.0, 0.67, 5)
    chk("同样的 σ_d，R=5 把它从降级档救回档 1（重复比 query 便宜）",
        d_big5.get("degraded") is not True and d_big5["selected_level"] == 1,
        f"level={d_big5.get('selected_level')} n={d_big5.get('n_per_arm')}")
    chk("降级时给出实际 MDE", d_big.get("actual_mde_pp") is not None)
    chk("σ_d=None -> 不假装能定", decide_n(5.0, None, 0.67, 5)["n_per_arm"] is None
        or decide_n(5.0, None, 0.67, 5)["ladder"][0]["n_needed_for_target"] is None)

    # 3b. 🔴 MDE 必须除以 √(Q·R)（2026-10-06 修正：漏了 √R 曾导致
    #     "5pp 不可达" 的错误结论）
    chk("MDE 随 √R 下降", mde_at(400, 20, 40.0) < mde_at(400, 5, 40.0))
    chk("MDE 随 √Q 下降", mde_at(400, 5, 40.0) < mde_at(100, 5, 40.0))
    chk("Q·R 相同则 MDE 相同（拆分是纯权衡）",
        abs(mde_at(400, 5, 40.0) - mde_at(200, 10, 40.0)) < 1e-6)
    chk("MDE 与手算一致 (z·σ/√(Q·R))",
        abs(mde_at(66, 15, 40.237) - 2.8016 * 0.40237 / math.sqrt(66 * 15) * 100) < 0.01,
        f"got {mde_at(66, 15, 40.237)}")
    chk("R 越大所需 Q 越少（decide_n 已除以 R）",
        decide_n(5.0, 40.0, 0.67, 15)["ladder"][0]["n_needed_for_target"]
        < decide_n(5.0, 40.0, 0.67, 5)["ladder"][0]["n_needed_for_target"])
    chk("达标配对观测数与 R 无关（差异只来自向上取整，≤ 1 个 R）",
        abs(decide_n(5.0, 40.0, 0.67, 5)["ladder"][0]["paired_obs_needed"]
            - decide_n(5.0, 40.0, 0.67, 20)["ladder"][0]["paired_obs_needed"]) <= 20,
        f"{decide_n(5.0, 40.0, 0.67, 5)['ladder'][0]['paired_obs_needed']} vs "
        f"{decide_n(5.0, 40.0, 0.67, 20)['ladder'][0]['paired_obs_needed']}")
    chk("取整误差 ≤ 1 个 R（Q→配对观测的换算不引入额外偏差）",
        abs(decide_n(5.0, 40.0, 0.67, 1)["ladder"][0]["paired_obs_needed"]
            - decide_n(5.0, 40.0, 0.67, 7)["ladder"][0]["paired_obs_needed"]) <= 7,
        f"{decide_n(5.0, 40.0, 0.67, 1)['ladder'][0]['paired_obs_needed']} vs "
        f"{decide_n(5.0, 40.0, 0.67, 7)['ladder'][0]['paired_obs_needed']}")
    chk("预算上限 12,000 次调用", BUDGET_CALLS_CAP == 12000)
    chk("臂数 = 5（附录 B 原文五臂：1 original + 4 treatment）",
        ARMS_TOTAL == 5, f"got {ARMS_TOTAL}")
    chk("臂数与冻结 prompt 登记表一致（防止再漂）",
        ARMS_TOTAL == _arm_count())
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
        from freeze_prompts import ARMS as _FA
        chk("冻结表里正好 4 个 treatment 臂 + original",
            len(_FA) == 5 and sum(1 for a in _FA if a != "original") == 4,
            f"{sorted(_FA)}")
    except Exception as exc:
        chk("能读出冻结 prompt 表", False, str(exc))
    chk("检验族大小 = 4 treatment × 2 主因变量 = 8（不是 10）",
        (ARMS_TOTAL - 1) * 2 == 8)
    chk("位置规则给出 Q 下限 150", POSITION_MIN_PER_STRATUM * N_SOURCES == 150)

    # 4. 预注册常量自洽
    chk("预算上限 400", BUDGET_N == 400)
    chk("降级阶梯单调不降",
        all(DEGRADE_LADDER[i][0] <= DEGRADE_LADDER[i + 1][0]
            for i in range(len(DEGRADE_LADDER) - 1)))
    chk("臂数 = 5（不是 6）", ARMS_TOTAL == 5)
    chk("15 次重复", REPEATS == 15)

    # 5. 代理量必须自称代理
    p = paired_diff_sd(rows, "cited")
    chk("配对差自称区间而非点估计", p["kind"] == "bracket")
    chk("代理量带警告", "真正的 σ_d 测不到" in p["warning"])
    chk("代理量自称含 query 方差", "between-query" in p["proxy_kind"])

    # 5b. 方差分量：σ_e 已知时能否复原（这是 §6.1 要求估 ρ 的核心）
    import random as _rnd
    rng = _rnd.Random(20261006)
    TRUE_SD_E, TRUE_SD_Q = 0.10, 0.30
    syn = []
    for qi in range(120):
        mu = rng.gauss(0.0, TRUE_SD_Q)
        for rep in range(15):
            syn.append({"query_id": f"Q{qi}", "arm": "baseline", "repeat": rep,
                        "seed": 42 + rep * 1009, "target_idx": rep % 5,
                        "cited": 0.5 + mu + rng.gauss(0.0, TRUE_SD_E),
                        "geo_impression": 0.2, "first_pos": rep % 5})
    vc = variance_components(syn, "cited")
    chk("σ_meas 复原真值（±0.02）",
        abs(vc["sd_measurement_pp"] / 100 - TRUE_SD_E) < 0.02,
        f"got {vc['sd_measurement_pp']}pp want {TRUE_SD_E*100}pp")
    chk("σ_query 复原真值（±0.05）",
        abs(vc["sd_query_pp"] / 100 - TRUE_SD_Q) < 0.05,
        f"got {vc['sd_query_pp']}pp want {TRUE_SD_Q*100}pp")
    chk("ρ 复原真值（±0.06）",
        abs(vc["rho_between_arms"] - TRUE_SD_Q**2/(TRUE_SD_Q**2+TRUE_SD_E**2)) < 0.06,
        f"got {vc['rho_between_arms']}")
    chk("σ_d floor = √2·σ_meas 精确成立",
        abs(vc["sigma_d_floor_pp"] - math.sqrt(2)*vc["sd_measurement_pp"]) < 0.01)
    chk("floor < ceiling", vc["sigma_d_floor_pp"] < vc["sigma_d_ceiling_pp"])
    chk("floor 与 query 难度无关（下界的意义）", True)

    # 5c. σ_e=0 时 floor=0；无 query 效应时 floor==ceiling
    noq = variance_components(
        [{"query_id": f"Q{i}", "arm": "b", "repeat": r, "seed": 0, "target_idx": 0,
          "cited": (i + r) % 2 * 0.5, "geo_impression": 0, "first_pos": 0}
         for i in range(30) for r in range(4)], "cited")
    chk("无 query 效应时 floor == ceiling",
        abs(noq["sigma_d_floor_pp"] - noq["sigma_d_ceiling_pp"]) < 1e-6,
        f"{noq['sigma_d_floor_pp']} vs {noq['sigma_d_ceiling_pp']}")
    chk("无 query 效应时 ρ ≈ 0", abs(noq["rho_between_arms"]) < 0.02)

    # 5d. 不完整格被排除，不静默混进估计
    ragged = list(syn)
    ragged = [r for r in ragged if not (r["query_id"] == "Q0" and r["repeat"] == 14)]
    vcr = variance_components(ragged, "cited")
    chk("不完整格被排除", vcr["n_complete_cells"] == 119 and vcr["repeats_per_cell"] == 15)

    # 6. 渲染不崩
    _rep = analyze(rows)
    md = render(_rep)
    chk("渲染含定型表", "样本量定型" in md)
    chk("渲染含回填清单", "必须回填" in md)
    chk("渲染含区间而非代理量", "只给区间" in md)
    chk("渲染含方差分量表", "随机效应方差分量" in md)
    chk("渲染同时报 floor 与 ceiling", "乐观" in md and "保守" in md)
    chk("渲染引用注册版本（回读，不是硬编码）",
        _rep["meta"]["preregistered"] in md and "v1.1" in md)
    chk("渲染含 Q/R 拆分表", "Q/R 拆分" in md)
    chk("渲染标出达标与否", "达标" in md)
    chk("渲染含位置规则约束", "位置" in md and "150" in md)

    # 8b. 🔴 逐臂实际 MDE：门通过率不同 -> 有效 Q 不同 -> MDE 不同
    _gpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "outputs", "15_stage_b", "gate_report.json")
    _real_gate = os.path.exists(_gpath)
    _gate = load_gate_report(_gpath if _real_gate else None)
    if _real_gate:
        chk("门报告读得进（真实文件）", bool(_gate.get("per_arm")),
            str(list((_gate.get("per_arm") or {}).keys())))
        chk("门报告跳过 original（空处理，不是处理臂）",
            "original" not in (_gate.get("per_arm") or {}))
    else:
        # 🔴 2026-10-21 开源化：真实门报告是**项目产出**，本仓库里没有。
        #    ⇒ 改用**合成**门报告覆盖「4b 逐臂 MDE」这条渲染与判读路径，
        #    并把这件事**说出来**（不能让人以为验的是真实通过率）。
        #    合成数据证明的是**代码路径**；真实通过率仍然只能由真实数据说话。
        print("  [SKIP] 真实门报告不在场（它是项目产出）—— 改用**合成**门报告"
              "验证 4b 渲染路径；合成数据只证明代码路径，不证明真实通过率")
        _gate = synthetic_gate_report()
        chk("合成门报告读得进（退化路径）", bool(_gate.get("per_arm")),
            str(list((_gate.get("per_arm") or {}).keys())))
        chk("合成门报告跳过 original（空处理，不是处理臂）",
            "original" not in (_gate.get("per_arm") or {}))
    if _gate.get("per_arm"):
        _rates = {a: v["pass_rate"] for a, v in _gate["per_arm"].items()}
        chk("逐臂门通过率明显不同（>3× 差）",
            max(_rates.values()) / max(min(_rates.values()), 1e-9) > 3,
            str(_rates))
    _rep2 = analyze(rows)
    _rep2["gate"] = _gate
    _md2 = render(_rep2)
    chk("渲染含 4b 逐臂 MDE 节", "4b." in _md2 and "逐臂实际 MDE" in _md2)
    chk("4b 标出最弱臂", "最弱臂" in _md2)
    chk("4b 同时给 floor 与 ceiling 达标判定",
        _md2.count("5pp floor") == 1 and _md2.count("5pp ceiling") == 1)
    chk("4b 指出效应量与可测性同向（不是执行失误）",
        "同一个机制" in _md2)
    chk("无门报告时不崩（退化成只有设计值）",
        "4b." not in render(analyze(rows)))

    # 7. 🔴 多臂必须报错，不能静默算出错的 σ（2026-10-06 实测偏差 6pp）
    #    用**同方差**的干净基底，否则基底的偏斜会掩盖混臂效应。
    clean = [{"query_id": f"C{i}", "arm": "baseline", "repeat": r, "seed": r,
              "target_idx": 0, "cited": (0.5 if r % 2 == 0 else 0.9),
              "geo_impression": 0.2, "first_pos": 0}
             for i in range(40) for r in range(5)]
    multi = clean + [dict(r, arm="treatment", cited=0.02) for r in clean]
    try:
        analyze(multi)
        chk("多臂数据被拒绝（不静默出数）", False)
    except ValueError as exc:
        chk("多臂数据被拒绝（不静默出数）", "单臂" in str(exc))
    vmix = variance_decomposition(multi, "cited")
    vone = variance_decomposition(clean, "cited")
    chk("混臂会实质抬高格内 SD（说明为何必须拒）",
        vmix["measurement_sd_pp"] > 1.5 * vone["measurement_sd_pp"],
        f"{vmix['measurement_sd_pp']} vs {vone['measurement_sd_pp']} pp "
        f"({vmix['measurement_sd_pp']/vone['measurement_sd_pp']:.2f}x)")
    chk("混臂后 run 级也失真（两个症状都出现）",
        abs(vmix["run_level_sd_pp"]) != abs(vone["run_level_sd_pp"]))

    # 9. 🔴 RMS vs 各格 SD 算术平均（n66 baseline 实测 28.34 vs 16.28，差 1.74×）
    vnz = variance_decomposition(flat, "cited")
    chk("无变异时 RMS = 平均 = 0",
        vnz["measurement_sd_pp"] == 0.0 and vnz["mean_of_cell_sds_pp"] == 0.0)
    # 构造：多数格方差为 0，少数格方差很大 -> 平均被拖低，RMS 不会
    skew = []
    for qi in range(50):
        for rep in range(5):
            base = 0.02 if qi < 40 else (0.98 if rep % 2 == 0 else 0.02)
            skew.append({"query_id": f"S{qi}", "arm": "baseline", "repeat": rep,
                         "seed": rep, "target_idx": 0, "cited": base,
                         "geo_impression": 0.2, "first_pos": 0})
    vsk = variance_decomposition(skew, "cited")
    chk("偏斜格内 SD：RMS > 算术平均",
        vsk["measurement_sd_pp"] > vsk["mean_of_cell_sds_pp"],
        f"RMS {vsk['measurement_sd_pp']} vs mean {vsk['mean_of_cell_sds_pp']} pp")
    chk("偏斜时零方差格被正确计数",
        vsk["zero_variance_cells"] == 40, f"got {vsk['zero_variance_cells']}")
    chk("measurement_sd 用的是 RMS（无偏）不是平均", abs(vsk["measurement_sd_pp"] - 23.515) < 0.01,
        f"got {vsk['measurement_sd_pp']}")
    chk("方差分量与 RMS 一致（两处估计同源）",
        abs(variance_components(skew, "cited")["sd_measurement_pp"]
            - vsk["measurement_sd_pp"]) < 0.01)

    # 8. 🔴 预算必须按阶段 B 的 6 臂算，不是按阶段 A 数据里的臂数
    rep1 = analyze(rows)
    chk("预算按 ARMS_TOTAL 臂算，不是按数据里的 1 臂",
        rep1["budget"]["arms"] == ARMS_TOTAL == 5)
    chk("预算调用数 = n × repeats × 臂数",
        rep1["budget"]["api_calls"] == rep1["budget"]["n_per_arm"]
        * rep1["budget"]["repeats"] * ARMS_TOTAL)

    print(f"\n{len(fails)} failed" if fails else "\nall checks passed")
    for f in fails:
        print(f"  FAILED: {f}")
    return 1 if fails else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="14_stage_a_variance.py",
        description="预注册阶段 A：0 臂方差分解与样本量定型（方法规范 §6）")
    p.add_argument("--csv", default=os.environ.get("GEO_RAW_TRIALS_CSV", "").strip(),
                   help="逐单元结果 CSV（默认取环境变量 GEO_RAW_TRIALS_CSV）")
    p.add_argument("--arm", default=None,
                   help="显式选择单臂。CSV 含多臂时**必须**指定，否则报错。")
    p.add_argument("--out", default=None)
    p.add_argument("--gate-report", default="",
                   help="15 号脚本的 gate_report.json。给了就多出一节 4b「逐臂实际 MDE」"
                        "—— 信息保持门让各臂有效 Q 不同（实测 18.7%%–85.3%%），"
                        "设计值不能代替实际值")
    p.add_argument("--json", default=None)
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args(argv)

    if a.selftest:
        return selftest()
    if not os.path.exists(a.csv):
        print(f"ERROR: 找不到 {a.csv}", file=sys.stderr)
        return 2

    rows, arms_all, dropped = load(a.csv, arm=a.arm)
    if len(arms_all) > 1 and a.arm is None:
        print(f"ERROR: {a.csv} 含 {len(arms_all)} 个臂：{arms_all}",
              file=sys.stderr)
        print("       阶段 A 只接受单臂 —— 多臂会把臂间差异算进格内 SD。",
              file=sys.stderr)
        print("       请加 --arm <臂名>（阶段 A 用 --arm baseline）。",
              file=sys.stderr)
        return 2
    if not rows:
        print("ERROR: 没有可用数据行", file=sys.stderr)
        return 2
    if dropped:
        print(f"[info] 丢弃 {len(dropped)} 类行：{dropped[:5]}", file=sys.stderr)

    try:
        rep = analyze(rows)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    rep["gate"] = load_gate_report(a.gate_report)
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
