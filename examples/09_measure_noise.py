"""09_measure_noise.py — 测量 GEO harness 自身的噪声底噪，并给出"要多少样本才敢下结论"

这个脚本解决一个很具体、也很要命的问题：
    **你跑出来的那个差异，到底是"策略起作用"，还是"我又跑了一遍"？**

做法是方差分解。对每个 (query, 臂)，其他什么都不变、只是**重新跑一遍**，
看它的 geo_impression 变了多少 —— 那个变化幅度就是噪声底噪。
论文里报一个提升之前，必须先知道自己的测量精度是 0.1% 还是 5%。

输出：
  - 各臂跨 run 的均值 / 极差 / 符号是否稳定
  - 噪声底噪（同条件重跑的标准差）与总方差的比例
  - 给定目标效应量，需要多少样本（power analysis）
  - 位置偏置分组统计（按 target_idx）

用法：
    # 分析多次 run（目录或 raw_trials.csv 都行）
    python examples/09_measure_noise.py data/run1 data/run2

    # 用通配符分析所有 run
    python examples/09_measure_noise.py --glob "data/run*"

    # 分析单个 run 只能给出组内 SD 和功效，噪声底噪会标为不可估
    python examples/09_measure_noise.py data/run1

    # 指定"我想检测出多少个百分点的差异"来反推样本量
    python examples/09_measure_noise.py --glob "data/run*" --target-delta 3
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------- 统计工具

def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def stdev(xs):
    """样本标准差 ddof=1"""
    n = len(xs)
    if n < 2:
        return float("nan")
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def t_quantile_two_sided(alpha, df):
    """双尾 t 临界值（查表 + 线性插值，避免依赖 scipy）"""
    table = {
        0.10: {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 8: 1.860,
               10: 1.812, 12: 1.782, 15: 1.753, 20: 1.725, 25: 1.708, 30: 1.697,
               40: 1.684, 60: 1.671, 120: 1.658},
        0.05: {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 8: 2.306,
               10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042,
               40: 2.021, 60: 2.000, 120: 1.980},
        0.02: {1: 31.821, 2: 6.965, 3: 4.541, 4: 3.747, 5: 3.365, 6: 3.143, 8: 2.896,
               10: 2.764, 12: 2.681, 15: 2.602, 20: 2.528, 25: 2.485, 30: 2.457,
               40: 2.423, 60: 2.390, 120: 2.358},
        0.01: {1: 63.657, 2: 9.925, 3: 5.841, 4: 4.604, 5: 4.032, 6: 3.707, 8: 3.355,
               10: 3.169, 12: 3.055, 15: 2.947, 20: 2.845, 25: 2.787, 30: 2.750,
               40: 2.704, 60: 2.660, 120: 2.617},
    }
    rows = sorted(table)
    if alpha not in table:
        raise ValueError(f"alpha 只支持 {rows}")
    row = table[alpha]
    keys = sorted(row)
    if df <= keys[0]:
        return row[keys[0]]
    if df >= keys[-1]:
        return row[keys[-1]]
    for a, b in zip(keys, keys[1:]):
        if a <= df <= b:
            lo, hi = row[a], row[b]
            return lo + (hi - lo) * (df - a) / (b - a)
    return row[keys[-1]]


def n_required(sigma, delta, alpha=0.05, power=0.80):
    """两组等样本量 t 检验，检测出 delta 所需的每组样本量（取保守的 4σ² 近似 + t 修正迭代）"""
    if sigma <= 0 or delta <= 0:
        return None
    z_a = {0.05: 1.960, 0.02: 2.326, 0.01: 2.576}[alpha]
    z_b = {0.80: 0.842, 0.90: 1.282, 0.95: 1.645}[power]
    n = 2 * ((z_a + z_b) * sigma / delta) ** 2
    for _ in range(40):  # 用 t 分布修正（大方差时 t 更大，需要更多样本）
        df = max(1, 2 * n - 2)
        t = t_quantile_two_sided(alpha, df)
        n_new = 2 * ((t + z_b) * sigma / delta) ** 2
        if abs(n_new - n) < 0.5:
            return int(math.ceil(n_new))
        n = n_new
    return int(math.ceil(n))


def n_required_paired(sd, delta, alpha=0.05, power=0.80):
    """配对 t 检验所需样本量（sd 是配对差值的标准差）"""
    if sd <= 0 or delta <= 0:
        return None
    z_a = {0.05: 1.960, 0.02: 2.326, 0.01: 2.576}[alpha]
    z_b = {0.80: 0.842, 0.90: 1.282, 0.95: 1.645}[power]
    n = ((z_a + z_b) * sd / delta) ** 2
    for _ in range(40):
        df = max(1, n - 1)
        t = t_quantile_two_sided(alpha, df)
        n_new = ((t + z_b) * sd / delta) ** 2
        if abs(n_new - n) < 0.5:
            return int(math.ceil(n_new))
        n = n_new
    return int(math.ceil(n))


# ---- t 分布双尾 p 值（不依赖 scipy）

def _betacf(a, b, x):
    MAXIT, EPS, FPMIN = 300, 3.0e-12, 1.0e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def _betai(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbeta) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta) * _betacf(b, a, 1.0 - x) / b


def t_sf_two_sided(t, df):
    if df <= 0:
        return 1.0
    return _betai(df / 2.0, 0.5, df / (df + t * t))


# ---------------------------------------------------------------- 载入

def load_runs(paths):
    """读入多个 run 目录的 raw_trials.csv，按 (query_id, arm, target_idx) 聚成 (run → 值)"""
    data = []  # [{'run':..., 'rows':[...]}]
    for p in paths:
        d = Path(p)
        csvs = list(d.glob("raw_trials*.csv")) if d.is_dir() else [d]
        for c in csvs:
            rows = []
            for r in csv.DictReader(c.read_bytes().decode("utf-8-sig").splitlines()):
                if r.get("error"):
                    continue
                if "geo_impression" not in r:
                    continue
                rows.append({
                    "query_id": r["query_id"],
                    "arm": r["arm"],
                    "repeat": int(r.get("repeat") or 0),
                    "target_idx": int(r.get("target_idx") or 0),
                    "geo_impression": float(r["geo_impression"] or 0.0),
                    "cited": int(float(r.get("cited") or 0)),
                    "mentioned": int(float(r.get("mentioned") or 0)),
                })
            if rows:
                data.append({"run": d.name, "file": c.name, "rows": rows})
    return data


# ---------------------------------------------------------------- 分析

def analyse(runs, target_delta, alpha, power):
    out = {"runs": [], "arms": {}, "noise": None, "position": {}}

    for r in runs:
        gi = [x["geo_impression"] for x in r["rows"]]
        out["runs"].append({
            "run": r["run"], "n": len(r["rows"]),
            "gi_mean": round(mean(gi), 4), "gi_sd": round(stdev(gi), 4),
            "cited_rate": round(mean([x["cited"] for x in r["rows"]]), 4),
            "baseline_gi": round(mean([x["geo_impression"] for x in r["rows"] if x["arm"] == "baseline"]), 4),
        })

    # ---- 跨 run 的噪声底噪：固定 (query, arm)，只变 run
    by_cell = defaultdict(dict)  # (query, arm) -> {run: [values]}
    for r in runs:
        for x in r["rows"]:
            by_cell[(x["query_id"], x["arm"], x["repeat"])].setdefault(r["run"], []).append(
                x["geo_impression"]
            )
    diffs = []
    for cell, per_run in by_cell.items():
        if len(per_run) < 2:
            continue
        run_means = [mean(v) for v in per_run.values()]
        diffs.extend(run_means)
        out.setdefault("_cells", []).append({
            "cell": "|".join(map(str, cell)), "run_means": [round(v, 4) for v in run_means],
            "spread": round(max(run_means) - min(run_means), 4),
        })

    if len(runs) >= 2:
        per_cell_spread = [c["spread"] for c in out["_cells"]]
        # by_cell 的结构是 {(query,arm,repeat): {run: [gi, ...]}}，取均值前要先摊平两层
        all_vals = [x for cell in by_cell.values() for vals in cell.values() for x in vals]
        # 每个 cell = 固定 (query, arm, repeat) 在各 run 上的取值集合，取其均值作为"这个 cell 的值"
        cell_means = [mean([x for vals in v.values() for x in vals]) for v in by_cell.values()]
        noise_sd = stdev(per_cell_spread) if len(per_cell_spread) > 1 else float("nan")
        total_sd = stdev(all_vals) if len(all_vals) > 1 else float("nan")
        between_sd = stdev(cell_means) if len(cell_means) > 1 else float("nan")
        # 同一 cell 内的 run 间差异，就是纯生成噪声
        within_run_sd = float("nan")
        run_sds = []
        for r in runs:
            run_sds.append(stdev([x["geo_impression"] for x in r["rows"]]))
        run_sds = [s for s in run_sds if not math.isnan(s)]
        within_run_sd = mean(run_sds) if run_sds else float("nan")

        out["noise"] = {
            "n_cells": len(out["_cells"]),
            "mean_cell_spread": round(mean(per_cell_spread), 4) if per_cell_spread else None,
            "max_cell_spread": round(max(per_cell_spread), 4) if per_cell_spread else None,
            "between_run_sd": round(noise_sd, 4) if not math.isnan(noise_sd) else None,
            "total_sd": round(total_sd, 4) if not math.isnan(total_sd) else None,
            "between_cell_sd": round(between_sd, 4) if not math.isnan(between_sd) else None,
            "within_run_sd": round(within_run_sd, 4) if not math.isnan(within_run_sd) else None,
        }
        if noise_sd and total_sd and not math.isnan(total_sd):
            out["noise"]["noise_share_of_variance"] = round((noise_sd ** 2) / (total_sd ** 2), 3)

    # ---- 各臂跨 run 表现
    arms = sorted({x["arm"] for r in runs for x in r["rows"]})
    base_per_run = {r["run"]: mean([x["geo_impression"] for x in r["rows"] if x["arm"] == "baseline"])
                    for r in runs}
    for arm in arms:
        per_run = []
        for r in runs:
            v = [x["geo_impression"] for x in r["rows"] if x["arm"] == arm]
            if v:
                per_run.append((r["run"], mean(v), v))
        if not per_run:
            continue
        means = [p[1] for p in per_run]
        deltas = []
        for run_name, m, _v in per_run:
            b = base_per_run.get(run_name)
            if b is not None:
                deltas.append((m - b) * 100)
        signs = [1 if d > 0 else (-1 if d < 0 else 0) for d in deltas]
        nonzero = [s for s in signs if s != 0]
        out["arms"][arm] = {
            "n_per_run": [p[2].__len__() for p in per_run],
            "gi_per_run": [round(m, 4) for m in means],
            "gi_mean": round(mean(means), 4),
            "gi_range": round(max(means) - min(means), 4),
            "delta_pp_per_run": [round(d, 2) for d in deltas],
            "delta_pp_mean": round(mean(deltas), 2) if deltas else None,
            "delta_pp_sd": round(stdev(deltas), 2) if len(deltas) > 1 else None,
            "sign_stable": len(set(nonzero)) == 1 if nonzero else None,
            "all_same_sign": ("—（基线）" if arm == "baseline" or not nonzero else
                              "全正" if all(s == 1 for s in nonzero) else
                              "全负" if all(s == -1 for s in nonzero) else
                              "有翻转"),
        }

    # ---- 功效分析
    sig = out["noise"]["between_run_sd"] if out["noise"] and out["noise"].get("between_run_sd") else None
    allvals = [x["geo_impression"] for r in runs for x in r["rows"]]
    sig_fallback = stdev(allvals) if len(allvals) > 1 else None
    sigma = sig or sig_fallback
    power_rows = []
    # ⚠️ 单位：geo_impression 是 0–1 的比例，target_delta 传的是"百分点"。
    #    3 pp = 0.03，这个换算漏了会把 n 少算 100 倍（曾导致每臂 n=1 的荒谬结果）。
    for frac in (0.5, 1.0, 2.0, 3.0, 5.0):
        d_pp = target_delta * frac
        d = d_pp / 100.0
        n = n_required(sigma, d, alpha, power) if sigma else None
        power_rows.append({"delta_pp": round(d_pp, 2), "n_per_arm": n,
                           "total_calls": (n * 2 * 5) if n else None})
    out["power"] = {
        "sigma_used": round(sigma, 4) if sigma else None,
        "sigma_source": "跨 run 噪声" if sig else "组内 SD（只有单次 run，噪声高估）",
        "alpha": alpha, "power": power, "rows": power_rows,
    }

    # ---- 配对分析（重要）：所有臂跑的是同一批 query，应该在 query 内部比较
    #    非配对做法把 query 间方差也算进误差，功效被严重低估。
    q_arm = defaultdict(list)
    for r in runs:
        for x in r["rows"]:
            q_arm[(x["query_id"], x["arm"])].append(x["geo_impression"])
    base_by_q = {q: mean(v) for (q, a), v in q_arm.items() if a == "baseline"}
    paired = {}
    for arm in arms:
        if arm == "baseline":
            continue
        diffs = []
        for (q, a), v in q_arm.items():
            if a != arm or q not in base_by_q:
                continue
            # 用该 query 上所有 run/repeat 的均值做配对
            diffs.append((mean(v) - base_by_q[q]) * 100)
        if len(diffs) < 2:
            continue
        md, sdv = mean(diffs), stdev(diffs)
        se = sdv / math.sqrt(len(diffs)) if sdv == sdv else 0.0
        t = md / se if se > 0 else 0.0
        df = len(diffs) - 1
        p = t_sf_two_sided(t, df) if se > 0 else 1.0
        paired[arm] = {
            "n_queries": len(diffs),
            "deltas_pp": [round(d, 2) for d in diffs],
            "mean_delta_pp": round(md, 2),
            "sd_delta_pp": round(sdv, 2) if sdv == sdv else None,
            "se": round(se, 3),
            "t": round(t, 3), "df": df, "p": round(p, 4),
            "significant": bool(p < alpha),
            "n_required_for_observed": n_required_paired(sdv, abs(md), alpha, power)  # sdv 与 md 同为 pp，不做 /100
            if sdv == sdv and abs(md) > 0 else None,
        }
    out["paired"] = paired
    out["between_query_sd"] = round(stdev(list(base_by_q.values())), 4) if len(base_by_q) > 1 else None

    # ---- 位置偏置
    pos = defaultdict(list)
    for r in runs:
        for x in r["rows"]:
            if x["arm"] == "baseline" and x["target_idx"]:
                pos[x["target_idx"]].append(x["geo_impression"])
    for k in sorted(pos):
        v = pos[k]
        out["position"][k] = {"n": len(v), "gi_mean": round(mean(v), 4),
                              "gi_sd": round(stdev(v), 4) if len(v) > 1 else None}
    out.pop("_cells", None)
    return out


# ---------------------------------------------------------------- 报告

def render(out, paths, target_delta):
    L = ["# GEO harness 噪声底噪报告", "",
         f"- 分析目录：{', '.join(Path(p).name for p in paths)}",
         f"- run 数：{len(out['runs'])}",
         f"- 目标效应量（用于反推样本量）：{target_delta} pp（绝对百分点）", ""]

    L += ["## 1. 每次 run 的整体情况", "",
          "| run | n | baseline geo_impression | 全部 geo_impression 均值 | SD | 被引用率 |",
          "|---|---|---|---|---|---|"]
    for r in out["runs"]:
        L.append(f"| `{r['run']}` | {r['n']} | {r['baseline_gi']} | {r['gi_mean']} | {r['gi_sd']} | {r['cited_rate']} |")

    nz = out.get("noise")
    L += ["", "## 2. 噪声底噪（最关键的一张表）", ""]
    if not nz:
        L += ["> 只有 1 个 run，**无法估计噪声底噪**。至少跑 2 次同样的配置（其他都不变）才能算。", ""]
    else:
        L += [
            "对每个 `(query, 臂, repeat)` 三元组，只改变**重跑这件事**，看它的值变了多少 —— "
            "那个变化就是测量本身的噪声。", "",
            "| 量 | 值 | 含义 |", "|---|---|---|",
            f"| 同条件重跑的平均极差 | {nz['mean_cell_spread']} | 同一份活再干一遍，数值漂多少 |",
            f"| 同条件重跑的最大极差 | {nz['max_cell_spread']} | 最坏情况 |",
            f"| **跨 run 标准差** | **{nz['between_run_sd']}** | **用来做功效分析的 σ** |",
            f"| 组内总标准差 | {nz['total_sd']} | 你在报告里看到的那个 SD |",
            f"| query 间标准差 | {nz['between_cell_sd']} | 真正想测的信号 |",
        ]
        if nz.get("noise_share_of_variance") is not None:
            L.append(f"| 噪声占总方差比例 | {nz['noise_share_of_variance']:.1%} | 越接近 1 说明这个实验设计越没用 |")
        L += ["",
              "**读法**：组内总 SD 里同时混着「真实的 query 差异」和「单纯的重跑噪声」。"
              f"如果跨 run 噪声占的比例很高，说明这 {len(out['runs'])} 次 run 之间的差异基本是噪声，"
              "不能拿来当结论。", ""]

    L += ["## 3. 各臂跨 run 的稳定性", "",
          "| 臂 | 各 run geo_impression | 极差 | Δ vs baseline（每个 run） | Δ 均值 | Δ SD | 符号 |",
          "|---|---|---|---|---|---|---|"]
    for arm, a in out["arms"].items():
        dps = " / ".join(f"{d:+.2f}" for d in a["delta_pp_per_run"])
        L.append(
            f"| `{arm}` | {' / '.join(str(x) for x in a['gi_per_run'])} | {a['gi_range']} | "
            f"{dps} | {a['delta_pp_mean']} | {a['delta_pp_sd']} | {a['all_same_sign'] or '—'} |"
        )
    L += ["", "**符号是否稳定是唯一值得看的列。** 同一配置重跑，如果某个臂一会儿正一会儿负，"
          "那它的效应量小于你的噪声，任何基于它的结论都不成立。", ""]

    pw = out.get("power")
    if pw:
        L += ["## 4. 配对分析（★ 这才是对的做法）", ""]
        if out.get("between_query_sd") is not None:
            L += [f"所有臂跑的是**同一批 query**，所以应该在 query 内部比较（配对），"
                  f"而不是把两组当独立样本。query 间标准差 = {out['between_query_sd']} —— "
                  f"这个量在非配对分析里会被算进误差。", ""]
        L += ["| 臂 | 配对 query 数 | 逐 query Δ（pp） | 均值 Δ | SD | SE | t | p | 显著? |",
              "|---|---|---|---|---|---|---|---|---|"]
        for arm, a in out.get("paired", {}).items():
            L.append(
                f"| `{arm}` | {a['n_queries']} | {' / '.join(f'{d:+.2f}' for d in a['deltas_pp'])} | "
                f"**{a['mean_delta_pp']:+.2f}** | {a['sd_delta_pp']} | {a['se']} | {a['t']} | {a['p']} | "
                f"{'✅ 是' if a['significant'] else '❌ 否'} |"
            )
        L.append("")
        for arm, a in out.get("paired", {}).items():
            if a.get("n_required_for_observed"):
                L.append(f"- `{arm}`：要把它现在这个 {a['mean_delta_pp']:+.2f} pp 的效应测到显著，"
                         f"需要 **{a['n_required_for_observed']} 个 query**（现在 {a['n_queries']} 个）。")
        L.append("")

        L += ["## 5. 同一组效应量下，配对 vs 非配对各需要多少样本", "",
              f"σ_非配对 = {pw['sigma_used']}（跨 run 噪声）；"
              f"σ_配对 = 各臂配对差值的 SD（见上表）。"
              f"α={pw['alpha']}，power={pw['power']}。", "",
              "| 想检测出的差异 | 非配对：每臂 query 数 | 配对：所需 query 数 | 差距 |",
              "|---|---|---|---|"]
        for arm, a in out.get("paired", {}).items():
            if not a.get("sd_delta_pp"):
                continue
            for frac in (1.0, 2.0, 3.0):
                d_pp = abs(a["mean_delta_pp"]) * frac
                if d_pp <= 0:
                    continue
                n_p = n_required_paired(a["sd_delta_pp"], d_pp, pw["alpha"], pw["power"])
                n_u = n_required(pw["sigma_used"], d_pp / 100.0, pw["alpha"], pw["power"])
                if n_p and n_u:
                    L.append(f"| {d_pp:.1f} pp（`{arm}`） | {n_u} | {n_p} | 省 {n_u / n_p:.0f}× |")
        L += ["", "**配对显著更省样本**，因为 query 本身的难易被消掉了 —— "
              "这正是 arXiv:2605.25517 坚持 repeated paired comparisons 的原因。"
              "报论文时请用配对结果，并在方法里写明"
              "「每个 query 都有且只有一个被优化的来源、比较在 query 内配对进行」。", ""]

    if out.get("position"):
        L += ["## 6. 位置偏置（按目标文档所在位置分组，仅 baseline 臂）", "",
              "| 位置 | n | geo_impression 均值 | SD |", "|---|---|---|---|"]
        for k, v in sorted(out["position"].items(), key=lambda x: int(x[0])):
            L.append(f"| {k} | {v['n']} | {v['gi_mean']} | {v['gi_sd']} |")
        gis = [v["gi_mean"] for v in out["position"].values()]
        ns = [v["n"] for v in out["position"].values()]
        if len(gis) > 1:
            spread = max(gis) - min(gis)
            L += ["", f"位置间极差 {round(spread, 4)}。"]
            if min(ns) < 15:
                L += [
                    "",
                    f"> ⚠️ **这一节现在还不能下结论。** 每个位置只有 {min(ns)}–{max(ns)} 个样本，"
                    "而且 `doc_pos = (item 序号 + repeat) % 文档数` 的轮换方式意味着"
                    "**位置和 query 在这个样本量下高度混淆** —— 某个位置上的低分很可能只是"
                    "那两三个 query 本身难，而不是位置造成的。",
                    "",
                    "> 要真正测位置偏置，需要 `--limit` 大到每个位置能分到 30+ 个**不同 query**，"
                    "并用 08 号脚本的 `target_idx` 列做分组回归（而不是比均值）。"
                    "这也是 arXiv:2605.25517 用 25.2 万次试验做的事。",
                ]
            else:
                noise = (out.get("noise") or {}).get("between_run_sd") or 0.0
                L.append("**位置偏置与噪声底噪同量级，位置反平衡是必需的。**"
                         if spread > noise / 2 else "位置偏置目前看小于噪声底噪，暂不影响结论。")
        L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------- 自检
def selftest() -> int:
    """注入已知真值的合成数据，复原噪声底噪与样本量反解。

    🔴 **为什么必须有这个入口**：本脚本原先没有自检入口，打包器只能把它记成
    「无自检入口」（验证缺口）—— 于是它在包里**没有任何可执行的自我证明**。
    自检要钉住三件事：
      ① 跨 run 噪声能被测出来，且方向与量级对得上；
      ② **单 run 时噪声底噪必须标为不可估**（不能拿组内 SD 冒充跨 run 噪声）；
      ③ 样本量反解的**单位换算**（百分点 vs 比例）没漏 100 倍。
    """
    import tempfile

    checks = []

    def ck(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    def _write_run(d, offset_for):
        """写一个 run 的 raw_trials.csv。offset_for(q) 给出该 query 的 run 间偏移。"""
        p = Path(d) / "raw_trials.csv"
        lines = ["query_id,arm,repeat,target_idx,geo_impression,cited,mentioned"]
        for q in range(40):
            base = 0.30 + 0.01 * (q % 10)     # query 间异质性（σ_b 的来源）
            gi = base + offset_for(q)         # run 间噪声（σ_e 的来源）
            lines.append(f"Q{q:03d},baseline,0,0,{gi:.6f},1,1")
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return p

    with tempfile.TemporaryDirectory() as td:
        d1, d2 = Path(td) / "run1", Path(td) / "run2"
        d1.mkdir()
        d2.mkdir()
        _write_run(d1, lambda q: 0.0)
        # 偏移**逐 query 变化**（确定性，不用随机数）：否则所有 cell 的 spread
        # 相同，跨 run SD 会算成 0，测不出噪声。
        _write_run(d2, lambda q: 0.025 * ((q % 5) - 2))

        runs = load_runs([str(d1), str(d2)])
        ck("两个 run 都被读到", len(runs) == 2, str([r["run"] for r in runs]))
        out = analyse(runs, 3.0, 0.05, 0.80)
        noise = out.get("noise") or {}
        ck("两 run 时噪声底噪被分解出来", bool(noise))
        ck("cell 数 = query 数（40）", noise.get("n_cells") == 40, str(noise.get("n_cells")))
        ck("跨 run SD > 0（注入的 run 间差异被测到）",
           (noise.get("between_run_sd") or 0) > 0, str(noise.get("between_run_sd")))
        ck("平均 cell spread > 0（注入的偏移量级对得上）",
           (noise.get("mean_cell_spread") or 0) > 0, str(noise.get("mean_cell_spread")))
        share = noise.get("noise_share_of_variance")
        ck("噪声方差占比在 [0,1]", share is not None and 0 <= share <= 1, str(share))

        rows = out["power"]["rows"]
        ck("功效表给出 5 档效应量", len(rows) == 5, str(len(rows)))
        ns = [r["n_per_arm"] for r in rows]
        ck("n_per_arm 全部为正", all(n and n > 0 for n in ns), str(ns))
        ck("效应量越小需要的 n 越大（单调）", ns[0] > ns[-1], str(ns))
        ck("最小效应量档的 n 远大于 1（单位没漏 100 倍）", ns[0] > 1, str(ns[0]))

        one = analyse(load_runs([str(d1)]), 3.0, 0.05, 0.80)
        ck("单 run 时噪声底噪标为不可估", one.get("noise") is None, str(one.get("noise")))
        ck("单 run 时功效来源标为组内 SD", "组内" in one["power"]["sigma_source"],
           one["power"]["sigma_source"])

        md = render(out, [str(d1), str(d2)], 3.0)
        ck("渲染出非空 markdown", isinstance(md, str) and len(md) > 100, str(len(md)))
        ck("渲染报出 run 数", "run 数" in md)

    for n, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {n}"
              + (f"   <- {detail}" if detail and not ok else ""))
    np_ = sum(1 for _, o, _ in checks if o)
    print(f"\n{np_} PASS   {len(checks) - np_} FAIL")
    return 0 if np_ == len(checks) else 1


# ---------------------------------------------------------------- 入口
def main():
    # 🔴 中文 Windows 控制台默认 GBK(cp936)：输出里的 emoji 编不出来 ⇒
    #    UnicodeEncodeError **直接崩**（崩在打印那一刻，不是逻辑错）。
    #    兜底：只改 errors，不改 encoding —— 改 encoding="utf-8" 会让中文变乱码。
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(
        description="测量 GEO harness 的噪声底噪 + 反推所需样本量",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__[__doc__.index("用法：") :],
    )
    ap.add_argument("dirs", nargs="*", help="一个或多个 run 目录（或 raw_trials.csv）")
    ap.add_argument("--glob", default="", help="批量匹配 run 目录，例如 data/run*")
    ap.add_argument("--target-delta", type=float, default=3.0,
                    help="想检测出的效应量（geo_impression 绝对百分点），用于反推样本量，默认 3.0")
    ap.add_argument("--alpha", type=float, default=0.05, choices=[0.10, 0.05, 0.02, 0.01])
    ap.add_argument("--power", type=float, default=0.80, choices=[0.80, 0.90, 0.95])
    ap.add_argument("--out", default="", help="报告输出路径，默认写到**当前目录**"
                                          "（不写进输入数据目录，避免污染语料）")
    ap.add_argument("--selftest", action="store_true", help="只跑自检（注入已知真值的合成数据）")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    paths = list(args.dirs)
    if args.glob:
        paths += sorted(str(p) for p in Path(args.glob).parent.glob(Path(args.glob).name))
    paths = [p for p in paths if Path(p).exists()]
    if len(paths) < 1:
        ap.error("至少给一个存在的 run 目录，或用 --glob")

    runs = load_runs(paths)
    if not runs:
        sys.exit("[!] 没读到任何含 geo_impression 列的 raw_trials.csv")

    print(f"[+] 读到 {len(runs)} 个 run / {sum(len(r['rows']) for r in runs)} 条试验")
    out = analyse(runs, args.target_delta, args.alpha, args.power)

    md = render(out, paths, args.target_delta)
    outp = Path(args.out) if args.out else Path.cwd() / "noise_report.md"
    outp.write_text(md, encoding="utf-8")
    (outp.with_suffix(".json")).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(md)
    print(f"\n报告已写入：{outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
