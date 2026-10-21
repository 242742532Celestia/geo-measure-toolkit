"""E4 主对比的方差成分分解：回答一个决定预算的问题 ——
**「再加重复 R，MDE 还能降吗？降到多少为止？」**

这是纯噪声结构问题，**与效应方向、显著性无关**，
所以它可以在不破坏预注册完整性的前提下被单独回答。

模型（query 级随机效应 + 重复级残差）：

    d_i = μ + u_i + e_i/R̄ ,   u_i ~ N(0, σ_b²),  e_i ~ N(0, σ_e²)

`d_i` 是 query i 在 R 次重复上取平均后的配对差，其方差为

    Var(d_i) = σ_b² + σ_e² / R

于是 **σ_d(R)² = B + E/R**，B = σ_b²（query 间的系统异质性），
E = σ_e²（单次观测噪声）。用两个不同 R 的实测 σ_d² 就能解出 B 与 E。

🔴 **关键推论**：σ_d(R) → σ_b ≠ 0。所以 MDE = (z_{α/2}+z_β)·σ_d/√Q
   **有下界** `(z_{α/2}+z_β)·σ_b/√Q`，**与 R 无关**。
   若这个下界已经 > 停止阈值，那么「再多投几轮」在算术上不可能达标 ——
   继续烧配额买不到任何精度。这一点必须**用数据说**，不能凭感觉。

用法：python examples/24_e4_noise_floor.py [--csv ...] [--selftest]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (os.path.join(ROOT, "examples"), os.path.join(ROOT, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.stdout.reconfigure(errors="replace")

Z_SUM = 1.959964 + 0.841621
PRIMARY_A = "evidence_fabricated"
PRIMARY_B = "evidence_real"
ARMS_E4 = ("original", "evidence_fabricated", "evidence_real")
S1_MDE_PP = 5.0          # 与 the method spec 一致
MAX_R = 16


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


SEQ = _load("seq23", os.path.join(HERE, "23_e4_sequential.py"))


def fit_B_E(sd_at_R: dict[int, float]) -> tuple[float, float, float]:
    """由两个及以上 (R, σ_d) 反解 B = σ_b² 与 E = σ_e²。

    返回 (B, E, 拟合残差 RMS)。只有一个 R 时无法分解，返回 NaN。
    """
    pts = sorted((R, s * s) for R, s in sd_at_R.items() if s and s > 0)
    if len(pts) < 2:
        return float("nan"), float("nan"), float("nan")
    # 最小二乘：y = B + E·x，x = 1/R
    xs = [1.0 / R for R, _ in pts]
    ys = [v for _, v in pts]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return float("nan"), float("nan"), float("nan")
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    inter = my - slope * mx
    rms = math.sqrt(sum((y - (inter + slope * x)) ** 2
                        for x, y in zip(xs, ys)) / n)
    return inter, slope, rms


def mde_floor_pp(Q: int, sigma_b: float) -> float:
    """R → ∞ 时的 MDE 下界（pp）。"""
    return Z_SUM * sigma_b / math.sqrt(Q) * 100


def need_Q_for(mde_target_pp: float, sigma_b: float) -> float:
    """要达到目标 MDE，需要多少个独立 query。"""
    return (Z_SUM * sigma_b * 100 / mde_target_pp) ** 2


def analyse(rows: list[dict], sigma_run_pp: float = 2.2118) -> dict:
    out: dict = {"points": []}
    R, qids, _ = SEQ.balanced_prefix(rows)
    out["R_now"] = R
    out["Q"] = len(qids)
    if R < 1 or len(qids) < SEQ.MIN_Q:
        out["verdict"] = "TOO_FEW"
        return out

    treat = tuple(a for a in ARMS_E4 if a != "original")
    sd_at_R: dict[int, float] = {}
    for r in range(1, R + 1):
        sub = SEQ.slice_rows(rows, qids, r)
        rep = SEQ.PA.analyze(sub, arms=treat, sigma_run_pp=sigma_run_pp,
                            extra_contrasts=((PRIMARY_A, PRIMARY_B),))
        c = (rep.get("contrasts") or {}).get(f"{PRIMARY_A}::{PRIMARY_B}")
        if not c:
            continue
        sd = c.get("sigma_d_cited_pp")
        if sd:
            sd_at_R[r] = sd / 100.0
        out["points"].append({
            "R": r, "Q": len(qids), "sigma_d_pp": c.get("sigma_d_cited_pp"),
            "mde_pp": c.get("mde_cited_pp"),
            "delta_pp": c.get("mean_diff_cited_pp"),
        })
    out["sigma_d_by_R"] = sd_at_R
    B, E, rms = fit_B_E(sd_at_R)
    out["B_sigma_b2"] = B
    out["E_sigma_e2"] = E
    out["fit_rms"] = rms
    if B == B:  # not NaN
        out["sigma_b_pp"] = math.sqrt(B) * 100
        out["sigma_e_pp"] = math.sqrt(E) * 100 if E > 0 else None
        out["mde_floor_pp"] = mde_floor_pp(len(qids), math.sqrt(B))
        out["mde_floor_R16"] = None
        # R=16 时的实际 MDE（用 B + E/16）
        v16 = B + E / MAX_R if E == E else float("nan")
        if v16 == v16 and v16 > 0:
            out["mde_at_R16_pp"] = Z_SUM * math.sqrt(v16) / math.sqrt(len(qids)) * 100
        out["Q_needed_for_S1"] = need_Q_for(S1_MDE_PP, math.sqrt(B))
        out["S1_reachable_at_current_Q"] = out["mde_floor_pp"] <= S1_MDE_PP
        out["S1_reachable_at_R16"] = bool(
            out.get("mde_at_R16_pp") is not None
            and out["mde_at_R16_pp"] <= S1_MDE_PP)
    out["verdict"] = "OK" if B == B else "NEED_MORE_R"
    return out


def render(a: dict) -> str:
    L = ["", "=" * 68, "E4 主对比的噪声结构：再加 R 还能买到多少精度？", "=" * 68,
         f"当前平衡前缀 R={a['R_now']}  Q={a['Q']}"]
    if a.get("verdict") != "OK":
        L.append(f"  还不能分解（{a.get('verdict')}）—— 至少需要 2 个不同 R 的 σ_d。")
        return "\n".join(L)
    L += ["", "| R | σ_d (pp) | MDE (pp) | Δ (pp) |", "|---|---|---|---|"]
    for p in a["points"]:
        L.append(f"| {p['R']} | {p['sigma_d_pp']} | {p['mde_pp']} | {p['delta_pp']} |")
    L += ["", f"拟合 σ_d(R)² = B + E/R   （残差 RMS = {a['fit_rms']:.1f}）",
          f"  σ_b（query 间系统异质性）= **{a['sigma_b_pp']:.2f} pp**",
          f"  σ_e（单次观测噪声）      = "
          f"{('%.2f pp' % a['sigma_e_pp']) if a.get('sigma_e_pp') else '—'}",
          "",
          f"  → MDE 下界（R→∞，Q={a['Q']} 不变）= **{a['mde_floor_pp']:.2f} pp**",
          f"  → 投到 R={MAX_R} 的 MDE            = "
          f"{('%.2f pp' % a['mde_at_R16_pp']) if a.get('mde_at_R16_pp') else '—'}",
          f"  → 要让 MDE ≤ {S1_MDE_PP} pp，需要 Q ≥ "
          f"**{a['Q_needed_for_S1']:.0f}** 个独立 query（当前 {a['Q']}）",
          ""]
    if not a["S1_reachable_at_current_Q"]:
        L += [f"  ■ **S1（MDE ≤ {S1_MDE_PP} pp）在 Q={a['Q']} 下算术上不可达。**",
              f"    下界 {a['mde_floor_pp']:.2f} pp > {S1_MDE_PP} pp，"
              f"**与 R 无关** —— 继续加重复也过不了线。",
              f"    买到的是 {a['R_now']} → {MAX_R} 轮之间 MDE 从 "
              f"{a['points'][-1]['mde_pp']} 降到 {a['mde_at_R16_pp']:.2f} pp"
              f"（约 {(a['points'][-1]['mde_pp'] - a['mde_at_R16_pp']) / a['points'][-1]['mde_pp'] * 100:.0f}%），",
              f"    代价 {MAX_R - a['R_now']} 轮 × 450 = "
              f"{(MAX_R - a['R_now']) * 450:,} 次调用。"]
    return "\n".join(L)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=os.path.join(ROOT, "outputs", "15_e4",
                                                 "raw_trials.csv.gz"))
    p.add_argument("--json", default=os.path.join(ROOT, "outputs",
                                                  "e4_noise_floor.json"))
    p.add_argument("--sigma-run", type=float, default=2.2118)
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args(argv)
    if a.selftest:
        return selftest()
    if not os.path.exists(a.csv):
        print(f"ERROR: 找不到 {a.csv}", file=sys.stderr)
        return 2
    import csv as _csv
    import gzip as _gz
    op = (_gz.open(a.csv, "rt", encoding="utf-8", newline="")
          if a.csv.endswith(".gz") else open(a.csv, encoding="utf-8", newline=""))
    with op as fh:
        rows = list(_csv.DictReader(fh))
    res = analyse(rows, a.sigma_run)
    print(render(res))
    with open(a.json, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2, default=str)
        fh.write("\n")
    print(f"\n-> {a.json}")
    return 0


def selftest() -> int:
    checks: list[tuple[str, bool]] = []

    def ck(n, c, e=""):
        checks.append((n, bool(c)))
        if not c and e:
            print(f"       （诊断：{e}）")

    # 1) 分解公式：已知 B、E 时必须复原
    B0, E0, Q = 1438.0, 2068.0, 150
    sd = {R: math.sqrt(B0 + E0 / R) for R in (1, 2, 3, 4)}
    B, E, rms = fit_B_E(sd)
    ck("由 σ_d(R)²=B+E/R 反解出 B", abs(B - B0) / B0 < 1e-9, f"B={B}")
    ck("由 σ_d(R)²=B+E/R 反解出 E", abs(E - E0) / E0 < 1e-9, f"E={E}")
    ck("拟合残差为 0（构造无噪声）", rms < 1e-6, f"rms={rms}")

    # 2) MDE 下界与 R 无关，且是真实下界
    sb = math.sqrt(B0)
    floor = mde_floor_pp(Q, sb)
    ck("MDE 下界 = (z_a+z_b)·σ_b/√Q",
       abs(floor - Z_SUM * sb / math.sqrt(Q) * 100) < 1e-9)
    ck("任何有限 R 的 MDE 都 > 下界", all(
        Z_SUM * math.sqrt(B0 + E0 / R) / math.sqrt(Q) * 100 > floor
        for R in (1, 2, 4, 8, 16, 64, 1000)))
    ck("R 再大也逼近而不超过下界（加重复买不到下界以下的精度）",
       Z_SUM * math.sqrt(B0 + E0 / 10 ** 9) / math.sqrt(Q) * 100 - floor < 0.01)

    # 3) 要达标需要多少 Q
    need = need_Q_for(S1_MDE_PP, sb)
    ck("反解 Q 与正算一致（round-trip）",
       abs(mde_floor_pp(int(round(need)), sb) - S1_MDE_PP) < 0.05,
       f"need={need:.1f} mde={mde_floor_pp(int(round(need)), sb):.3f}")

    # 4) 只有 σ_b=0（纯重复噪声）时 MDE 才随 R 趋于 0
    B0z, E0z = 0.0, 1000.0
    sd0 = {R: math.sqrt(B0z + E0z / R) for R in (1, 2, 4, 8)}
    Bz, Ez, _ = fit_B_E(sd0)
    ck("纯重复噪声（B=0）时 MDE 下界为 0 —— 那是加重复真的有用的情形",
       abs(mde_floor_pp(Q, math.sqrt(Bz))) < 1e-9, f"floor={mde_floor_pp(Q, math.sqrt(Bz))}")

    # 5) 只有一个 R 时必须明说「还不能分解」，不能硬给一个数
    ck("单个 R 无法分解（返回 NaN 而不是编一个）",
       fit_B_E({3: 0.46})[0] != fit_B_E({3: 0.46})[0])

    # 6) 端到端：造 σ_b 占优的数据，验证 S1 被判为不可达
    def mk(qid, arm, rep, cited, pos=0.3, err=""):
        return {"query_id": qid, "arm": arm, "repeat": rep, "cited": cited,
                "cited_pos_ratio": pos, "geo_impression": 0.0, "first_pos": 1,
                "retention": "", "length_ratio": "", "response_chars": "10",
                "target_tokens": "x", "gate_pass": 1, "gate_codes": "", "error": err}
    import random as _rnd
    rng = _rnd.Random(7)
    rows = []
    for i in range(60):
        q = f"F{i:03d}"
        # σ_b 主导：每条 query 有一个固定的偏移
        off = -1 if i % 2 == 0 else 0
        for rep in range(4):
            a_ = 1 if (i + rep) % 2 == 0 else 0
            b_ = 1 if a_ != off else 0
            rows.append(mk(q, "original", rep, a_))
            rows.append(mk(q, "evidence_real", rep, a_))
            rows.append(mk(q, "evidence_fabricated", rep, b_))
    res = analyse(rows, 2.2118)
    ck("端到端：能分解出 σ_b", res.get("verdict") == "OK"
       and res.get("sigma_b_pp", 0) > 0, str(res.get("verdict")))
    ck("端到端：Q=60 时若下界 > 5pp 则必须判 S1 不可达",
       (res["mde_floor_pp"] > S1_MDE_PP) == (not res["S1_reachable_at_current_Q"]),
       f"floor={res.get('mde_floor_pp')}")
    ck("端到端：反解出的「达标所需 Q」大于当前 Q（既然不可达就必须更大）",
       res["Q_needed_for_S1"] > res["Q"],
       f"need={res.get('Q_needed_for_S1'):.0f} Q={res['Q']}")
    ck("渲染里出现「算术上不可达」的字样", "算术上不可达" in render(res))

    for n, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {n}")
    np_ = sum(1 for _, o in checks if o)
    print(f"\n{np_} PASS   {len(checks) - np_} FAIL")
    return 0 if np_ == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
