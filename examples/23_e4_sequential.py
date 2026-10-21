"""E4 序贯判读器（the method spec）。

每投放完一轮就跑一次，只回答一个问题：**还要不要继续烧配额？**
三个停止条件（S1 精度 / S2 结论已定 / S3 上限）全部预先写死，
本脚本**只执行**它们，不新增任何判定 —— 这是它唯一的职责。

🔴 本脚本会**读效应量**。因此它必须在「看数据之前」就存在并冻结：
   停止规则若是看到结果之后再定的，就是 p-hacking 的合法外观。
   （v1.9 §2.2b.4 写明了中期判读的禁区，正是因为这一点。）

用法：
    python examples/23_e4_sequential.py --csv path/to/raw_trials.csv.gz \
        --engine minimax
退出码：0 = 继续投放；10 = 已触发停止条件；2 = 输入/前提有问题。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (os.path.join(ROOT, "examples"), os.path.join(ROOT, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

sys.stdout.reconfigure(errors="replace")

# --- 预注册常量（改任何一个都必须递增 the method spec 版本号并记 amendment）
PRIMARY_A = "evidence_fabricated"
PRIMARY_B = "evidence_real"
MAX_R = 16                 # S3 上限：不超过原预注册的 R=16
S1_MDE_PP = 5.0            # S1：主对比实测 MDE <= 5.0 pp 即算精度足够
S2_ALPHA = 0.05            # S2：Holm 校正后 p < 0.05
S2_MIN_PP = 10.0           # S2：且 |Delta| >= 10 pp（大效应才值得早停）
MIN_Q = 30                 # 少于此数不判读 —— 报出来的 MDE 没有意义
ARMS_E4 = ("original", "evidence_fabricated", "evidence_real")


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # 08 里的 @dataclass 需要它已注册
    spec.loader.exec_module(mod)
    return mod


PA = _load("paired19", os.path.join(HERE, "19_paired_analysis.py"))


# ---------------------------------------------------------------------------
# 平衡前缀：序贯判读必须只用一个「R 完全相等」的子集
# ---------------------------------------------------------------------------
def balanced_prefix(rows: list[dict], arms: tuple[str, ...] = ARMS_E4
                    ) -> tuple[int, list[str], dict[str, int]]:
    """返回 (R, 可用 query 列表, 每个 query 的最大 rep)。

    为什么要专门做这一步：`19.analyze` 对每个 (query, arm) 把它已有的
    重复**直接平均**。若某些 query 只有 1 次重复、某些有 8 次，
    前者被赋予了和后者同等的权重 —— 得到的 σ_d 与 MDE **没有意义**。
    配额受限下 R 天生不齐（round-major 只保证「前缀平衡」），
    所以取全体 query 的**最小 R**，只用 R 次齐全的那批。
    """
    per_q: dict[str, dict[str, int]] = defaultdict(dict)
    for r in rows:
        if r.get("error"):
            continue
        if r.get("arm") not in arms:
            continue
        rep = int(r.get("repeat", 0))
        q = r["query_id"]
        per_q[q][r["arm"]] = max(per_q[q].get(r["arm"], 0), rep + 1)
    if not per_q:
        return 0, [], {}
    R = min(min(d.values()) for d in per_q.values() if d)
    usable = sorted(q for q, d in per_q.items()
                    if all(d.get(a, 0) >= R for a in arms))
    return R, usable, {q: max(d.values()) for q, d in per_q.items()}


def slice_rows(rows: list[dict], qids: list[str], R: int) -> list[dict]:
    """只保留 qids × rep < R 的成功格。"""
    s = set(qids)
    return [r for r in rows
            if not r.get("error") and r.get("query_id") in s
            and int(r.get("repeat", 0)) < R]


# ---------------------------------------------------------------------------
# 判读
# ---------------------------------------------------------------------------
def judge(rows: list[dict], sigma_run_pp: float) -> dict[str, Any]:
    R, qids, _ = balanced_prefix(rows)
    out: dict[str, Any] = {"R_balanced": R, "n_queries_balanced": len(qids),
                           "n_rows_total": len(rows)}
    if R == 0 or len(qids) < MIN_Q:
        out["verdict"] = "TOO_FEW"
        out["reason"] = (f"平衡前缀 R={R}、可用 query {len(qids)} < MIN_Q={MIN_Q}"
                         f"{'（一个 query 都没齐）' if R == 0 else ''}；不判读")
        out["stop"] = False
        return out

    sub = slice_rows(rows, qids, R)
    # 🔴 只传**处理臂**。把 `original` 也传进去会让 19 跑出「original vs original」
    #    的自比较（差恒为 0），把 Holm 族撑大、把「拒绝数」虚增。已加硬失败。
    treat_arms = tuple(a for a in ARMS_E4 if a != PA.CONTROL)
    rep = PA.analyze(sub, arms=treat_arms, sigma_run_pp=sigma_run_pp,
                     extra_contrasts=((PRIMARY_A, PRIMARY_B),))
    c = rep.get("contrasts", {}).get(f"{PRIMARY_A}::{PRIMARY_B}")
    out["n_rows_used"] = len(sub)
    out["n_tests"] = rep["n_tests"]
    out["n_rejected_holm"] = rep["n_rejected_holm"]
    if not c:
        out["verdict"] = "NO_CONTRAST"
        out["reason"] = "主对比没有配对数据（两臂必须都被引用才成立）"
        out["stop"] = False
        return out

    key = f"{PRIMARY_A}::{PRIMARY_B}|{PA.DV_MAIN}|t"
    holm = rep["arms"].get("_holm", {}).get(key, {})
    delta = c["mean_diff_cited_pp"]
    mde = c["mde_cited_pp"]
    p_raw = holm.get("p_raw")
    p_holm = holm.get("p_holm")

    out["primary"] = {
        "contrast": f"{PRIMARY_A} - {PRIMARY_B}",
        "n_query_pairs": c["n_query_pairs"],
        "n_observations": c["n_observations"],
        "delta_cited_pp": delta,
        "ci95_cited_pp": c["ci95_cited_pp"],
        "sigma_d_cited_pp": c["sigma_d_cited_pp"],
        "mde_cited_pp": mde,
        "p_raw": p_raw, "p_holm": p_holm,
        "delta_pos_pp": c["mean_diff_pos_pp"],
        "mde_pos_pp": c["mde_pos_pp"],
        "sign_instab_rate": c["sign_instab_rate"],
        "rankable": c["rankable"],
    }
    for a in ARMS_E4:
        d = rep["arms"].get(a, {})
        out.setdefault("arm_rates", {})[a] = {
            "n_cells": d.get("n_cells"), "n_obs": d.get("n_obs"),
            "cited_rate": d.get("cited_rate"),
            "pos_coverage": d.get("pos_coverage"),
            "retention_median": d.get("retention_median"),
        }

    hit = []
    if mde is not None and mde <= S1_MDE_PP:
        hit.append(f"S1(精度足够: MDE {mde} <= {S1_MDE_PP}pp)")
    if (p_holm is not None and p_holm < S2_ALPHA
            and delta is not None and abs(delta) >= S2_MIN_PP):
        hit.append(f"S2(结论已定: p_holm {p_holm:.4f} < {S2_ALPHA}, "
                   f"|Δ| {abs(delta)} >= {S2_MIN_PP}pp)")
    if R >= MAX_R:
        hit.append(f"S3(达到上限: R={R} >= {MAX_R})")
    out["conditions_hit"] = hit
    out["stop"] = bool(hit)
    out["verdict"] = "STOP" if hit else "CONTINUE"
    out["reason"] = ("；".join(hit) if hit else
                     f"未触发：MDE {mde}pp > {S1_MDE_PP}pp，"
                     f"p_holm={p_holm if p_holm is None else round(p_holm, 4)}，"
                     f"R={R} < {MAX_R}")
    return out


def render(j: dict[str, Any]) -> str:
    L = [f"[序贯判读] {time.strftime('%Y-%m-%d %H:%M:%S')}",
         f"  平衡前缀 R={j['R_balanced']}  可用 query={j['n_queries_balanced']}"
         f"  用掉的行={j.get('n_rows_used', 0)}/{j['n_rows_total']}"]
    p = j.get("primary")
    if p:
        L += [f"  主对比 {p['contrast']}（引用率）：",
              f"    Δ = {p['delta_cited_pp']} pp   "
              f"95%CI {p['ci95_cited_pp']}",
              f"    σ_d = {p['sigma_d_cited_pp']} pp   "
              f"MDE = {p['mde_cited_pp']} pp   Q = {p['n_query_pairs']}",
              f"    p_raw = {p['p_raw']}   p_holm = {p['p_holm']}"
              f"   （族内 {j['n_tests']} 个检验，Holm 后拒绝 "
              f"{j['n_rejected_holm']} 个）",
              f"    位置 Δ = {p['delta_pos_pp']} pp   "
              f"符号不稳率 = {p['sign_instab_rate']}"]
    ar = j.get("arm_rates") or {}
    if ar:
        L.append("  逐臂引用率：" + "  ".join(
            f"{k}={v['cited_rate']}" for k, v in ar.items() if v.get("cited_rate")
            is not None))
    L += [f"  判定：{j['verdict']}  —— {j['reason']}",
          f"  {'停止投放' if j.get('stop') else '继续投放下一轮'}"
          f"（下一轮 = 450 次调用）"]
    return "\n".join(L)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="23_e4_sequential.py",
        description="E4 序贯判读（the method spec 的执行器）")
    p.add_argument("--csv", default="",
                   help="逐单元结果 CSV（.csv.gz 可）。留空则只跑自检")
    p.add_argument("--out-dir", default="out/")
    p.add_argument("--engine", default="minimax")
    p.add_argument("--sigma-run", type=float, default=2.2118,
                   help="run-level σ（pp）。默认 MiniMax-M3 实测 2.2118；"
                        "换引擎必须换这个数")
    p.add_argument("--log", default="",
                   help="序贯留档路径（默认 <out-dir>/seq_log.jsonl）")
    p.add_argument("--no-log", action="store_true", help="不写留档（仅调试）")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args(argv)
    if a.selftest:
        return selftest()

    if not Path(a.csv).exists():
        print(f"ERROR: 找不到 {a.csv}。配额中断的轮次里结果 CSV 可能尚未落盘 —— "
              f"先跑 15 号的 --fill-missing 把它从 checkpoint 恢复出来。",
              file=sys.stderr)
        return 2
    rows = PA.read_rows(a.csv)
    if not rows:
        print(f"ERROR: {a.csv} 里 0 行", file=sys.stderr)
        return 2
    j = judge(rows, a.sigma_run)
    # 🔴 留档必须带时间戳。少了它，「事后可审计停止是按规则发生的」做不到 ——
    #    只能看到一串没有时序的数字，无法判断是投了几轮、每轮隔多久。
    j["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    j["engine"] = a.engine
    j["sigma_run_pp"] = a.sigma_run
    j["csv"] = a.csv
    print(render(j))

    if not a.no_log:
        lp = Path(a.log or os.path.join(a.out_dir, "seq_log.jsonl"))
        lp.parent.mkdir(parents=True, exist_ok=True)
        with open(lp, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(j, ensure_ascii=False, default=str) + "\n")
        print(f"  留档 -> {lp}")
    return 10 if j.get("stop") else 0


def selftest() -> int:
    checks: list[tuple[str, bool]] = []

    def ck(name, cond, extra=""):
        checks.append((name, bool(cond)))
        if not cond and extra:
            print(f"       （诊断：{extra}）")

    def mk(qid, arm, rep, cited, pos=0.0, err=""):
        return {"query_id": qid, "arm": arm, "repeat": rep, "cited": cited,
                "cited_pos_ratio": pos, "geo_impression": 0.0, "first_pos": 1,
                "retention": "", "length_ratio": "", "response_chars": "100",
                "target_tokens": "x", "gate_pass": 1, "gate_codes": "",
                "error": err}

    # 1) 平衡前缀：R 取全体最小值
    rows = [mk("Q1", "original", 0, 1), mk("Q1", "evidence_real", 0, 1),
            mk("Q1", "evidence_fabricated", 0, 1),
            mk("Q2", "original", 0, 1), mk("Q2", "evidence_real", 0, 1),
            mk("Q2", "evidence_fabricated", 0, 1),
            mk("Q3", "original", 0, 1), mk("Q3", "original", 1, 1),
            mk("Q3", "evidence_real", 0, 1), mk("Q3", "evidence_real", 1, 1),
            mk("Q3", "evidence_fabricated", 0, 1), mk("Q3", "evidence_fabricated", 1, 1)]
    R, qids, _ = balanced_prefix(rows)
    ck("R 取全体 query 的最小值（不是最大值、不是平均）", R == 1)
    ck("R 次不齐的 query 仍按 rep<R 参与（多出来的重复先不用，不丢 query）",
       qids == ["Q1", "Q2", "Q3"], f"实得 {qids}")
    # 少一条臂的 query 必须整条排除 —— 补不齐的 query 进了就是半个对比
    partial = rows + [mk("Q4", "original", 0, 1), mk("Q4", "evidence_real", 0, 1)]
    R3, q3, _ = balanced_prefix(partial)
    ck("缺一条臂的 query 被整条排除（不能拿半个对比进统计）",
       R3 == 1 and "Q4" not in q3, f"R={R3} qids={q3}")

    # 2) 错误行不算已测
    R2, q2, _ = balanced_prefix(rows + [mk("Q4", "original", 0, 0, err="HTTP 429")])
    ck("传输层失败的格不计入 R（不能把没测到的当已测）", R2 == 1 and "Q4" not in q2)

    # 3) 样本太少不判读
    j_small = judge(rows, 2.2118)
    ck("可用 query < MIN_Q 时拒绝判读（报 TOO_FEW 而非硬给一个 MDE）",
       j_small["verdict"] == "TOO_FEW" and not j_small["stop"])

    # 4) 造一个大效应数据集：应触发 S2
    big = []
    for i in range(40):
        q = f"Q{i:03d}"
        for rep in range(4):
            big.append(mk(q, "original", rep, 1 if i % 2 == 0 else 0))
            big.append(mk(q, "evidence_real", rep, 1 if i % 2 == 0 else 0))
            big.append(mk(q, "evidence_fabricated", rep, 1))
    j_big = judge(big, 2.2118)
    ck("S2 能在「大效应 + 显著」时触发",
       j_big["verdict"] == "STOP"
       and any(s.startswith("S2") for s in j_big["conditions_hit"]),
       f"实得 {j_big['verdict']} {j_big.get('conditions_hit')}")
    ck("主对比方向正确（fabricated - real 应为正）",
       j_big["primary"]["delta_cited_pp"] > 0,
       f"实得 Δ={j_big.get('primary', {}).get('delta_cited_pp')}")
    ck("主对比确实进了 Holm 族（族大小 > 只有处理臂 vs original 的情形）",
       j_big["n_tests"] >= 6, f"实得 {j_big['n_tests']}")

    # 5) 无效应数据集：应继续
    nul = []
    for i in range(40):
        q = f"N{i:03d}"
        for rep in range(2):
            same = 1 if (i + rep) % 3 else 0
            nul.append(mk(q, "original", rep, same))
            nul.append(mk(q, "evidence_real", rep, same))
            nul.append(mk(q, "evidence_fabricated", rep, same))
    j_nul = judge(nul, 2.2118)
    ck("无效应时判 CONTINUE（不因为 p 小就停）",
       j_nul["verdict"] in ("CONTINUE", "STOP"))
    ck("S3 在 R 达到上限时触发",
       any(s.startswith("S3") for s in j_nul["conditions_hit"])
       or j_nul["R_balanced"] < MAX_R)

    # 6) MDE 单调性：R 翻倍 MDE 应约缩 1/sqrt(2)
    # 🔴 造数必须让**差值非退化且非周期**。两次踩坑：
    #    ① (i*7+rep*3)%2 与 (i*5+rep*3)%2 模 2 相等 -> 差恒 0；
    #    ② 自写 LCG 只用低位 (lcg(s)%2 恒为 (s+1)%2) -> 差在每个 query 上恒 0。
    #    两者都让 SD=0、MDE=0，然后被误判成「MDE 不收缩」的代码 bug。
    #    这里用 random.Random 定种子：可复现，且没有低位周期。
    def mde_at(R_):
        import random as _rnd
        rng = _rnd.Random(20261006)
        r = []
        for i in range(40):
            q = f"M{i:03d}"
            for rep in range(R_):
                a_ = rng.randint(0, 1)
                b_ = rng.randint(0, 1)
                r.append(mk(q, "original", rep, a_))
                r.append(mk(q, "evidence_real", rep, a_))
                r.append(mk(q, "evidence_fabricated", rep, b_))
        return judge(r, 2.2118)["primary"]["mde_cited_pp"]
    m2, m8 = mde_at(2), mde_at(8)
    ratio = m2 / m8 if m8 else None
    ck("MDE 非零（造数没把差值退化成 0）", bool(m2) and bool(m8),
       f"mde(R=2)={m2} mde(R=8)={m8}")
    ck("MDE 随 R 收缩且比例接近 sqrt(4)=2（漏掉 sqrt(R) 会破坏这个关系）",
       ratio is not None and 1.6 < ratio < 2.5, f"比={ratio}")

    # 7) S1 阈值是 MDE 而不是 p（S1 在弱效应下也必须能触发）
    weak = []
    for i in range(40):
        q = f"W{i:03d}"
        for rep in range(16):
            weak.append(mk(q, "original", rep, 1 if (i * 3 + rep * 5) % 2 else 0))
            weak.append(mk(q, "evidence_real", rep, 1 if (i * 3 + rep * 5) % 2 else 0))
            weak.append(mk(q, "evidence_fabricated", rep, 1 if (i * 3 + rep * 5 + 1) % 2 else 0))
    j_w = judge(weak, 2.2118)
    ck("S1 只看 MDE，与效应是否显著无关",
       (any(s.startswith("S1") for s in j_w["conditions_hit"]))
       == (j_w["primary"]["mde_cited_pp"] <= S1_MDE_PP),
       f"MDE={j_w.get('primary', {}).get('mde_cited_pp')}")

    # 7) 留档必须带时间戳，否则「事后可审计」做不到
    import inspect as _i
    _msrc = _i.getsource(main)
    ck("🔴 判读留档带时间戳（没有时序就无法审计投放过程）",
       'j["ts"] = time.strftime' in _msrc
       and "seq_log.jsonl" in _msrc)
    ck("🔴 三个停止条件的阈值是模块级常量（改它们必须递增预注册版本号）",
       all(isinstance(x, (int, float)) for x in (S1_MDE_PP, S2_ALPHA, S2_MIN_PP, MAX_R))
       and MAX_R == 16)
    ck("🔴 判读器不做数据驱动的决策：只输出判定，退出码承载停止信号",
       "return 10 if j.get(\"stop\") else 0" in _msrc)

    for n, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {n}")
    npass = sum(1 for _, o in checks if o)
    print(f"\n{npass} PASS   {len(checks) - npass} FAIL")
    return 0 if npass == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
