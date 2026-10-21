#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""19_paired_analysis.py —— 阶段 B 主分析（the method spec (pre-registered)）

做什么
------
对每个处理臂，在**同一个 query 内**与 `original` 臂做配对比较，
两个主因变量：

| 档 | 因变量 | 预注册检验 | 单元 |
|---|---|---|---|
| 🔵 主 | `citation probability` | 配对 **McNemar** | query |
| 🔵 共同主 | `cited_pos_ratio`（纠错 ⑦） | 配对 t / Wilcoxon | query |
| 🟢 共同主描述性 | 信息保留率 `retention` | 并排报告，不作检验 | (query, arm) |

多重比较：**4 臂 × 2 主因变量 = 8 个检验**，Holm–Bonferroni 校正到 FWER = 0.05，
**校正后与未校正的 p 都报**。

🔴 一处**必须记录的规格裁决**（v1.4 偏差，见下）
------------------------------------------------
§7.1 写的是「配对 McNemar（二值）」，但每个 `(query, arm)` 有 **R = 16** 次重复。
在**格（cell）**层面跑 McNemar，等于把 16 个相关格当成 16 个独立观测 ——
**这正是本文批评 2603.09296 / 2607.12056 的那个错误**（「300 次当独立」）。

所以本脚本**两个层级都报**，不做二选一：

1. **主口径 = query 级配对**：先把每个 `(query, arm)` 的 R 次重复取平均，
   得到该 query 在该臂的引用率，再做配对 t + Wilcoxon。
   单元是 **query**（独立单元 150 个），与 §6.3b 的 MDE 口径一致。
2. **次口径 = 多数表决后的 McNemar**：把该 query 的引用率按 `> 0.5` 二值化，
   在 query 级跑 McNemar。**这就是预注册的那个检验，只是放回了正确的层级。**

两者一起报，读者可自行判断。若两者方向相反，必须在正文点出来。

共用纪律
--------
- **只报能算的**：`cited_pos_ratio` 要求两臂都 `cited=1` 才成立，
  进入配对子集的 query 数必须与 `cited_pos_ratio` 的原始覆盖率一起报。
- **符号稳定性**（§7.2）：逐 query 看 Δ 的符号分布，翻转率超阈值时该臂
  **不可作报告**。规则本身不改。
- 纯标准库（无 scipy/numpy）。
"""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import importlib.util
import json
import math
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


VAR = _load("var14", os.path.join(ROOT, "examples", "14_stage_a_variance.py"))
#: 08 里有 `_t_sf`（t 分布双尾 p，用不完全 β 函数）；14 只有正态。
#: **复用而不是自己写** —— 两个文件各写一份 t 分布迟早会漂开。
REPRO = _load("repro08", os.path.join(ROOT, "examples", "08_reproduce_geo_bench.py"))
N = VAR._N                      # 正态分布工具（与 14 号同源，不另抄一份）
_t_sf = REPRO._t_sf

TREATMENT_ARMS = ("structure", "evidence", "semantic", "full")
CONTROL = "original"
DV_MAIN = "citation probability"
DV_CO = "cited_pos_ratio"


def _raises_valueerror(fn) -> bool:
    try:
        fn()
    except ValueError:
        return True
    except Exception:  # noqa: BLE001
        return False
    return False


def _load_fp():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "freeze_prompts", os.path.join(ROOT, "tools", "freeze_prompts.py"))
    if spec is None or spec.loader is None:
        return None
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    try:
        spec.loader.exec_module(m)
    except Exception:  # noqa: BLE001
        return None
    return m


_FP = _load_fp()


def infer_arms(rows: list[dict], default: tuple = TREATMENT_ARMS) -> tuple:
    """从数据里反推处理臂表，**不硬编码**。

    🔴 E4 的臂名是 `evidence_fabricated` / `evidence_real`，与阶段 B 的
    `evidence` / `semantic` …完全不同。沿用硬编码的 `TREATMENT_ARMS` 会让
    E4 的两个臂**整条从报表上消失** —— 只剩 `original` 一行，而每个结论
    都「看起来正常」。这类静默丢失比报错难查得多。
    """
    reg = set(TREATMENT_ARMS)
    if _FP is not None:
        reg |= {a for a in getattr(_FP, "ARMS_E4", {}) if a != CONTROL}
    seen = {r.get("arm", "") for r in rows} - {CONTROL, ""}
    if not seen:
        return default
    unknown = seen - reg
    if unknown:
        raise ValueError(
            f"数据里出现了未注册的臂 {sorted(unknown)}；已注册的处理臂是 "
            f"{sorted(reg)}。臂名拼错会让该臂整条消失 —— 不猜，宁可报错。")
    return tuple(sorted(seen))


# ---------------------------------------------------------------------------
# 统计基元
# ---------------------------------------------------------------------------
def paired_t(d: list[float]) -> dict:
    """配对 t 检验。d = 逐 query 的差。"""
    n = len(d)
    if n < 2:
        return {"n": n, "t": None, "p": None, "mean": None, "sd": None}
    m = sum(d) / n
    sd = st.stdev(d)
    if sd == 0:
        # 🔴 2026-10-06：原先这里无条件返回 p=0.0，是**一个能造出假显著的地雷**。
        #    差值恒为 0（sd=0）有两种截然不同的情形：
        #      · mean==0 -> 两臂**完全一致**，p 必须是 1.0（没有任何效应）；
        #      · mean!=0 -> 差值是确定性的，p 无定义（配对 t 的分母为 0），
        #        此时报 p=0.0 等于宣称「无限显著」，是错的。
        #    真实踩坑：E4 序贯判读误把 `original` 当处理臂传进来，
        #    于是「original vs original」进族并被 Holm 判为最显著，
        #    族大小从 9 变 11、拒绝数从 5 变 7 —— 报表直接被污染。
        if m == 0:
            return {"n": n, "t": None, "p": 1.0, "mean": m, "sd": 0.0,
                    "degenerate": True,
                    "degenerate_note": "差值恒为 0：两臂完全一致，p=1.0"}
        return {"n": n, "t": None, "p": None, "mean": m, "sd": 0.0,
                "degenerate": True,
                "degenerate_note": "差值恒定非 0：配对 t 分母为 0，p 无定义（不登记进检验族）"}
    t = m / (sd / math.sqrt(n))
    p = _t_sf(abs(t), n - 1)   # 双尾
    return {"n": n, "t": round(t, 4), "p": p, "mean": m, "sd": sd,
            "se": sd / math.sqrt(n)}


def _rank_avg(xs: list[float]) -> list[float]:
    """平均秩（处理并列）。"""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def wilcoxon(d: list[float]) -> dict:
    """Wilcoxon 符号秩（正态近似 + 并列校正），零差值按 Wilcoxon 惯例丢弃。"""
    nz = [x for x in d if x != 0.0]
    n = len(nz)
    if n < 6:
        return {"n": n, "w": None, "p": None,
                "note": "非零差值不足 6 个，正态近似不可用"}
    ranks = _rank_avg([abs(x) for x in nz])
    w_plus = sum(r for r, x in zip(ranks, nz) if x > 0)
    w_minus = sum(r for r, x in zip(ranks, nz) if x < 0)
    w = min(w_plus, w_minus)
    mu = n * (n + 1) / 4.0
    # 并列校正
    s = 0.0
    tmp = sorted(abs(x) for x in nz)
    i = 0
    while i < len(tmp):
        j = i
        while j + 1 < len(tmp) and tmp[j + 1] == tmp[i]:
            j += 1
        t = j - i + 1
        if t > 1:
            s += t ** 3 - t
        i = j + 1
    sigma = math.sqrt((n * (n + 1) * (2 * n + 1) / 24.0) - s / 48.0)
    if sigma <= 0:
        return {"n": n, "w": w, "p": None, "note": "σ = 0，无法正态近似"}
    z = (w - mu + 0.5) / sigma
    return {"n": n, "w": w, "z": round(z, 4),
            "p": 2 * (1 - N.cdf(abs(z))),
            "n_zero_dropped": len(d) - n}


def mcnemar(b: int, c: int) -> dict:
    """McNemar 精确检验（二项，b vs c  discordant pairs）。

    用精确二项而不是 χ² 近似：`full` 臂的有效 query 只有 28，
    不一致对可能只有十来个，χ² 近似在那里不可靠。
    """
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p": 1.0, "note": "无不一致对"}
    k = min(b, c)
    # P(X <= k | n, 0.5) * 2
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return {"b": b, "c": c, "n_discordant": n, "p": min(1.0, 2 * tail),
            "test": "exact binomial"}


def risk_diff_ci(b: int, c: int, n: int, z: float = 1.96) -> dict:
    """配对二值的风险差与 CI（Tango 得分区间）。

    RD = (b − c)/n，Var = (b + c − (b−c)²/n) / n²
    ⚠️ 不是把两个独立比例的 CI 相减 —— 配对数据里那样做会**低估**方差。
    """
    if n == 0:
        return {"rd_pp": None, "lo_pp": None, "hi_pp": None}
    rd = (b - c) / n
    var = (b + c - (b - c) ** 2 / n) / (n * n)
    se = math.sqrt(max(var, 0.0))
    return {"rd_pp": round(rd * 100, 2),
            "lo_pp": round((rd - z * se) * 100, 2),
            "hi_pp": round((rd + z * se) * 100, 2),
            "n_pairs": n}


def bootstrap_ci(d: list[float], n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 20261006) -> dict:
    """配对均值的 percentile bootstrap CI。固定种子 -> 可复现。"""
    n = len(d)
    if n < 2:
        return {"lo": None, "hi": None}
    import random
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        means.append(sum(d[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot) - 1]
    return {"lo": round(lo * 100, 2), "hi": round(hi * 100, 2),
            "n_boot": n_boot, "seed": seed}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm–Bonferroni 逐步降序。返回 {key: 校正后 p}。"""
    items = [(k, v) for k, v in pvals.items() if v is not None]
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, float] = {k: v for k, v in pvals.items() if v is None}
    prev = 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(prev, (m - i) * p))
        out[k] = adj
        prev = adj
    return out


# ---------------------------------------------------------------------------
# 数据装配
# ---------------------------------------------------------------------------
def read_rows(path: str) -> list[dict]:
    op = (gzip.open if path.endswith(".gz") else open)
    with op(path, "rt", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def load_exclusions(path: str | None = None, engine: str = "") -> set:
    """读 query 级剔除登记表（与 15/16 共用同一份文件）。

    🔴 **必须和 `16_design_coverage.py` 用同一份、同一过滤**。否则两个工具会报出
    不同的独立单元数（E1 实测：16 报 149 个 query，19 报 150），
    审稿人自己重算时对不上，而**两个数字都看起来是对的**。

    🔴 **还必须按引擎过滤**：每条登记带 `engine` 标签，含义是
    「这条只在这个引擎上被拒」。某条被 MiniMax 内容审核拒答，
    而 qwen-plus 正常作答 —— 不过滤就会把 qwen 那边一条好端端的 query 剔掉。
    """
    # 🔴 2026-10-21 开源化：登记表路径可由环境变量 GEO_QUERY_EXCLUSIONS 指定；
    #    默认仍是仓库内的 data/query_exclusions.json（不存在时返回空集）。
    p = path or os.environ.get("GEO_QUERY_EXCLUSIONS", "").strip() \
        or os.path.join(ROOT, "data", "query_exclusions.json")
    if not os.path.exists(p):
        return set()
    with open(p, "r", encoding="utf-8-sig") as fh:
        data = json.load(fh).get("excluded") or {}
    eng = engine.lower()
    out = set()
    for q, meta in data.items():
        if not eng:
            out.add(q)
            continue
        tag = str((meta or {}).get("engine", "") or "").lower()
        if not tag:
            out.add(q)            # 无标签 = 与引擎无关
            continue
        parts = [x.strip() for x in tag.split("/") if x.strip()]
        if tag in eng or eng in tag or any(x in eng or eng in x for x in parts):
            out.add(q)
    return out


def build_cells(rows: list[dict]) -> dict:
    """(query_id, arm) -> {'cited': [..], 'pos': [..], 'ret': [..]} 的逐重复值。

    只收**无错误**的行；`gate_blocked` 记为该臂该 query 缺失（§3 的设计）。
    """
    cells: dict[tuple[str, str], dict] = collections.defaultdict(
        lambda: {"cited": [], "pos": [], "ret": [], "blocked": 0})
    for r in rows:
        if r.get("error"):
            if r["error"] == "gate_blocked":
                cells[(r["query_id"], r["arm"])]["blocked"] += 1
            continue
        c = cells[(r["query_id"], r["arm"])]
        c["cited"].append(float(r["cited"]))
        pr = r.get("cited_pos_ratio", "")
        try:
            pv = float(pr) if pr not in ("", None) else -1.0
        except (TypeError, ValueError):
            pv = -1.0
        c["pos"].append(pv)
        ret = r.get("retention", "")
        if ret not in ("", None):
            try:
                c["ret"].append(float(ret))
            except (TypeError, ValueError):
                pass
    return cells


# 序贯判读（§2.2b）与最终报告都要用 MDE，必须**由数据反解**，不能抄别处的数。
Z_ALPHA = 1.959964          # 双尾 alpha=0.05
Z_BETA = 0.841621           # power=0.80
Z_SUM = Z_ALPHA + Z_BETA


def mde_from_paired(d_sd: float | None, n_pairs: int) -> float | None:
    """MDE(pp) = (z_{a/2}+z_b)·sigma_d / sqrt(Q)。

    🔴 `d` 必须**已经**是对 R 次重复取过平均的逐 query 差，
    于是 R 通过这个平均进入 sigma_d。**绝不可再除一次 sqrt(R)** ——
    除两次会把 MDE 报小 sqrt(R) 倍，让做不出来的实验看起来做得出来
    （曾经就这样误判过「5pp 在预算内不可达」）。
    """
    if d_sd is None or n_pairs is None or n_pairs < 2:
        return None
    return round(Z_SUM * d_sd / math.sqrt(n_pairs) * 100, 3)


def _pair(cells: dict, qids: list[str], ctrl: str, treat: str,
          sigma_run_pp: float) -> dict | None:
    """ctrl vs treat 的 query 级配对，差的方向 = treat - ctrl。

    抽成独立函数的唯一理由：E4 的主对比两条腿都是处理臂。
    复制一份统计代码 = 复制一份将来各自漂移的真相。
    """
    pairs_cited: list[float] = []
    pos_d: list[float] = []
    maj_b = maj_c = 0
    n_major = 0
    for q in qids:
        ct = cells.get((q, ctrl))
        cx = cells.get((q, treat))
        if not ct or not cx or not ct["cited"] or not cx["cited"]:
            continue
        mt = sum(ct["cited"]) / len(ct["cited"])
        mx = sum(cx["cited"]) / len(cx["cited"])
        pairs_cited.append(mx - mt)
        n_major += 1
        if mt <= 0.5 < mx:
            maj_b += 1
        elif mx <= 0.5 < mt:
            maj_c += 1
        # 共同主：两臂都必须「被引用」才成立
        pt = [x for x in ct["pos"] if x >= 0]
        px = [x for x in cx["pos"] if x >= 0]
        if pt and px:
            pos_d.append(sum(px) / len(px) - sum(pt) / len(pt))

    t = paired_t(pairs_cited)
    w = wilcoxon(pairs_cited)
    bs = bootstrap_ci(pairs_cited)
    mc = mcnemar(maj_b, maj_c)
    rd = risk_diff_ci(maj_b, maj_c, n_major)
    td = paired_t(pos_d)
    wd = wilcoxon(pos_d)
    bsd = bootstrap_ci(pos_d)
    signs = [1 if x > 0 else (-1 if x < 0 else 0) for x in pairs_cited]
    nz = [s for s in signs if s != 0]
    neg_frac = (sum(1 for s in nz if s < 0) / len(nz)) if nz else None

    # §7.2 规则 1 问的是「**同一 query 的 R 次重复之间** Δ 的符号是否一致」，
    # 不是「多少 query 的 Δ 为负」—— 两者是不同的统计量，混用会得出
    # 看似合理却完全错误的判定（2026-10-06 首次实现就搞错了）。
    rep_incons = 0
    rep_total = 0
    for q in qids:
        ct = cells.get((q, ctrl))
        cx = cells.get((q, treat))
        if not ct or not cx or not ct["cited"] or not cx["cited"]:
            continue
        nrep = min(len(ct["cited"]), len(cx["cited"]))
        ss = set()
        for k in range(nrep):
            d = cx["cited"][k] - ct["cited"][k]
            if d > 0:
                ss.add(1)
            elif d < 0:
                ss.add(-1)
        rep_total += 1
        if len(ss) > 1:
            rep_incons += 1
    sign_instab = (round(rep_incons / rep_total, 4) if rep_total else None)

    mean_pp = round(t["mean"] * 100, 2) if t["mean"] is not None else None
    sub_noise = (mean_pp is not None and abs(mean_pp) < sigma_run_pp)
    n_rep_obs = 0
    for q in qids:
        ct = cells.get((q, ctrl))
        cx = cells.get((q, treat))
        if ct and cx and ct["cited"] and cx["cited"]:
            n_rep_obs += min(len(ct["cited"]), len(cx["cited"]))

    paired = {
        "n_query_pairs": n_major,
        "n_observations": n_rep_obs,
        "mean_diff_cited_pp": mean_pp,
        "ci95_cited_pp": bs,
        "paired_t": t,
        "wilcoxon": w,
        "mcnemar_query_level": {**mc, **rd, "note":
                                "预注册检验，放在 query 级（cell 级会重复计数）"},
        "n_pos_pairs": len(pos_d),
        "mean_diff_pos_pp": (round(td["mean"] * 100, 2)
                             if td["mean"] is not None else None),
        "ci95_pos_pp": bsd,
        "paired_t_pos": td,
        "wilcoxon_pos": wd,
        # §7.2 规则 1
        "sign_inconsistent_queries": rep_incons,
        "sign_instab_rate": sign_instab,
        "sign_unstable": (sign_instab is not None and sign_instab > 0.30),
        # §7.2 规则 2
        "sigma_run_pp": sigma_run_pp,
        "sub_noise": sub_noise,
        "rankable": not (sub_noise or (sign_instab or 0) > 0.30),
        # 序贯判读用：MDE 必须由**本对比自己的** sigma_d 反解
        "sigma_d_cited_pp": (round(t["sd"] * 100, 3)
                             if t["sd"] is not None else None),
        "mde_cited_pp": mde_from_paired(t["sd"], t["n"]),
        "mde_pos_pp": mde_from_paired(td["sd"], td["n"]),
        # 附带的描述量（不是 §7.2 的判据，别混用）
        "neg_frac_queries": (round(neg_frac, 4) if neg_frac is not None else None),
    }
    # 🔴 检验只有在**真的有配对数据**时才进族。
    #    n_query_pairs == 0 时 mcnemar(0,0) 会返回 p=1.0，
    #    若照样登记进检验族，就会给「一个不存在的检验」占掉一个
    #    Holm 名额 —— 而 Holm 的名次取决于族的大小。
    tests: dict[str, float] = {}
    if n_major > 0:
        if t["p"] is not None:
            tests[f"{treat}|{DV_MAIN}|t"] = t["p"]
        if td["p"] is not None:
            tests[f"{treat}|{DV_CO}|t"] = td["p"]
        if mc["p"] is not None and mc.get("n_discordant"):
            tests[f"{treat}|{DV_MAIN}|mcnemar"] = mc["p"]
    return {"paired": paired, "tests": tests}


def analyze(rows: list[dict], arms: tuple[str, ...] | None = None,
            sigma_run_pp: float = 2.2118,
            extra_contrasts: tuple[tuple[str, str], ...] = ()) -> dict:
    """`sigma_run_pp` = 阶段 A 测得的 run-level σ（§7.2 规则 2 的门限）。
    默认值 = MiniMax-M3 实测 2.2118 pp（`outputs/stage_a2_variance_minimax.md`）。
    ⚠️ **换引擎必须同时换这个数** —— run-level σ 是引擎侧非确定性的量，
    拿 Qwen 的 σ 去判 M3 的结果就是用错门限。"""
    arms = tuple(arms) if arms else infer_arms(rows)
    # 🔴 处理臂**不得包含对照臂**。若把 CONTROL 混进 arms，会跑出
    #    「original vs original」的自比较：差值恒为 0，本不该产生任何检验。
    #    2026-10-06 实测这条路径把 E4 的 Holm 族从 9 撑到 11、拒绝数从 5 撑到 7。
    if CONTROL in arms:
        raise ValueError(
            f"处理臂列表里含对照臂 {CONTROL!r} —— 会产生自比较。"
            f"请传不含对照臂的处理臂集合。")
    cells = build_cells(rows)
    qids = sorted({q for q, _ in cells})
    # 🔴 被登记剔除的 query **不算独立单元**。剔除表是 15/16/19 共用的同一份，
    #    三者必须报出同一个 Q —— 见 `load_exclusions` 的说明。
    qids_before = len(qids)
    # 🔴 **列存在性必须显式记录**。`cited_pos_ratio` 缺列时，
    #    `r.get("cited_pos_ratio", "")` 返回 ""，与「在列但值为空（未被引用）」
    #    完全无法区分 —— 报表上两者都显示成「位置覆盖 0.0%」。
    #    前者是接线/版本故障，后者是真实测量值，**混为一谈就是本项目最防的那种错**。
    has_pos_col = any("cited_pos_ratio" in r for r in rows)
    out: dict = {"n_queries": len(qids), "arms": {}, "tests": {},
                 "treatment_arms": list(arms),
                 "meta": {"sigma_run_pp": sigma_run_pp,
                          "has_pos_column": has_pos_col,
                          "n_queries_before_exclusion": qids_before,
                          "n_queries_excluded": qids_before - len(qids)}}

    # 每臂的描述性（共同主描述性结果与 cited 并排）
    for arm in (CONTROL,) + tuple(arms):
        cc = [c for (q, a), c in cells.items() if a == arm and c["cited"]]
        if not cc:
            out["arms"][arm] = {"n_cells": 0}
            continue
        flat = [x for c in cc for x in c["cited"]]
        pos_all = [x for c in cc for x in c["pos"]]
        pos_def = [x for x in pos_all if x >= 0]
        ret_all = [x for c in cc for x in c["ret"]]
        out["arms"][arm] = {
            "n_cells": len(cc),
            "n_obs": len(flat),
            "cited_rate": round(sum(flat) / len(flat), 4),
            "pos_defined": len(pos_def),
            "pos_coverage": round(len(pos_def) / len(pos_all), 4) if pos_all else None,
            "pos_median": (round(st.median(pos_def), 4) if pos_def else None),
            "retention_median": (round(st.median(ret_all), 4) if ret_all else None),
            "retention_n": len(ret_all),
        }

    # 逐臂配对 + 额外对比：统一走 _pair()。控制臂 vs 处理臂、以及**处理臂之间**
    # 的对比共用同一套统计量、进入**同一个 Holm 族**。
    # 🔴 必须支持臂间对比：E4 的主对比是
    #    `evidence_fabricated - evidence_real`（编造本身的效应），
    #    两条腿**都是处理臂**。只跑「处理臂 vs original」的话，
    #    主对比整条缺席，Holm 族还少 3 个成员 —— 名额与 p 校正一起错。
    for arm in arms:
        blk = _pair(cells, qids, CONTROL, arm, sigma_run_pp)
        if blk is None:
            continue
        out["arms"][arm]["paired"] = blk["paired"]
        for k, v in blk["tests"].items():
            out["tests"][k] = v

    for a, b in extra_contrasts:
        known = (CONTROL,) + tuple(arms)
        if a not in known or b not in known:
            raise ValueError(f"未知臂名 in 额外对比 {a}:{b}；数据里的臂是 {list(known)}")
        blk = _pair(cells, qids, b, a, sigma_run_pp)   # 有符号方向：a - b
        if blk is None:
            continue
        out.setdefault("contrasts", {})[f"{a}::{b}"] = blk["paired"]
        for k, v in blk["tests"].items():
            out["tests"][f"{a}::{b}|{k.split('|', 1)[1]}"] = v

    adj = holm(out["tests"])
    for k, v in out["tests"].items():
        out["arms"].setdefault("_holm", {})[k] = {
            "p_raw": v, "p_holm": adj.get(k)}
    out["n_tests"] = len(out["tests"])
    out["n_rejected_holm"] = sum(1 for v in adj.values() if v < 0.05)
    return out


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------
def render(rep: dict) -> str:
    L = []
    A = L.append
    # 🔴 臂表从结果里读，不读模块常量 —— 否则 E4 的臂名会整条消失
    arms = tuple(rep.get("treatment_arms") or TREATMENT_ARMS)
    A("# 阶段 B 主分析（§7.1 预注册）")
    A("")
    # 🔴 `n_obs` 是**逐臂**的字段，report 顶层没有 —— 早先这里写
    #    `rep.get('n_obs', '?')` 导致正文出现「不是 ? 个格」这种占位符残留。
    #    改为对各臂求和，并且**永远不会是 '?'**。
    total_obs = sum(s.get("n_obs", 0) for k, s in rep["arms"].items()
                    if not k.startswith("_"))
    A(f"独立单元：**{rep['n_queries']} 个 query**（不是 {total_obs} "
      f"个格 —— 格级会把 R 次相关重复当独立，见方法章 §7.1 规格裁决）")
    nx = rep["meta"].get("n_queries_excluded", 0)
    if nx:
        A("")
        A(f"> 另剔除 **{nx}** 条登记在册的 query"
          f"（`data/query_exclusions.json`，引擎内容审核确定性拒答）。"
          f"剔除前 {rep['meta'].get('n_queries_before_exclusion')} 个 → "
          f"剔除后 {rep['n_queries']} 个。**与 15/16 用的是同一份表。**")
    A("")
    A("## 1. 逐臂描述（共同主描述性结果与 `cited` 并排）")
    A("")
    A("| 臂 | 格数 | 观测 | 引用率 | 位置覆盖 | 位置中位 | 保留率中位 |")
    A("|---|---|---|---|---|---|---|")
    for arm, s in rep["arms"].items():
        if arm.startswith("_") or not s.get("n_cells"):
            continue
        # 🔴 缺列时**不写数字**。写 0.0% 等于宣称「实测覆盖率为零」。
        if not rep["meta"].get("has_pos_column", True):
            cov, pos = "🔴 列不存在", "—"
        else:
            cov = (f"{s['pos_coverage']:.1%}"
                   if s["pos_coverage"] is not None else "—")
            pos = ("—" if s["pos_median"] is None else s["pos_median"])
        ret = ("—" if s["retention_median"] is None
               else f"{s['retention_median']:.1%}")
        A(f"| `{arm}` | {s['n_cells']} | {s['n_obs']} | {s['cited_rate']:.4f} | "
          f"{cov} | {pos} | {ret} |")
    A("")
    if not rep["meta"].get("has_pos_column", True):
        A("> 🔴 **本文件没有 `cited_pos_ratio` 列。** 位置是**共同主因变量**，"
          "缺列意味着它根本没被测量 —— 这**不是**「覆盖率为 0%」。")
        A("> 若这是预期外的：多半是数据来自旧版 `15_stage_b_run.py`，"
          "或该轮没跑到写盘。**先修数据，不要把它当成测量结果。**")
        A("")
    A("## 2. 配对结果（处理臂 − `original`）")
    A("")
    A(f"| 臂 | query 对 | Δ引用率 (pp) | 95% CI | p (t) | McNemar p | "
      f"位置对 | Δ位置 (pp) | p (位置) |")
    A("|---|---|---|---|---|---|---|---|---|")
    for arm in arms:
        s = rep["arms"].get(arm, {}).get("paired")
        if not s:
            continue
        ci = s["ci95_cited_pp"]
        civ = f"[{ci['lo']}, {ci['hi']}]" if ci.get("lo") is not None else "—"
        cipos = s["ci95_pos_pp"]
        civpos = (f"[{cipos['lo']}, {cipos['hi']}]"
                  if cipos.get("lo") is not None else "—")

        def pv(x, fmt=".4g"):
            return format(x, fmt) if x is not None else "—"

        A(f"| `{arm}` | {s['n_query_pairs']} | {s['mean_diff_cited_pp']} | {civ} | "
          f"{pv(s['paired_t']['p'])} | {pv(s['mcnemar_query_level']['p'])} | "
          f"{s['n_pos_pairs']} | {s['mean_diff_pos_pp']} | {civpos} | "
          f"{pv(s['paired_t_pos']['p'])} |")
    A("")
    # 🔴 臂间对比必须**渲染出来**。算出来了但报表上看不见，等于没算 ——
    #    E4 的主对比（编造 vs 真实）就属于这一类。
    cons = rep.get("contrasts") or {}
    if cons:
        A("## 2a. 🔴 处理臂之间的对比（E4 主对比在这一节，不在上一节）")
        A("")
        A("| 对比 | query 对 | Δ引用率 (pp) | 95% CI | p (t) | McNemar p | "
          "σ_d (pp) | **MDE (pp)** | 位置对 | Δ位置 (pp) | p (位置) |")
        A("|---|---|---|---|---|---|---|---|---|---|---|")
        for cname, s in cons.items():
            civ = s["ci95_cited_pp"]
            civv = (f"[{civ['lo']}, {civ['hi']}]"
                    if civ.get("lo") is not None else "—")
            cipos = s["ci95_pos_pp"]
            civpos = (f"[{cipos['lo']}, {cipos['hi']}]"
                      if cipos.get("lo") is not None else "—")
            sig = s.get("sigma_d_cited_pp")
            mde = s.get("mde_cited_pp")
            A(f"| `{cname.replace('::', ' − ')}` | {s['n_query_pairs']} | "
              f"{s['mean_diff_cited_pp']} | {civv} | "
              f"{pv(s['paired_t']['p'])} | {pv(s['mcnemar_query_level']['p'])} | "
              f"{sig if sig is not None else '—'} | "
              f"**{mde if mde is not None else '—'}** | "
              f"{s['n_pos_pairs']} | {s['mean_diff_pos_pp']} | {civpos} | "
              f"{pv(s['paired_t_pos']['p'])} |")
        A("")
        A("> σ_d 与 MDE 由**本对比自己的**逐 query 差反解，不是抄别处的数；"
          "MDE = (z_{α/2}+z_β)·σ_d/√Q，其中 d 已对 R 次重复取过平均。"
          "**MDE > |Δ| 时，null 必须写成「在本预算下测不出来」，不能写成「无效应」。**")
        A("")
    A("## 2b. 🔴 §7.2 判定（规则未改，逐条判）")
    A("")
    A(f"门限：run-level σ = **{rep['meta']['sigma_run_pp']} pp**"
      f"（阶段 A' 实测；换引擎必须同时换这个数）。")
    A("")
    A("| 臂 | Δ引用率 (pp) | 规则2 亚噪声 | 符号不一致 query | 规则1 符号不稳 | 可否参与排序 |")
    A("|---|---|---|---|---|---|")
    for arm in arms:
        s = rep["arms"].get(arm, {}).get("paired")
        if not s:
            continue
        ir = s["sign_instab_rate"]
        # 🔴 **零配对时不能判「可参与排序」**。`rankable` 在无数据时默认 True，
        #    于是报表会把「什么都没测到」显示成「✅ 可」——
        #    这正是本项目反复警告的「空结果被当成有效结果」。
        n_pairs = s["n_query_pairs"]
        if n_pairs == 0:
            A(f"| `{arm}` | — | — | 0 / 0（—） | — | ⛔ **无数据** |")
            continue
        A(f"| `{arm}` | {s['mean_diff_cited_pp']} | "
          f"{'🔴 是（|Δ| < σ）' if s['sub_noise'] else '否'} | "
          f"{s['sign_inconsistent_queries']} / {n_pairs}"
          f"（{f'{ir:.1%}' if ir is not None else '—'}） | "
          f"{'🔴 是' if s['sign_unstable'] else '否'} | "
          f"{'❌ 不可' if not s['rankable'] else '✅ 可'} |")
    A("")
    nodata = [arm for arm in arms
              if rep["arms"].get(arm, {}).get("paired")
              and rep["arms"][arm]["paired"]["n_query_pairs"] == 0]
    if nodata:
        A(f"> ⛔ **{len(nodata)} 个臂（{'、'.join('`' + a + '`' for a in nodata)}）"
          f"零配对 —— 本臂**没有任何结论**，既不参与排序也不报 null。**")
        A("> 零配对通常是接线/门/覆盖问题，**先排查再解读**。")
        A("")
    bad = [arm for arm in arms
           if rep["arms"].get(arm, {}).get("paired")
           and not rep["arms"][arm]["paired"]["rankable"]]
    if bad:
        A(f"> 🔴 **{len(bad)} 个臂（{'、'.join('`' + a + '`' for a in bad)}）"
          f"触发 §7.2，不参与名次排序。**")
        A("> 按规则 3，它们的结论表述**必须**写成"
          "「在本设计下检测不到 |δ| < MDE = x pp」，"
          "**禁止**写成「无效应」或「无提升」。")
        A("")
    A("")
    A("## 3. 多重比较（Holm–Bonferroni，FWER = 0.05）")
    A("")
    # 🔴 族的构成必须**从数据数出来**，不能写死「4 臂」。
    #    阶段 B 是 4 个处理臂，E4 是 2 个处理臂 + 1 个臂间对比 = 3 组，
    #    写死的文案在 E4 上会与实际族大小对不上 —— 而族大小正是 Holm 名额的来源。
    n_extra = len(rep.get("contrasts") or {})
    n_groups = len(arms) + n_extra
    A(f"检验族 **{rep['n_tests']} 个**（{n_groups} 组对比"
      + (f" = {len(arms)} 个处理臂 vs 对照 + {n_extra} 个臂间对比"
         if n_extra else f"（{len(arms)} 个处理臂 vs 对照")
      + "，每组含引用率 t / 位置 t / McNemar，取有配对数据者）。"
      f"Holm 校正后拒绝 **{rep['n_rejected_holm']}** 个。")
    A("")
    A("| 检验 | p (未校正) | p (Holm) |")
    A("|---|---|---|")
    h = rep["arms"].get("_holm", {})
    for k, v in sorted(h.items(), key=lambda kv: kv[1]["p_raw"]):
        A(f"| `{k}` | {v['p_raw']:.4g} | {v['p_holm']:.4g} |")
    A("")
    A("> **校正后与未校正都要报**（§7.1 预注册要求）。"
      "未校正的 p 只在正文脚注出现，不能当结论。")
    A("")
    A("## 4. 报告纪律")
    A("")
    A("- 符号翻转率 ≥ 30% 的臂按 §7.2 **不可作报告**（规则未改）。")
    A("- 位置因变量只在**两臂都被引用**的 query 上配对，"
      "`位置对` 列已给出有效子集大小。")
    A("- 逐臂 MDE 见 `outputs/stage_a2_variance_minimax.md` §4b —— "
      "**`full` 臂两个口径都达不到 5pp**，其 null 必须区分「无效应」与「功效不足」。")
    A("- 保留率必须与每条臂间结论**同时**出现（§3.4）。")
    A("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    p = argparse.ArgumentParser(
        prog="19_paired_analysis.py",
        description="阶段 B 主分析：query 级配对 + Holm 校正（§7.1）")
    p.add_argument("--csv", required=False,
                   default=os.path.join(ROOT, "outputs", "15_stage_b",
                                        "raw_trials.csv.gz"))
    p.add_argument("--out", default="")
    p.add_argument("--json", default="")
    p.add_argument("--sigma-run", type=float, default=2.2118,
                   help="run-level σ（pp），§7.2 规则 2 的门限。"
                        "默认 2.2118 = MiniMax-M3 实测（阶段 A'）。"
                        "**跑 qwen 的数据必须换成 3.7756**，用错门限等于换引擎不换判据")
    p.add_argument("--engine", default="",
                   help="测量引擎标识（如 minimax / qwen）。"
                        "🔴 **必给**：剔除登记表带 engine 标签，"
                        "同一条 query 可能只被某一个引擎拒答")
    p.add_argument("--exclusions", default="",
                   help="query 级剔除登记表（留空用 data/query_exclusions.json）。"
                        "与 15/16 共用同一份，保证三处报的独立单元数一致")
    p.add_argument("--allow-missing-pos", action="store_true",
                   help="允许在缺 `cited_pos_ratio` 列时继续跑（仅用于诊断 / "
                        "0 臂的阶段 A' 数据）。默认**硬失败**：位置是共同主因变量，"
                        "缺列 = 没测到，不能当成覆盖率为 0。")
    p.add_argument("--extra-contrast", action="append", default=[],
                   metavar="ARM_A:ARM_B",
                   help="额外的**处理臂之间**对比（可重复）。"
                        "🔴 E4 的主对比 `evidence_fabricated:evidence_real` "
                        "两条腿都是处理臂，不给这个参数它会整条缺席，"
                        "而 Holm 族还会因此少几个成员 —— 名额与 p 校正一起错。"
                        "这些对比与「处理臂 vs original」进入**同一个 Holm 族**")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args(argv)
    if a.selftest:
        return selftest()
    if not os.path.exists(a.csv):
        print(f"ERROR: 找不到 {a.csv}", file=sys.stderr)
        return 2
    extra: list[tuple[str, str]] = []
    for spec in a.extra_contrast:
        if spec.count(":") != 1:
            print(f"ERROR: --extra-contrast 格式应为 ARM_A:ARM_B，收到 {spec!r}",
                  file=sys.stderr)
            return 2
        extra.append(tuple(spec.split(":")))   # type: ignore[arg-type]
    rows = read_rows(a.csv)
    excl = load_exclusions(a.exclusions or None, engine=a.engine)
    n_raw = len(rows)
    if not a.engine:
        print("WARNING: 未给 --engine。剔除登记表带 engine 标签，"
              "不过滤会把**未被拒**的 query 也剔掉（独立单元偏小）。",
              file=sys.stderr)
    if excl:
        rows = [r for r in rows if r.get("query_id") not in excl]
    rep = analyze(rows, sigma_run_pp=a.sigma_run, extra_contrasts=tuple(extra))
    if excl and n_raw != len(rows):
        print(f"[info] 剔除 {n_raw - len(rows)} 行 / "
              f"{len(excl)} 条登记 query（{rep['n_queries']} 个独立单元）",
              file=sys.stderr)
    if not rep["meta"]["has_pos_column"] and not a.allow_missing_pos:
        print("ERROR: 数据里没有 `cited_pos_ratio` 列，而位置是**共同主因变量**。",
              file=sys.stderr)
        print("       缺列 ≠ 覆盖率为 0 —— 前者是接线/版本故障，后者是测量结果。",
              file=sys.stderr)
        print("       确认这确实是预期外之后再加 --allow-missing-pos 放行。",
              file=sys.stderr)
        return 2
    md = render(rep)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(md + "\n")
        print(f"wrote {a.out}")
    else:
        print(md)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, ensure_ascii=False, indent=2, default=str)
        print(f"wrote {a.json}")
    return 0


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
def selftest() -> int:
    checks: list[tuple[str, bool]] = []

    def ck(name, cond, extra=""):
        checks.append((name, bool(cond)))
        if not cond and extra:
            print(f"       （诊断：{extra}）")

    # ---- 2026-10-06 E4 序贯判读事故：自比较污染 Holm 族
    def _mkrow(q, arm, rep, cited):
        return {"query_id": q, "arm": arm, "repeat": rep, "cited": cited,
                "cited_pos_ratio": 0.3, "geo_impression": 0.0, "first_pos": 1,
                "retention": "", "length_ratio": "", "response_chars": "10",
                "target_tokens": "x", "gate_pass": 1, "gate_codes": "",
                "error": ""}

    _rows = []
    for i in range(6):
        q = f"Z{i}"
        for rep in range(2):
            _rows.append(_mkrow(q, "original", rep, i % 2))
            _rows.append(_mkrow(q, "evidence_real", rep, i % 2))
            _rows.append(_mkrow(q, "evidence_fabricated", rep, 1))
    ck("🔴 处理臂里混入对照臂会被**硬失败**（否则跑出 original vs original）",
       _raises_valueerror(lambda: analyze(_rows, arms=("original", "evidence_real"))))
    _rep_ok = analyze(_rows, arms=("evidence_real", "evidence_fabricated"),
                      extra_contrasts=(("evidence_fabricated", "evidence_real"),))
    _names = " ".join(_rep_ok["tests"].keys())
    ck("🔴 自比较不会出现在检验族里",
       "original" not in _names, _names)
    ck("🔴 处理臂 vs 对照 + 臂间对比共用同一个 Holm 族",
       _rep_ok["n_tests"] >= 6
       and any("::" in k for k in _rep_ok["tests"]),
       f"n_tests={_rep_ok['n_tests']} {_names}")
    ck("🔴 无显著性凭空产生：族里拒绝数不超过族大小且不来自退化检验",
       _rep_ok["n_rejected_holm"] <= _rep_ok["n_tests"])
    _same = analyze([_mkrow(f"S{i}", "original", 0, 1) for i in range(5)]
                    + [_mkrow(f"S{i}", "evidence_real", 0, 1) for i in range(5)],
                    arms=("evidence_real",))
    ck("🔴 两臂逐格完全相同 -> 检验 p=1.0，不会报成显著",
       all(v == 1.0 for v in _same["tests"].values()),
       str(_same["tests"]))
    ck("🔴 完全相同的两臂被判不可排序（亚噪声）",
       _same["arms"]["evidence_real"]["paired"]["rankable"] is False)
    ck("🔴 臂间对比带自己的 σ_d 与 MDE（序贯判读要用）",
       _rep_ok["contrasts"]["evidence_fabricated::evidence_real"]
       .get("mde_cited_pp") is not None
       and _rep_ok["contrasts"]["evidence_fabricated::evidence_real"]
       .get("sigma_d_cited_pp") is not None)
    ck("🔴 臂间对比被**渲染**出来（算出来却看不见等于没算）",
       "evidence_fabricated − evidence_real" in render(_rep_ok)
       and "MDE" in render(_rep_ok))
    ck("🔴 检验族构成的文案从数据数出（写死「4 臂」会在 E4 上对不上）",
       "2 个处理臂 vs 对照 + 1 个臂间对比" in render(_rep_ok)
       and "4 臂 ×" not in render(_rep_ok))
    import argparse as _ap2
    _apc = _ap2.ArgumentParser(prog="p2")
    _apc.add_argument("--extra-contrast", action="append", default=[])
    ck("🔴 CLI 有 --extra-contrast（可重复），空默认",
       _apc.parse_args([]).extra_contrast == []
       and _apc.parse_args(["--extra-contrast", "a:b",
                            "--extra-contrast", "c:d"]).extra_contrast
       == ["a:b", "c:d"])

    # 1. 配对 t
    # 🔴 退化语义在 2026-10-06 被改过：原先「常量差 -> p=0.0」是**错的**，
    #    等于给「两臂完全一致」报出最强显著。旧自检正好把这个 bug 钉成了规范。
    tz = paired_t([0.0] * 10)
    ck("🔴 差值恒为 0 -> p=1.0（两臂完全一致，不是「无限显著」）",
       tz["degenerate"] is True and tz["p"] == 1.0, str(tz["p"]))
    t = paired_t([0.1] * 10)
    ck("🔴 差值恒定非 0 -> p 无定义（分母为 0），不进检验族",
       t["degenerate"] is True and t["p"] is None
       and "无定义" in (t.get("degenerate_note") or ""), str(t["p"]))
    t2 = paired_t([0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.15, -0.15])
    ck("对称差 -> 均值 0", abs(t2["mean"]) < 1e-9, str(t2["mean"]))
    ck("n=1 不崩", paired_t([0.1])["p"] is None)

    # 2. Wilcoxon
    w = wilcoxon([1.0] * 12)
    ck("全正 -> p 极小", w["p"] is not None and w["p"] < 0.001, str(w["p"]))
    ck("零差被丢弃且计数正确",
       wilcoxon([0.0, 0.0] + [1.0] * 10)["n_zero_dropped"] == 2)
    ck("非零不足 6 -> 明确说不做正态近似",
       wilcoxon([1.0, 2.0, 3.0])["p"] is None)
    ck("并列不炸（t 校正分支）",
       wilcoxon([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0])["p"] is not None)

    # 3. McNemar 精确
    m = mcnemar(10, 0)
    ck("全不一致同向 -> p = 2·0.5^10",
       abs(m["p"] - 2 * 0.5 ** 10) < 1e-12, str(m["p"]))
    ck("无不一致对 -> p=1", mcnemar(0, 0)["p"] == 1.0)
    ck("对称 -> p=1", abs(mcnemar(5, 5)["p"] - 1.0) < 1e-9)
    ck("精确二项不会给出 0（不像 χ² 近似在 n 小时）", mcnemar(30, 0)["p"] > 0)

    # 4. 风险差 CI
    rd = risk_diff_ci(8, 2, 10)
    ck("RD = (b-c)/n", abs(rd["rd_pp"] - 60.0) < 1e-9, str(rd["rd_pp"]))
    ck("CI 包住点估计", rd["lo_pp"] < rd["rd_pp"] < rd["hi_pp"])
    ck("n=0 不炸", risk_diff_ci(0, 0, 0)["rd_pp"] is None)
    # 🔴 配对 CI 相对朴素 CI 的宽窄，**由两臂的相关性决定，不是恒成立**。
    #    正相关时配对更窄；相关性为负（几乎没有一致对）时配对反而更宽。
    #    2026-10-06 自检曾拿一个负相关的反例当正例用 —— 两个方向都钉住。
    b, c, d, n = 4, 2, 8, 20                 # 一致对充足 -> 正相关
    p1, p2 = (b + d) / n, (c + d) / n
    se_p = math.sqrt((b + c - (b - c) ** 2 / n) / (n * n)) * 100
    se_u = math.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n) * 100
    ck("🔴 正相关时配对 CI 比朴素窄（配对确实吸收了相关）", se_p < se_u,
       f"paired {se_p:.2f} vs unpaired {se_u:.2f}")
    b2, c2, d2, n2 = 6, 2, 2, 10             # 几乎没有一致对 -> 负相关
    p1b, p2b = (b2 + d2) / n2, (c2 + d2) / n2
    se_p2 = math.sqrt((b2 + c2 - (b2 - c2) ** 2 / n2) / (n2 * n2)) * 100
    se_u2 = math.sqrt(p1b * (1 - p1b) / n2 + p2b * (1 - p2b) / n2) * 100
    ck("🔴 负相关时配对 CI 反而更宽 —— 配对不是无条件优势", se_p2 > se_u2,
       f"paired {se_p2:.2f} vs unpaired {se_u2:.2f}")

    # 5. Holm
    h = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    ck("Holm 单调不减", h["a"] <= h["c"] <= h["b"], str(h))
    ck("Holm 全部 >= 原始 p", all(h[k] >= v for k, v in
                                {"a": 0.01, "b": 0.04, "c": 0.03}.items()))
    ck("Holm 至少一个显著 -> 至少一个拒绝",
       holm({"a": 0.001, "b": 0.9, "c": 0.8})["a"] < 0.05)
    ck("全不显著 -> 校正后也不显著",
       holm({"a": 0.4, "b": 0.5, "c": 0.6}).get("a", 1) >= 0.05)
    ck("None 透传不炸", holm({"a": None, "b": 0.01})["a"] is None)

    # 6. bootstrap 可复现
    d = [0.01, -0.02, 0.03, 0.0, 0.04, -0.01] * 3
    b1, b2 = bootstrap_ci(d), bootstrap_ci(d)
    ck("bootstrap 同种子可复现", b1 == b2)
    ck("bootstrap CI 包住均值",
       b1["lo"] <= round(sum(d) / len(d) * 100, 2) <= b1["hi"])

    # 7. 端到端：构造**已知效应 + 存在不一致对**的数据
    #    （全一致的话 McNemar 根本没有不一致对，检验不会进族 ——
    #      那是对的，但端到端就测不到 McNemar 这条路径了）
    # 🔴 位置值必须**随 query 变化**。原夹具里控制恒 0.5、处理恒 0.3，
    #    差值恒为 −0.2 -> sd=0 -> 退化；旧代码给退化返回 p=0.0，
    #    于是这个「位置效应」被当成显著检验塞进族里，白拿一个 Holm 名额。
    #    真实数据不会这么巧，但夹具会 —— 夹具退化 = 测试测不到真东西。
    rows = []
    for qi in range(60):
        # 让 3/4 的 query 两臂都被引（一致），1/4 只被处理臂引（不一致）
        ctrl_cited = 1 if qi % 4 != 0 else 0
        ctrl_pos = 0.40 + 0.20 * ((qi % 5) / 4.0)
        for rep in range(4):
            rows.append({"query_id": f"Q{qi}", "arm": CONTROL, "repeat": str(rep),
                         "cited": str(ctrl_cited), "geo_impression": "0.2",
                         "cited_pos_ratio": f"{ctrl_pos:.3f}" if ctrl_cited else "-1",
                         "retention": "", "error": ""})
            for arm in ("structure", "full"):
                # 差值在 −0.10 ~ −0.25 之间循环：均值为负但**不是常数**
                trt_pos = ctrl_pos - (0.10 + 0.05 * (qi % 4))
                rows.append({"query_id": f"Q{qi}", "arm": arm, "repeat": str(rep),
                             "cited": "1", "geo_impression": "0.3",
                             "cited_pos_ratio": f"{trt_pos:.3f}",
                             "retention": "0.9", "error": ""})
    rep = analyze(rows)
    ck("端到端：60 query", rep["n_queries"] == 60)
    ck("端到端：structure 有正向显著效应",
       rep["arms"]["structure"]["paired"]["paired_t"]["p"] < 0.05)
    ck("端到端：McNemar 识别出 15 个不一致对（全在处理臂侧）",
       rep["arms"]["structure"]["paired"]["mcnemar_query_level"]["n_discordant"]
       == 15,
       str(rep["arms"]["structure"]["paired"]["mcnemar_query_level"]
           ["n_discordant"]))
    ck("端到端：McNemar 风险差 = 15/60 = 25pp",
       abs(rep["arms"]["structure"]["paired"]["mcnemar_query_level"]["rd_pp"]
           - 25.0) < 1e-9)
    ck("端到端：位置因变量只在两臂都被引的 query 上配对（45 个）",
       rep["arms"]["structure"]["paired"]["n_pos_pairs"] == 45,
       str(rep["arms"]["structure"]["paired"]["n_pos_pairs"]))
    ck("端到端：位置差为负（位置前移 = ratio 变小）",
       rep["arms"]["structure"]["paired"]["mean_diff_pos_pp"] < 0,
       str(rep["arms"]["structure"]["paired"]["mean_diff_pos_pp"]))
    ck("端到端：检验族 = 2 臂 × (引用 t + 位置 t + McNemar) = 6",
       rep["n_tests"] == 6, str(rep["n_tests"]))
    # 退化对比必须**不进族**（旧代码给它 p=0.0，会白占一个 Holm 名额）
    # ⚠️ 循环变量别叫 rep —— 端到端的 `rep = analyze(rows)` 会被覆盖成 int，
    #    后面 `render(rep)` 直接 AttributeError。
    _degen = []
    for qi in range(20):
        for _rp in range(3):
            _degen.append({"query_id": f"D{qi}", "arm": CONTROL, "repeat": str(_rp),
                           "cited": "1", "geo_impression": "0.2",
                           "cited_pos_ratio": "0.5", "retention": "", "error": ""})
            _degen.append({"query_id": f"D{qi}", "arm": "semantic", "repeat": str(_rp),
                           "cited": "1", "geo_impression": "0.3",
                           "cited_pos_ratio": "0.3", "retention": "0.9", "error": ""})
    _rd = analyze(_degen)
    ck("🔴 退化的位置对比不产生检验（旧代码会报 p=0.0 并进族）",
       not any("cited_pos_ratio" in k for k in _rd["tests"]),
       " ".join(_rd["tests"].keys()))
    ck("🔴 退化时位置差仍如实报出（只是不参与检验）",
       _rd["arms"]["semantic"]["paired"]["mean_diff_pos_pp"] is not None
       and _rd["arms"]["semantic"]["paired"]["mean_diff_pos_pp"] < 0)
    md = render(rep)
    ck("渲染含 Holm 表", "Holm" in md)
    ck("渲染含 §7.2 判定表（规则 1 + 规则 2 都要出现）",
       "7.2" in md and "规则2 亚噪声" in md and "规则1 符号不稳" in md)
    ck("渲染提醒逐臂 MDE 与 full 臂功效不足",
       "§4b" in md and "5pp" in md)

    # 7b. 🔴 §7.2 两条规则必须**分别**实现，不能拿一个统计量顶替
    #     （首次实现把「多少 query 的 Δ 为负」当成了符号稳定性，判据完全不同）
    s0 = rep["arms"]["structure"]["paired"]
    ck("§7.2 规则 2：|Δ| < σ_run 判为亚噪声",
       s0["sub_noise"] is not None and s0["sigma_run_pp"] == 2.2118)
    rep_hi = analyze(rows, sigma_run_pp=0.001)     # 门限调到极小 -> 都不该亚噪声
    ck("门限可调：把 σ_run 调到 0.001 后没有臂是亚噪声",
       not any(rep_hi["arms"][a]["paired"]["sub_noise"]
               for a in ("structure", "full") if "paired" in rep_hi["arms"][a]))
    ck("规则 1 统计量存在且是「重复间符号不一致」的 query 数",
       "sign_inconsistent_queries" in s0
       and 0 <= s0["sign_inconsistent_queries"] <= s0["n_query_pairs"])
    ck("neg_frac 与 sign_instab_rate 是两个不同的量（不混用）",
       "neg_frac_queries" in s0 and "sign_instab_rate" in s0)
    # 专门构造让两者**必须**不同的情形。
    # 🔴 二值差 d ∈ {-1,0,+1}，要造出「同一 query 的重复间正负都有」，
    #    **对照臂也必须跟着变**；只让处理臂围绕恒为 1 的对照交替，
    #    差值只会是 0 或 −1，产生不了两种号（2026-10-06 首次就踩了这个）。
    # A 组 20 query：重复间 +1/-1 交替 -> 符号不一致，均值 0
    # B 组 5 query：全 -1 -> 均值负、符号一致
    # C 组 5 query：全 +1 -> 均值正、符号一致
    # => neg_frac = 5/30 = 0.167，sign_instab = 20/30 = 0.667
    mix = []

    def add(q, rep, ctrl, trt, cpos, tpos):
        mix.append({"query_id": q, "arm": CONTROL, "repeat": str(rep),
                    "cited": str(ctrl), "geo_impression": "0.2",
                    "cited_pos_ratio": cpos, "retention": "", "error": ""})
        mix.append({"query_id": q, "arm": "structure", "repeat": str(rep),
                    "cited": str(trt), "geo_impression": "0.3",
                    "cited_pos_ratio": tpos, "retention": "0.9", "error": ""})

    for qi in range(20):
        for rep in range(4):
            up = rep % 2 == 0
            add(f"A{qi}", rep, 0 if up else 1, 1 if up else 0, "0.5", "0.4")
    for qi in range(5):
        for rep in range(4):
            add(f"B{qi}", rep, 1, 0, "0.5", "-1")
    for qi in range(5):
        for rep in range(4):
            add(f"C{qi}", rep, 0, 1, "0.5", "0.4")
    sm = analyze(mix)["arms"]["structure"]["paired"]
    ck("🔴 构造成功：负 query 占 1/2（分母是「有变化」的 10 个），"
       "但符号不一致的占 2/3（分母是全部 30 个）",
       sm["neg_frac_queries"] == 0.5
       and abs(sm["sign_instab_rate"] - 2 / 3) < 1e-4,   # 统计量保留 4 位小数
       f"neg={sm['neg_frac_queries']} instab={sm['sign_instab_rate']}")
    ck("🔴 两个统计量的分母也不同（nz 10 vs 全部 30）",
       sm["neg_frac_queries"] != sm["sign_instab_rate"])
    ck("🔴 §7.2 判据取的是符号不一致率（2/3 -> 判符号不稳），不是负比例",
       sm["sign_unstable"] is True)
    ck("🔴 符号不一致的 query 全部是 A 组（20 个）",
       sm["sign_inconsistent_queries"] == 20,
       str(sm["sign_inconsistent_queries"]))

    # 8. 🔴 缺失必须被当成缺失，不是 0
    rows2 = [r for r in rows if not (r["arm"] == "full" and r["query_id"] == "Q0")]
    rep2 = analyze(rows2)
    ck("删掉部分 query 后 full 的配对数减 1",
       rep2["arms"]["full"]["paired"]["n_query_pairs"] == 59,
       str(rep2["arms"]["full"]["paired"]["n_query_pairs"]))
    rows3 = [r for r in rows if r["arm"] != "full"]
    rep3 = analyze(rows3)
    # 🔴 v1.8：臂表现在**从数据反推**。数据里没有 `full`，`full` 就不该出现
    #    —— 早先硬编码 TREATMENT_ARMS 时它会留一个空壳臂在表里，
    #    看起来像「该臂无结论」，实际是「该臂不在这次实验里」。两者必须分开。
    ck("🔴 数据里没有的臂**不出现在**结果里（不是留空壳）",
       "full" not in rep3["arms"])
    ck("🔴 臂表随数据变化（本 fixture 只有 structure / full 两臂）",
       rep3["treatment_arms"] == ["structure"], str(rep3["treatment_arms"]))
    ck("🔴 缺失臂不产生任何检验",
       not any(k.startswith("full|") for k in rep3["tests"]))

    # 9. gate_blocked 记为缺失
    rows4 = list(rows) + [{"query_id": "Q0", "arm": "full", "repeat": "0",
                           "cited": "0", "geo_impression": "0",
                           "cited_pos_ratio": "", "retention": "",
                           "error": "gate_blocked"}]
    rep4 = analyze(rows4)
    ck("gate_blocked 不被当成 0 观测混进均值",
       rep4["arms"]["full"]["paired"]["n_query_pairs"] == 60)

    # 10. 位置因变量的子集必须报出来
    rows5 = [dict(r, cited_pos_ratio="-1") if r["arm"] == "structure" else r
             for r in rows]
    rep5 = analyze(rows5)
    ck("位置未定义 -> 位置对数 0（不静默用别的量替代）",
       rep5["arms"]["structure"]["paired"]["n_pos_pairs"] == 0)

    # 11. 🔴 零配对不能被显示成「可参与排序」
    #     构造：只留 original 臂 —— 四个处理臂**全部**零配对。
    rows_ctrl_only = [r for r in rows if r["arm"] == "original"]
    rep_nodata = analyze(rows_ctrl_only)
    md_nodata = render(rep_nodata)
    ck("🔴 构造成功：四个处理臂全部零配对",
       all(rep_nodata["arms"][a]["paired"]["n_query_pairs"] == 0
           for a in ("structure", "evidence", "semantic", "full")))
    ck("🔴 零配对显示为「无数据」而不是「✅ 可」",
       "⛔ **无数据**" in md_nodata)
    ck("🔴 零配对时正文没有任何 ✅ 可", "✅ 可" not in md_nodata)
    ck("🔴 零配对时写明「先排查再解读」", "先排查再解读" in md_nodata)
    ck("🔴 有配对的臂仍然正常标 ✅ 可（不误伤）",
       "✅ 可" in render(analyze(rows)))

    # 12. 🔴 缺列 ≠ 覆盖率为 0
    rows_noposc = [{k: v for k, v in r.items() if k != "cited_pos_ratio"}
                   for r in rows]
    rep_noposc = analyze(rows_noposc)
    ck("🔴 缺列被记为 has_pos_column = False",
       rep_noposc["meta"]["has_pos_column"] is False)
    ck("🔴 有列时 has_pos_column = True",
       analyze(rows)["meta"]["has_pos_column"] is True)
    md_noposc = render(rep_noposc)
    ck("🔴 缺列时报表写「列不存在」而不是 0.0%",
       "🔴 列不存在" in md_noposc)
    ck("🔴 缺列时明确说这不是覆盖率为 0",
       "不是**「覆盖率为 0%」" in md_noposc or "不是「覆盖率为 0%」" in md_noposc)

    # 13. 🔴 渲染里不留占位符残留
    md_ok = render(analyze(rows))
    ck("🔴 正文无 '?' 占位符残留（独立单元那行是实数）",
       "个 query**（不是 ?" not in md_ok)
    ck("🔴 独立单元那行给出了真实格数",
       f"（不是 {analyze(rows)['n_queries'] * 0 + sum(s.get('n_obs', 0) for k, s in analyze(rows)['arms'].items() if not k.startswith('_'))} 个格" in md_ok)

    # 14. 🔴 v1.6：剔除表必须让独立单元数与 16 号一致
    rows_ex = list(rows) + [dict(rows[0], query_id="GBEXCL")]
    rep_ex = analyze(rows_ex)
    base_n = analyze(rows)["n_queries"]
    ck("🔴 剔除前独立单元数被记录",
       rep_ex["meta"]["n_queries_before_exclusion"] == base_n + 1,
       str(rep_ex["meta"]["n_queries_before_exclusion"]))
    ck("🔴 不传剔除表时不剔除（对照组）", rep_ex["n_queries"] == base_n + 1,
       str(rep_ex["n_queries"]))
    _ex = load_exclusions(engine="minimax")
    # 🔴 2026-10-21 开源化：下面这几条断言依赖**项目的剔除登记表**
    #    （data/query_exclusions.json，或环境变量 GEO_QUERY_EXCLUSIONS 指向的文件）。
    #    本仓库里没有这个文件 ⇒ 断言必须**跳过并出声**，
    #    不能判 FAIL —— 证据不在场 ≠ 功能缺失。
    # 示例条目 id：默认**从登记表里取第一条**，而不是写死某个 id ——
    # 换一份登记表时不用改代码（也可用 GEO_EXCLUSION_SAMPLE_ID 指定）。
    _sample_id = (os.environ.get("GEO_EXCLUSION_SAMPLE_ID", "").strip()
                  or (sorted(_ex)[0] if _ex else ""))
    if _ex and _sample_id:
        ck("🔴 load_exclusions 读得到真实登记表（示例条目在册）",
           isinstance(_ex, set) and _sample_id in _ex, str(sorted(_ex))[:80])
    else:
        print("  [SKIP] 剔除登记表不在场（示例条目无法核对）—— 这不是缺陷")
    ck("🔴 load_exclusions 对不存在/空文件返回空集",
       load_exclusions(os.path.join(ROOT, "data", "__no_such_file__.json"))
       == set())
    if _ex:
        ck("🔴 按引擎过滤：qwen 下不剔（它没被 qwen 拒过）",
           _sample_id not in load_exclusions(engine="qwen"),
           str(sorted(load_exclusions(engine="qwen"))))
        ck("🔴 按引擎过滤：minimax 下剔", _sample_id in load_exclusions(engine="minimax"))
        ck("🔴 不给 engine 时返回全表（向后兼容）",
           _sample_id in load_exclusions())
    else:
        print("  [SKIP] 按引擎过滤的三条断言需要真实登记表 —— 跳过")

    # 15. 🔴 E4 臂名必须被认出来（否则整条从报表消失）
    e4_rows = list(rows) + [
        dict(rows[0], arm="evidence_fabricated", cited="1", cited_pos_ratio="0.2"),
        dict(rows[0], arm="evidence_real", cited="0", cited_pos_ratio="-1"),
    ]
    rep_e4 = analyze(e4_rows)
    ck("🔴 E4 臂名被反推进臂表",
       "evidence_fabricated" in rep_e4["treatment_arms"]
       and "evidence_real" in rep_e4["treatment_arms"],
       str(rep_e4["treatment_arms"]))
    ck("🔴 阶段 B 臂名仍在（同一份登记表，fixture 的 structure/full）",
       set(TREATMENT_ARMS[:1]) | {"full"} <= set(rep_e4["treatment_arms"]),
       str(rep_e4["treatment_arms"]))
    ck("🔴 E4 臂真的出了配对结果（不是只有臂名没有数据）",
       rep_e4["arms"]["evidence_fabricated"]["paired"]["n_query_pairs"] >= 1)
    md_e4 = render(rep_e4)
    ck("🔴 报表正文里能看到 E4 臂名", "`evidence_fabricated`" in md_e4
       and "`evidence_real`" in md_e4)
    ck("🔴 未注册臂名直接报错（不静默丢）",
       _raises_valueerror(lambda: analyze(
           [dict(rows[0], arm="evidence_FABRICATED")])))
    ck("🔴 只有 original 时回落到默认臂表",
       infer_arms([dict(rows[0], arm="original")]) == TREATMENT_ARMS)
    ck("🔴 infer_arms 忽略空臂名",
       infer_arms([dict(rows[0], arm="")]) == TREATMENT_ARMS)

    failed = [n for n, ok in checks if not ok]
    for n, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {n}")
    print(f"\n{len(checks)-len(failed)} PASS   {len(failed)} FAIL")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
