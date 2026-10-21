#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
20_rewriter_noise.py -- 改写器层面的噪声：两次独立改写之间的一致率

为什么需要它
------------
本文（方法章 §4.1）的跨引擎复现声称「**固定处理、换测量引擎**」。
但 `15_stage_b_run.py` 的改写是**每次运行各自独立生成**的
（`rewrite_cache` 只存在于内存，不落盘复用）。改写器 `qwen-plus`
即使 `temperature=0` 也**不是逐位确定**的 —— qwen-plus 未承诺 seed 语义。

于是「换引擎」这个操作在因果上并不是干净的：
**引擎变了，改写文本这个抽样也变了。** 两者混在一起。

这个脚本直接测量这件事：把两次运行的**门判定**逐格对齐，
看同一 `(arm, query)` 在两次独立改写下是否得到同一个结论。
门判定（F1–F6）是一个**离散的、可复现的**函数，所以它的不一致率
就是改写器层面噪声的一个**下界估计**（真噪声 ≤ 门判定不一致率，
因为两次文本可能不同但门恰好同判）。

🔴 **这个量的意义**：它是本文所有效应量的**另一层底噪**。
如果处理臂的文本本身就在漂，那么任何小于该漂移的 Δ 都不能归因于处理。

用法
----
    python examples/20_rewriter_noise.py \\
        --a path/to/fill_checkpoint.jsonl \\
        --b path/to/fill_checkpoint.jsonl
    python examples/20_rewriter_noise.py --selftest

退出码：0 = 跑通；2 = 参数/文件错误。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_A = os.path.join(ROOT, "outputs", "15_stage_b", "fill_checkpoint.jsonl")
DEFAULT_B = os.path.join(ROOT, "outputs", "15_stage_b_qwen", "fill_checkpoint.jsonl")


def load_cells(path: str) -> Dict[Tuple[str, str], dict]:
    """读 checkpoint，只取处理臂（original 没有改写，不参与）。

    同一格多次出现时**保留最后一条** —— checkpoint 是追加写的，
    断点续跑会重写已完成的格，文件里因此可能有重复键。
    """
    out: Dict[Tuple[str, str], dict] = {}
    with open(path, "r", encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("arm") == "original":
                continue
            out[(r.get("arm", ""), r.get("query_id", ""))] = r
    return out


def codes(r: dict) -> List[str]:
    c = r.get("gate_codes") or []
    return sorted(str(x) for x in c)


def compare(A: Dict[Tuple[str, str], dict],
            B: Dict[Tuple[str, str], dict],
            arms: Optional[List[str]] = None) -> dict:
    """逐臂返回 (n, 门判定一致, 门码完全一致) 以及不一致的样例。"""
    keys = set(A) & set(B)
    if arms:
        keys = {k for k in keys if k[0] in arms}
    per: Dict[str, dict] = {}
    for (arm, _q) in sorted(keys):
        per.setdefault(arm, {"n": 0, "gate_same": 0, "codes_same": 0,
                             "discord": []})
    for k in sorted(keys):
        a, b = A[k], B[k]
        s = per[k[0]]
        s["n"] += 1
        gp_same = bool(a.get("gate_pass")) == bool(b.get("gate_pass"))
        cd_same = codes(a) == codes(b)
        s["gate_same"] += int(gp_same)
        s["codes_same"] += int(cd_same)
        if not gp_same and len(s["discord"]) < 5:
            s["discord"].append({
                "query_id": k[1],
                "a_gate_pass": bool(a.get("gate_pass")),
                "a_codes": codes(a),
                "b_gate_pass": bool(b.get("gate_pass")),
                "b_codes": codes(b),
            })
    for s in per.values():
        n = s["n"] or 1
        s["gate_agreement"] = s["gate_same"] / n
        s["code_agreement"] = s["codes_same"] / n
        s["gate_disagreement"] = 1.0 - s["gate_agreement"]

    # 🔴🔴 **必须把「query 集合不同」和「改写漂移」分开报。**
    #    2026-10-06 实测踩过：两轮门通过数不同（evidence 66/67、full 28/27），
    #    一度被当成「改写器不逐位确定」的证据。逐 query 对齐后真相是
    #    **两轮的 query 集合本身就差一条**（剔除登记表带 engine 标签，
    #    每条登记都带引擎标签，跨引擎要换对应的条目），
    #    而 **144 条共享 query 上门判定零分歧**。
    #    不显式报这个差异，读者会把集合差异误读成改写噪声。
    qa = {q for (_a, q) in A}
    qb = {q for (_a, q) in B}
    only_a = sorted(qa - qb)
    only_b = sorted(qb - qa)
    return {"per_arm": per, "n_cells": len(keys),
            "n_queries_a": len(qa), "n_queries_b": len(qb),
            "n_queries_shared": len(qa & qb),
            "only_in_a": only_a, "only_in_b": only_b,
            "query_set_identical": not only_a and not only_b}


def render(rep: dict) -> str:
    out = []
    out.append("=" * 78)
    out.append("改写器层面噪声：两次独立改写运行的逐格一致率")
    out.append("=" * 78)
    out.append(f"query 集合 A = {rep['n_queries_a']} 条，B = {rep['n_queries_b']} 条，"
               f"共享 {rep['n_queries_shared']} 条")
    if rep["query_set_identical"]:
        out.append("query 集合：**两轮完全相同**")
    else:
        out.append(f"🔴 query 集合**不相同** —— 仅 A 有 {rep['only_in_a']}，"
                   f"仅 B 有 {rep['only_in_b']}")
        out.append("   （通常是剔除登记表带 engine 标签造成的：**这不是改写漂移**。"
                   "门通过数若不同，先看这一行再下结论。）")
    out.append("-" * 78)
    out.append(f"{'臂':<12}{'格数':>7}{'门判定一致':>14}{'门码完全一致':>16}{'判定分歧':>12}")
    for arm, s in sorted(rep["per_arm"].items()):
        out.append(f"{arm:<12}{s['n']:>7}"
                   f"{s['gate_same']:>9} ({s['gate_agreement']:>5.1%})"
                   f"{s['codes_same']:>11} ({s['code_agreement']:>5.1%})"
                   f"{s['gate_disagreement']:>12.1%}")
    tot = sum(s["n"] for s in rep["per_arm"].values())
    gs = sum(s["gate_same"] for s in rep["per_arm"].values())
    cs = sum(s["codes_same"] for s in rep["per_arm"].values())
    if tot:
        out.append(f"{'合计':<12}{tot:>7}{gs:>9} ({gs/tot:>5.1%})"
                   f"{cs:>11} ({cs/tot:>5.1%}){1-gs/tot:>12.1%}")
    out.append("-" * 78)
    if rep["query_set_identical"]:
        out.append("判读：两轮 query 集合相同，因此这里的分歧率**就是**改写器层面噪声的")
    else:
        out.append("判读：🔴 两轮 query 集合不同，上面的一致率是在**共享 query** 上算的，")
        out.append("      它度量的是「改写器在两次独立抽样下的门判定分歧」= 噪声下界。")
    out.append("      门判定分歧率是改写器层面噪声的**下界**（真噪声可能更小：")
    out.append("      两次文本不同但门恰好同判）。")
    out.append("      ⚠️ 它**不**等于效应量下界；它只是「文本是否稳定」的体检。")
    return "\n".join(out)


def selftest() -> int:
    ok = fail = 0

    def ck(name, cond, extra=""):
        nonlocal ok, fail
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f"  {extra}" if (extra and not cond) else ""))
        if cond:
            ok += 1
        else:
            fail += 1

    print("== 20_rewriter_noise selftest ==")

    # 1. original 臂必须被排除 —— 它没有改写，比较它没有意义
    rows = [
        {"arm": "original", "query_id": "q1", "gate_pass": True, "gate_codes": []},
        {"arm": "full", "query_id": "q1", "gate_pass": True, "gate_codes": []},
        {"arm": "full", "query_id": "q2", "gate_pass": False, "gate_codes": ["F5_RETENTION"]},
    ]
    d = {"rows": rows}

    def _load_from(rows_in):  # 用内存行替代表盘读取
        out = {}
        for r in rows_in:
            if r.get("arm") == "original":
                continue
            out[(r.get("arm", ""), r.get("query_id", ""))] = r
        return out

    cells = _load_from(d["rows"])
    ck("original 臂被排除", all(k[0] != "original" for k in cells))
    ck("处理臂保留", ("full", "q1") in cells and ("full", "q2") in cells)

    # 2. 重复键保留最后一条（断点续跑会重写已完成的格）
    dup = [
        {"arm": "full", "query_id": "q1", "gate_pass": True, "gate_codes": []},
        {"arm": "full", "query_id": "q1", "gate_pass": False, "gate_codes": ["F1_NEW_NUMBER"]},
    ]
    ck("重复键保留最后一条", _load_from(dup)[("full", "q1")]["gate_pass"] is False)

    # 3. 完全一致 → 100%
    same = _load_from(dup)
    r1 = compare(same, same)
    ck("自比一致率 = 100%", all(s["gate_agreement"] == 1.0
                                for s in r1["per_arm"].values()))
    ck("自比分歧率 = 0", all(s["gate_disagreement"] == 0.0
                            for s in r1["per_arm"].values()))

    # 4. 全部分歧 → 0%
    A = _load_from([{"arm": "full", "query_id": "q1", "gate_pass": True, "gate_codes": []},
                    {"arm": "full", "query_id": "q2", "gate_pass": True, "gate_codes": []}])
    B = _load_from([{"arm": "full", "query_id": "q1", "gate_pass": False, "gate_codes": ["F5"]},
                    {"arm": "full", "query_id": "q2", "gate_pass": False, "gate_codes": ["F2"]}])
    r2 = compare(A, B)
    ck("全部分歧 → 一致率 0", r2["per_arm"]["full"]["gate_agreement"] == 0.0)
    ck("全部分歧 → 分歧率 1", r2["per_arm"]["full"]["gate_disagreement"] == 1.0)

    # 5. 只统计交集，不把对方没有的格算成分歧
    C = _load_from([{"arm": "full", "query_id": "q1", "gate_pass": True, "gate_codes": []}])
    r3 = compare(A, C)
    ck("只统计交集（不把缺席算分歧）", r3["n_cells"] == 1
       and r3["per_arm"]["full"]["gate_agreement"] == 1.0)

    # 6. 门码顺序无关（同一组码不同顺序 = 一致）
    X = _load_from([{"arm": "full", "query_id": "q1", "gate_pass": False,
                     "gate_codes": ["F1_NEW_NUMBER", "F5_RETENTION"]}])
    Y = _load_from([{"arm": "full", "query_id": "q1", "gate_pass": False,
                     "gate_codes": ["F5_RETENTION", "F1_NEW_NUMBER"]}])
    r4 = compare(X, Y)
    ck("门码顺序无关（code_agreement = 1）",
       r4["per_arm"]["full"]["code_agreement"] == 1.0)

    # 7. 门码不同但门判定相同 → 门判定一致、门码不一致（两个口径必须分开）
    r5 = compare(X, _load_from([{"arm": "full", "query_id": "q1", "gate_pass": False,
                                 "gate_codes": ["F1_NEW_NUMBER"]}]))
    s5 = r5["per_arm"]["full"]
    ck("两口径可分离：门判定一致", s5["gate_agreement"] == 1.0)
    ck("两口径可分离：门码不一致", s5["code_agreement"] == 0.0)

    # 8. 分歧样例被记录且带两侧门码
    r6 = compare(A, B)
    ck("分歧样例被记录", len(r6["per_arm"]["full"]["discord"]) == 2)
    ck("分歧样例带两侧门码",
       all("a_codes" in x and "b_codes" in x
           for x in r6["per_arm"]["full"]["discord"]))

    # 9. --arms 过滤生效
    AA = _load_from([{"arm": "full", "query_id": "q1", "gate_pass": True, "gate_codes": []},
                     {"arm": "evidence", "query_id": "q1", "gate_pass": True, "gate_codes": []}])
    r7 = compare(AA, AA, arms=["full"])
    ck("--arms 过滤生效", list(r7["per_arm"].keys()) == ["full"])

    # 10. codes() 对 None / 字符串输入都稳
    ck("codes(None) = []", codes({"gate_codes": None}) == [])
    ck("codes 字符串化", codes({"gate_codes": [1, 2]}) == ["1", "2"])

    # 11. 🔴🔴 query 集合差异必须与「改写分歧」分开报
    #      真实两轮的教训：门通过数不同（evidence 66/67、full 28/27），
    #      一度被当成「改写器不逐位确定」的证据。逐 query 对齐后真相是
    #      **query 集合本身就差一条**，而 144 条共享 query 上门判定**零分歧**。
    #      不显式报这个差异，读者会把集合差异误读成改写噪声。
    A2 = _load_from([{"arm": "full", "query_id": f"Q{i}", "gate_pass": True,
                      "gate_codes": []} for i in range(3)])
    B2 = _load_from([{"arm": "full", "query_id": f"Q{i}", "gate_pass": True,
                      "gate_codes": []} for i in (1, 2, 9)])
    r8 = compare(A2, B2)
    ck("🔴 query 集合差异被单独报出（不混进分歧率）",
       r8["only_in_a"] == ["Q0"] and r8["only_in_b"] == ["Q9"],
       f"{r8['only_in_a']} / {r8['only_in_b']}")
    ck("🔴 query_set_identical = False", r8["query_set_identical"] is False)
    ck("🔴 共享部分 2 条、零分歧 -> 分歧率仍是 0",
       r8["n_queries_shared"] == 2
       and r8["per_arm"]["full"]["gate_agreement"] == 1.0)
    ck("🔴 同一集合时 query_set_identical = True",
       compare(A2, A2)["query_set_identical"] is True)
    ck("🔴 render 明写 query 集合不相同", "query 集合**不相同**" in render(r8))
    ck("🔴 render 提醒这不是改写漂移", "不是改写漂移" in render(r8))
    ck("🔴 集合相同时不写那句提醒",
       "query 集合：**两轮完全相同**" in render(compare(A2, A2)))

    # 12. render 不炸
    try:
        s = render(r6)
        ck("render 产出非空表", "合计" in s and "改写器" in s)
    except Exception as e:  # noqa: BLE001
        ck("render 产出非空表", False, str(e))

    print(f"\n{ok} PASS   {fail} FAIL")
    return 1 if fail else 0


def main(argv=None) -> int:
    try:
        _sys = sys.stdout
        _sys.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    p = argparse.ArgumentParser(
        prog="20_rewriter_noise.py",
        description="两次独立改写运行之间的逐格一致率（改写器层面噪声）")
    p.add_argument("--a", default=DEFAULT_A, help="第一次运行的 checkpoint")
    p.add_argument("--b", default=DEFAULT_B, help="第二次运行的 checkpoint")
    p.add_argument("--arms", default="", help="逗号分隔，只看这些臂")
    p.add_argument("--json", default="", help="另存 JSON 的路径")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args(argv)
    if a.selftest:
        return selftest()
    for f in (a.a, a.b):
        if not os.path.exists(f):
            print(f"ERROR: 找不到 {f}", file=sys.stderr)
            return 2
    arms = [x for x in a.arms.split(",") if x] or None
    rep = compare(load_cells(a.a), load_cells(a.b), arms)
    print(render(rep))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, ensure_ascii=False, indent=2)
        print(f"-> {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())