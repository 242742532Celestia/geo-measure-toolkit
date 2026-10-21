#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""17_extend_geobench_cache.py —— 扩展 GEO-bench 本地缓存到 N 条。

为什么需要
----------
阶段 A'（MiniMax-M3）在 150 条缓存上跑完后，**合格 query 只剩 132 条**：
- 17 条目标文档 < 500 字符（v1.3 的 query 级资格门槛）
- 1 条被内容审核确定性拒答（见剔除登记表）

而 §11 局限 5 的**位置规则**要求每位置 ≥ 30 个 query × 5 位置 = **Q ≥ 150**。
这是硬约束，且与功效无关（功效只要 41 条）。

两条出路
--------
(a) 调低 `MIN_SOURCE_CHARS` —— 那是**实质性**判断（文档够不够承载改写对比），
    为了满足一个统计约束去动它，是拿设计换数字。
(b) **扩大候选池** —— GEO-bench 的 test.jsonl 实测 **83.5 MB（约 950 条）**，
    本地只缓存了前 150 条。把池子扩到 1000 条、**沿用完全相同的门槛重筛**，
    两条规则一个字都不用改。

选 (b)。本脚本只做「拉更多样本进缓存」，不含任何筛选逻辑 ——
筛选仍然全部由 `15_stage_b_run.py` 的 `eligible()` + 剔除登记表负责，
所以扩池前后用的是同一把尺子。

🔴 扩池是**预注册修订**（v1.4）：候选池从 150 扩到 1000，门槛不变。
   必须在 `your preregistration log` 留时间戳与理由。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load08():
    p = os.path.join(ROOT, "examples", "08_reproduce_geo_bench.py")
    spec = importlib.util.spec_from_file_location("repro08", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["repro08"] = mod
    spec.loader.exec_module(mod)
    return mod


def main(argv=None) -> int:
    # Windows 控制台默认 GBK，中文/emoji 会抛 UnicodeEncodeError -> 切 UTF-8
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    p = argparse.ArgumentParser(
        prog="17_extend_geobench_cache.py",
        description="扩展 GEO-bench 缓存到 N 条（不做任何筛选）")
    p.add_argument("--target", type=int, default=1000,
                   help="缓存至少要有多少条（默认 1000）")
    p.add_argument("--cache-dir", default=os.path.join(ROOT, "data", "geo-bench"))
    p.add_argument("--geobench-url", default="")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args(argv)
    if a.selftest:
        return selftest()

    R = load08()
    cd = Path(a.cache_dir)
    cache = cd / "test.head.jsonl"
    before = 0
    if cache.exists():
        before = sum(1 for l in cache.read_text(encoding="utf-8").splitlines()
                     if l.strip())
    print(f"[+] 扩池前缓存：{before} 条（{cache}）")
    if before >= a.target:
        print(f"[=] 已达目标 {a.target}，无需下载。")
        return 0
    print(f"[+] 流式拉取到 {a.target} 条（83.5 MB 全量，耐心等）…")
    recs, cdir, url = R.load_geobench(a.target, cd, a.geobench_url or None,
                                      max_chars=4000)
    after = sum(1 for l in cache.read_text(encoding="utf-8").splitlines()
                if l.strip())
    print(f"[+] 扩池后缓存：{after} 条（{cache.stat().st_size/1024/1024:.1f} MB）")

    # ---- 顺带报一下「扩池到底能不能解决 Q≥150」：只统计，不筛选 ----
    lens = []
    for r in recs:
        s = r.get("sources") or []
        i = int(r.get("sugg_idx", 0))
        if 0 <= i < len(s):
            lens.append(len(s[i].get("cleaned_text") or ""))
    lens.sort()
    n_ge = sum(1 for x in lens if x >= 500)
    print(f"[i] 目标文档长度：中位 {lens[len(lens)//2] if lens else 0}，"
          f">=500 字符的样本 {n_ge}/{len(lens)} = "
          f"{n_ge/len(lens):.1%}" if lens else "[i] 空")
    print(f"[+] 结论：{len(lens)} 条候选池里有 {n_ge} 条满足现行门槛，"
          f"{'足以' if n_ge >= 150 else '仍不足以'}满足 Q>=150 的位置规则。")
    meta = cd / "test.head.meta.json"
    meta.write_text(json.dumps({
        "url": url, "count": after,
        "bytes": cache.stat().st_size,
        "extended_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "note": "候选池扩容；筛选规则（MIN_SOURCE_CHARS=500 + 剔除登记表）未改",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[+] meta -> {meta}")
    return 0


def selftest() -> int:
    checks: list[tuple[str, bool]] = []

    def ck(name, cond, extra=""):
        checks.append((name, bool(cond)))

    R = load08()
    ck("能加载 08", hasattr(R, "load_geobench"))
    import inspect as _i
    ck("缓存路径约定 = test.head.jsonl",
       "test.head.jsonl" in _i.getsource(R.load_geobench))
    src = open(os.path.join(ROOT, "examples", "17_extend_geobench_cache.py"),
               encoding="utf-8").read()
    # 🔴 关键纪律：扩池脚本**不得**自带任何门槛、也不得复制 15 的筛选规则。
    #    否则「扩池」会偷偷变成「换筛选规则」，而那是要走预注册修订的规则改动。
    import ast as _ast
    tree = _ast.parse(src)
    imported = set()
    for n in _ast.walk(tree):
        if isinstance(n, _ast.Import):
            imported.update(a.name for a in n.names)
        elif isinstance(n, _ast.ImportFrom) and n.module:
            imported.add(n.module)
    ck("扩池脚本不 import 15（无法复制/偷用它的门槛）",
       not any("15_stage_b" in i for i in imported), str(sorted(imported)))
    # `recs` 只能来自 load_geobench 的一次解包 —— 任何重新赋值/过滤都是「偷偷筛选」
    recs_assigns = []
    for n in _ast.walk(tree):
        tgts = []
        if isinstance(n, _ast.Assign):
            tgts = list(n.targets)
        elif isinstance(n, _ast.AugAssign):
            tgts = [n.target]
        elif isinstance(n, _ast.For):
            tgts = [n.target]
        for t in tgts:
            if any(isinstance(x, _ast.Name) and x.id == "recs"
                   for x in _ast.walk(t)):
                recs_assigns.append(type(n).__name__)
    ck("recs 只被 load_geobench 赋值一次（无任何过滤/重排）",
       recs_assigns == ["Assign"], str(recs_assigns))
    ck("拉取的是完整前缀（未按 limit 之外的量裁剪）",
       any(isinstance(n, _ast.Call)
           and isinstance(n.func, _ast.Attribute) and n.func.attr == "load_geobench"
           and any(isinstance(a, _ast.Attribute) and a.attr == "target"
                   for a in n.args)
           for n in _ast.walk(tree)))
    # 500 阈值只允许喂给「计数」，且计数只用于打印
    # 🔴 只看 main()，**排除 selftest 自身**（自检里那句 "500" in unparse
    #    也是个 Compare，会把自己判成违规）。
    main_fn = next((f for f in tree.body
                    if isinstance(f, _ast.FunctionDef) and f.name == "main"), None)
    mtree = main_fn if main_fn is not None else tree
    thr_stmts = [st for st in _ast.walk(mtree)
                 if isinstance(st, _ast.stmt)
                 and any(isinstance(n, _ast.Compare) and "500" in _ast.unparse(n)
                         for n in _ast.walk(st))]
    in_count = bool(thr_stmts) and all(
        any(isinstance(a, _ast.Call)
            and isinstance(a.func, _ast.Name) and a.func.id == "sum"
            for a in _ast.walk(st))
        for st in thr_stmts)
    ck("500 阈值只出现在 sum(...) 计数里，从不用于过滤 recs",
       in_count, str([_ast.unparse(s)[:60] for s in thr_stmts]))
    ck("扩池脚本声明了预注册修订", "预注册修订" in src)
    ck("目标默认 1000（远超 150，足以满足 Q>=150）",
       "--target" in src and "default=1000" in src)
    failed = [n for n, ok in checks if not ok]
    for n, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {n}")
    print(f"\n{len(checks)-len(failed)} PASS   {len(failed)} FAIL")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
