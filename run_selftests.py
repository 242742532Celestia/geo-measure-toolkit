#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_selftests.py -- 工具包自证入口

一条命令跑完包内**所有可独立运行**的自检，并核对每一条的**实测项数**
与 `MANIFEST.json` 登记的是否一致。

    python run_selftests.py            # 全跑
    python run_selftests.py --list     # 只看清单

🔴 三条纪律（与本项目一致）
--------------------------------
1. **结项行格式固定**：`N PASS   M FAIL`。核对器靠这个正则判断「工具是否跑完了」
   —— 没有结项行会被判成「崩在自检中途」，处置方向正好相反。
   ⚠️ 本仓存在**四种**结项格式，必须都认（否则会把「跑完了」误判成「崩了」）：
     A. `N PASS   M FAIL`（多数工具）
     B. `all checks passed`（statpower / geo_stat_lint / tautology_audit …）
     C. `24/24` 单独一行（fetch_citation_pdfs）
     D. `■ …自洽` 收尾（info_gate_selfcontained_check）
2. **「需要项目数据」不等于「坏了」**：Tier B 工具依赖本项目的
   `outputs/` `data/` `prompts/` `docs/` `paper/`，本包内**没有**这些。
   它们被**跳过并出声**，**不记 PASS**、**也不记 FAIL** ——
   证据不在场与功能缺失不能混报。
3. **实测项数与 MANIFEST 不符 = FAIL**。MANIFEST 是打包时实跑生成的，
   不是手抄的；跑出不同数字说明**包内的东西和登记的不一致**，必须查。
   ⚠️ `expected` 记的是**仓库里**跑出的项数，不是包内的 ——
   包内少跑断言 = **静默失检**，必须报出来。
4. 🔴 **「前置缺失」不是「崩了」**（2026-10-21 加）：
   工具的前置条件在本机不成立时，它**主动声明**「前置缺失」并打出结项行
   `N PASS   M FAIL   前置缺失：<原因>`。
   这时候 runner 必须判 **SKIP**，**不能**判 FAIL ——
   判 FAIL 会给出「工具崩在自检中途，先修工具」这种**反方向的处置指引**，
   而工具其实行为完全正确。
   ⇒ 没有结项行才是「崩了」；有结项行但声明了前置缺失 = 跳过并出声。
5. 🔴 **「少跑 N 项」有两种解释，处置方向相反**（2026-10-21 加）：
   ① **静默失检** —— 工具没说为什么就少跑了 ⇒ FAIL，必须查；
   ② **声明过的缺席** —— 工具自己打了 `[SKIP]` 说明「包外参考不在场，
      这几条不适用」⇒ 跳过并出声，**不是缺陷**。
   例：`info_gate.py` 的自检在 `_geo_core` 不在场时是 49 项（在场是 53），
   差的 4 条是「与包外参考逐位相等」，工具明说「缺席不算缺陷」。
   ⇒ 只看「有没有 FAIL / 少没少跑」会把 ② 报成 ①，**方向反了**。

退出码：0 = 全过（有跳过不算失败）；1 = 有失败；2 = 环境问题（manifest 缺失等）。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

#: 与本项目核对器同一套计数口径（`tools/check_doc_consistency.py::run_selftest`）
#: ⚠️ 本仓存在**五种**结项格式，必须都认：
#:   A. `N PASS   M FAIL`（多数工具）
#:   B. `all checks passed`（statpower / geo_stat_lint / tautology_audit —— **不输出 N/M**）
#:   C. `24/24` 单独一行（fetch_citation_pdfs）
#:   D. `■ …自洽` 收尾（info_gate_selfcontained_check）
#:   E. `N PASS   M FAIL   前置缺失：…`（工具主动声明前置条件不成立 → 跳过）
#: 只认 A 会把 B/C/D/E 类工具判成「崩在自检中途」。
#: 另：部分工具把 PASS 打在**行首无缩进**处，`^\s+PASS\s` 匹配不到，
#:   故补一条 `^PASS\s`。
RE_PASS = re.compile(r"\[PASS\]")
RE_PASS_LINE = re.compile(r"^\s+PASS\s", re.M)
RE_PASS_COL0 = re.compile(r"^PASS\s", re.M)
RE_FAIL = re.compile(r"\[FAIL\]")
RE_FAIL_LINE = re.compile(r"^\s+FAIL\s", re.M)
RE_FAIL_COL0 = re.compile(r"^FAIL\s", re.M)
#: 工具**显式声明**「这条检查不适用/缺席不算缺陷」时打的标记。
#: 🔴 有了它，「少跑 N 项」才有第二种可能解释：
#:      ① 静默失检 —— 工具没说为什么就少跑了（危险，必须查）；
#:      ② 声明过的缺席 —— 工具明说「包外参考不在场，这几条不适用」（不是缺陷）。
#:    两者处置方向相反：**① 要修工具，② 要补环境**。
RE_SKIP = re.compile(r"\[SKIP\]")
RE_FINISHED = re.compile(r"\d+\s+PASS\s+\d+\s+FAIL")
RE_FINISHED_B = re.compile(r"all checks passed", re.I)
#: C. `24/24` 单独一行（fetch_citation_pdfs：PASS 打在行首，结项是分数比）
RE_FINISHED_C = re.compile(r"^\s*\d+\s*/\s*\d+\s*$", re.M)
#: D. `■ …自洽` 收尾（info_gate_selfcontained_check：自结项是带 ■ 的结论句）
RE_FINISHED_D = re.compile(r"^■.*自洽", re.M)
#: E. `N PASS   M FAIL   前置缺失：<原因>` —— 工具**主动声明**「前置条件不成立」
#:    （2026-10-21 加）。🔴 这是「跳过」，不是「崩了」：
#:    崩了 ⇒ 处置方向是修工具；前置缺失 ⇒ 处置方向是补环境/换机器。
#:    判错的代价：把一次正确的跳过报成 bug，还让人去修一个没坏的工具。
#:    ⚠️ 必须**结项行与标记同行**（`^…$` + re.M）——
#:       否则「FAIL 前置缺失：xxx」这种**失败行**会被误当成跳过声明。
RE_FINISHED_E = re.compile(r"^\s*\d+\s+PASS\s+\d+\s+FAIL\s*[:：]?\s*前置缺失", re.M)


def count_checks(out: str) -> tuple:
    """返回 (n_pass, n_fail, finished, skipped)。

    `finished` 的判据是**有没有打出结项行**，不是退出码 ——
    一条合法的 FAIL 不等于工具崩了，两者的处置方向正好相反。

    `skipped` 的判据是**结项行是否声明了「前置缺失」**：
    工具的前置条件在本机不成立时，它主动声明并退出 —— 那是**跳过**，
    不是崩了。🔴 把 skipped 误判成 crashed 会给出**反方向**的处置指引
    （让人去修一个行为完全正确的工具）。

    ⚠️ 返回值是 4 元组 —— 打包器 `_measure()` 也调这个函数，
       两边必须同时改，否则打包口径会漂移。
    """
    n_pass = (len(RE_PASS.findall(out)) + len(RE_PASS_LINE.findall(out))
              + len(RE_PASS_COL0.findall(out)))
    n_fail = (len(RE_FAIL.findall(out)) + len(RE_FAIL_LINE.findall(out))
              + len(RE_FAIL_COL0.findall(out)))
    skipped = bool(RE_FINISHED_E.search(out))
    finished = bool(RE_FINISHED.search(out) or RE_FINISHED_B.search(out)
                    or RE_FINISHED_C.search(out) or RE_FINISHED_D.search(out)
                    or RE_FINISHED_E.search(out))
    return n_pass, n_fail, finished, skipped


def run_one(entry: dict, timeout: int = 300) -> dict:
    """跑一个工具的自检，返回 (n_pass, n_fail, crashed, skipped, rc, tail)。"""
    cmd = [sys.executable, entry["path"]] + list(entry["selftest"]["args"])
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           encoding="utf-8", errors="replace", cwd=str(ROOT))
    except subprocess.TimeoutExpired:
        return {"pass": 0, "fail": 0, "crashed": True, "skipped": False,
                "declared_skip": False, "skip_lines": [], "rc": -1,
                "tail": f"timeout >{timeout}s"}
    except Exception as e:  # noqa: BLE001
        return {"pass": 0, "fail": 0, "crashed": True, "skipped": False,
                "declared_skip": False, "skip_lines": [], "rc": -1,
                "tail": f"{type(e).__name__}: {e}"}
    out = p.stdout or ""
    n_pass, n_fail, finished, skipped = count_checks(out)
    tail = "\n".join([ln for ln in out.splitlines() if ln.strip()][-3:])
    #: 工具是否**显式声明**过缺席（[SKIP] 行）。
    #: 这是「少跑断言」的两种解释的分水岭：
    #:   有声明 ⇒ 工具明说了为什么少跑，不是静默失检 ⇒ 跳过并出声；
    #:   无声明 ⇒ 静默失检 ⇒ FAIL，必须查。
    skip_lines = [ln.strip() for ln in out.splitlines() if "[SKIP]" in ln]
    return {"pass": n_pass, "fail": n_fail,
            # 跑完了没有？判据是**有没有打出结项行**，不是退出码 ——
            # 一条合法的 FAIL 不等于工具崩了。
            # ⚠️ 前置缺失（skipped）有结项行 ⇒ 不是 crashed。
            "crashed": (n_fail == 0) and (not finished),
            "skipped": skipped,
            "declared_skip": bool(skip_lines),
            "skip_lines": skip_lines,
            "rc": p.returncode, "tail": tail}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="run_selftests.py",
                                 description="工具包自证入口")
    ap.add_argument("--list", action="store_true", help="只打印清单")
    ap.add_argument("--only", default=None, help="只跑名字含该子串的工具")
    a = ap.parse_args(argv)

    mpath = ROOT / "MANIFEST.json"
    if not mpath.exists():
        print(f"[FATAL] 缺 {mpath} —— 这不是一个完整的包", file=sys.stderr)
        return 2
    with open(mpath, encoding="utf-8") as fh:
        man = json.load(fh)

    tools = [t for t in man["tools"] if t.get("selftest")]
    if a.only:
        tools = [t for t in tools if a.only in t["path"]]

    if a.list:
        for t in man["tools"]:
            st = t.get("selftest")
            scope = st["scope"] if st else "—"
            exp = st.get("expected") if st else None
            print(f"  {scope:<14} {str(exp):>6}  {t['path']}")
        return 0

    print(f"== 工具包自证：{man['package']['name']} v{man['package']['version']} ==")
    n_pass = n_fail = n_skip = n_pre = n_decl = 0
    problems = []
    for t in tools:
        st = t["selftest"]
        if st["scope"] != "standalone":
            n_skip += 1
            print(f"  [SKIP] {t['path']:<44} {st['reason']}")
            continue
        r = run_one(t)
        exp = st.get("expected")
        # 🔴 前置缺失 = 工具**主动声明**跳过，不是崩了，也不是失败。
        #    必须先于 ok 判定处理，否则会被下面的 crashed 分支误报成
        #    「工具崩在自检中途，先修工具」—— 方向完全反了。
        if r["skipped"]:
            n_pre += 1
            why = r["tail"].strip().splitlines()[-1] if r["tail"].strip() else "前置缺失"
            print(f"  [SKIP] {t['path']:<44} 前置缺失（跳过，不记失败）: {why}")
            continue
        ok = (r["fail"] == 0) and (not r["crashed"]) and (exp is None or r["pass"] >= exp)
        if ok:
            n_pass += 1
            # 实测比登记**多** = 工具比登记时更严，不是缺陷，是登记过期了。
            extra = ("" if (exp is None or r["pass"] == exp)
                     else f"  ⚠ 实测比登记多 {r['pass'] - exp} 项 —— "
                          f"登记过期，重跑 `python tools/build_manifest.py` 更新")
            print(f"  [PASS] {t['path']:<44} {r['pass']} 项{extra}")
            continue
        # ── 走到这里才可能是失败 ──
        # 🔴 但「少跑 N 项 + 工具自己声明过缺席」= 声明过的缺席，不是失败。
        #    ⚠️ 这个判定必须**在 n_fail += 1 之前**：
        #    早前放在 else 分支里自增后再 continue，计数已经脏了 ——
        #    表现为「没有任何 [FAIL] 行，结尾却是 1 FAIL」，而且退出码是 1。
        if (exp is not None and r["pass"] < exp and r.get("declared_skip")
                and r["fail"] == 0 and not r["crashed"]):
            n_decl += 1
            print(f"  [SKIP] {t['path']:<44} 实测 {r['pass']} 项，"
                  f"比登记的 {exp} 少 {exp - r['pass']} 项 —— "
                  f"**工具已声明缺席**，非静默失检（跳过，不记失败）")
            for ln in r.get("skip_lines", [])[:4]:
                print(f"         | {ln[:100]}")
            continue
        n_fail += 1
        if r["crashed"]:
            why = (f"未打出结项行（rc={r['rc']}）—— **工具崩在自检中途，先修工具**"
                   if exp is None or r["pass"] >= exp else
                   f"只跑到 {r['pass']} 项且无结项行（rc={r['rc']}）—— "
                   f"**工具崩在自检中途，先修工具**")
        elif r["fail"] > 0:
            why = f"{r['fail']} 项断言不成立 —— **先修工具的断言**"
        elif exp is not None and r["pass"] < exp:
            # 🔴 第三类，与前两类处置方向**都不同**：
            #   工具跑完了、没崩、没断言不成立，但**少跑了断言** ——
            #   这是「静默失检」：包里的工具在丢检查项，且丢得悄无声息。
            #   处置：把该工具降级为 needs-project（它需要项目数据），
            #   不是改断言，也不是改登记数字。
            #   （工具声明过缺席的情况已在上面提前拦掉，那不是静默失检。）
            why = (f"实测 {r['pass']} 项，比 MANIFEST 登记的 {exp} **少 "
                   f"{exp - r['pass']} 项** —— **静默失检**：该工具在包内丢了检查项，"
                   f"应降级为 needs-project")
        else:
            why = f"实测 {r['pass']} 项，与 MANIFEST 登记的 {exp} 不符"
        problems.append((t["path"], why, r["tail"]))
        print(f"  [FAIL] {t['path']:<44} {why}")
    print()
    for path, why, tail in problems:
        print(f"--- {path}\n    {why}")
        if tail:
            for ln in tail.splitlines():
                print(f"    | {ln}")
    print(f"\n{n_pass} PASS   {n_fail} FAIL   （另 {n_skip} 个需项目数据，已跳过并出声）")
    if n_pre:
        print(f"另 {n_pre} 个因**前置缺失**跳过（不算失败）")
    if n_decl:
        print(f"另 {n_decl} 个因**工具已声明缺席**而少跑断言（不是静默失检，不算失败）")
    # 🔴 自检：计数必须与打印出的行数一致。
    #    本项目踩过「没有任何 [FAIL] 行、结尾却是 1 FAIL」的坑 ——
    #    那是 n_fail 在判定之前就自增了。断言放在返回前，坏一次就当场炸。
    n_fail_lines = sum(1 for _ in problems)
    if n_fail != n_fail_lines:
        print(f"[FATAL] 计数自相矛盾：n_fail={n_fail} 但 FAIL 明细 {n_fail_lines} 条")
        return 1
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
