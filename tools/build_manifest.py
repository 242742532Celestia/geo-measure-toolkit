#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_manifest.py -- 生成本仓库的 MANIFEST.json

为什么需要它
------------
`MANIFEST.json` 里每个工具的 `expected`（自检项数）**必须由实跑得到**，
不许手抄 —— 手抄的数字一旦和代码脱节，「包里的东西」和「登记的东西」
就不一致了，而这种漂移**不报错**，只是悄悄地不再可信。

本脚本是那个「不许手抄」的机器保证：

    python tools/build_manifest.py

它做三件事：
  1. 逐个工具跑自检，用**包内 runner 的同一个** `count_checks()` 实现计数
     （从 `run_selftests.py` import 进来，不另写一份正则 —— 两套正则必然漂移）；
  2. 把「实测项数」写进 `MANIFEST.json`；
  3. 重算每个文件的 sha256 与字节数。

🔴 三条纪律（与本仓库一致）
--------------------------
1. **结项行格式固定**：`N PASS   M FAIL`。核对器靠这个正则判断「工具是否跑完了」
   —— 没有结项行会被判成「崩在自检中途」，处置方向正好相反。
   ⚠️ 已知存在**五种**结项格式，必须都认：
     A. `N PASS   M FAIL`（多数工具）
     B. `all checks passed`（statpower / geo_stat_lint / tautology_audit …）
     C. `24/24` 单独一行（fetch_citation_pdfs）
     D. `■ …自洽` 收尾（info_gate_selfcontained_check）
     E. `N PASS   M FAIL   前置缺失：…`（工具主动声明前置条件不成立 → 跳过）
2. **「需要项目数据」不等于「坏了」**：依赖外部数据/参考模块的工具被
   **跳过并出声**，**不记 PASS**、**也不记 FAIL** —— 证据不在场与功能缺失不能混报。
3. **实测项数与 MANIFEST 不符 = FAIL**。MANIFEST 是实跑生成的，不是手抄的；
   跑出不同数字说明**仓库里的东西和登记的不一致**，必须查。
   ⚠️ `expected` 记的是**本机这次实跑**的项数 —— 某些工具的项数依赖环境
   （例如 `info_gate` 在 `GEO_CORE_PATH` 指向的参考模块在场时多跑几条
   「与包外参考逐位相等」的断言）。**少跑且工具自己打了 `[SKIP]`** ⇒
   声明过的缺席，跳过不算失败；**少跑且工具什么都没说** ⇒ 静默失检，必须查。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── 工具登记表 ────────────────────────────────────────────────────────────────
#: `args` 是**配置**（怎么跑自检），可以手写；`expected`（跑出几项）是**实测**，
#: 由下面的实跑填进去 —— 两者严格分开，别混。
#:
#: `scope` 三种：
#:   standalone    —— 自检能在本仓库内独立跑通（无项目数据、无外网、无包外模块）
#:   needs-project —— 自检需要本项目的产出数据（在这个仓库里**跑不通**），
#:                    放进来是为了保留判据，**跳过并出声**，不记 PASS 也不记 FAIL
#:   none          —— 工具没有自检入口（**验证缺口**，必须人工确认）
TOOLS = [
    dict(path="tools/statpower.py", args=["selftest"], scope="standalone",
         role="Power/MDE inversion kernel: mde_* / n_required_* / deff / wilson_ci / permutation_p"),
    dict(path="tools/geo_stat_lint.py", args=["selftest"], scope="standalone",
         role="Six statistical lint rules L1-L6 (no-interval / below-noise / rare-binary-r / cluster-as-independent / uninterpretable-null / under-identified)"),
    dict(path="tools/tautology_audit.py", args=["--selftest"], scope="standalone",
         role="Tautology audit: leakage score L(T,Y) with four severity bands"),
    dict(path="tools/freeze_prompts.py", args=["--selftest"], scope="standalone",
         role="Prompt freezing + --verify: snapshot + sha256, the basis for rebuilding paired runs"),
    dict(path="tools/geo_page_features.py", args=["--selftest"], scope="standalone",
         role="Page-level candidate feature extractor (no scores, no weights, no grades)"),
    dict(path="tools/info_gate.py", args=["--selftest"], scope="standalone",
         role="Information-preservation gate G1-G9 for free-text rewrites"),
    dict(path="tools/info_gate_selfcontained_check.py", args=[], scope="standalone",
         role="Self-consistency proof: hides the optional external reference module, then re-runs the key paths"),
    dict(path="examples/08_reproduce_geo_bench.py", args=[], scope="none",
         role="Reproduces the official GEO-bench harness (rate limiter, retry, verification)",
         note="No self-test entry -- verification gap, confirm by hand before relying on it"),
    dict(path="examples/10_citation_noise.py", args=["--selftest"], scope="standalone",
         role="Re-measures run-to-run noise on the citation-rate outcome"),
    dict(path="examples/11_construct_validity.py", args=["--selftest"], scope="standalone",
         role="Construct validity: is the measurement measuring what we claim?"),
    dict(path="examples/12_probe_site_facts.py", args=["--selftest"], scope="standalone",
         role="Probes site-level facts for a list of domains"),
    dict(path="examples/13_infrastructure_survey.py", args=["--selftest"], scope="standalone",
         role="Splits literature into infrastructure layer vs method layer"),
    dict(path="examples/14_stage_a_variance.py", args=["--selftest"], scope="standalone",
         role="Variance decomposition sigma_b / sigma_e with a synthetic-data recovery self-test"),
    dict(path="examples/17_extend_geobench_cache.py", args=["--selftest"], scope="standalone",
         role="Extends the official GEO-bench query cache without changing the filter rule"),
    dict(path="examples/19_paired_analysis.py", args=["--selftest"], scope="standalone",
         role="Paired analysis main program (Holm correction, paired_t degenerate semantics)"),
    dict(path="examples/20_rewriter_noise.py", args=["--selftest"], scope="standalone",
         role="Measures the noise a rewriting system introduces"),
    dict(path="examples/23_e4_sequential.py", args=["--selftest"], scope="standalone",
         role="Sequential stopping rules S1/S2/S3 over a balanced prefix"),
    dict(path="examples/24_e4_noise_floor.py", args=["--selftest"], scope="standalone",
         role="Variance decomposition + arithmetic lower bound on MDE"),
]


def _load_counter():
    """从仓库根 import 包内 runner，拿到**同一个** count_checks 实现。

    这是「一个事实来源」的技术保证：本脚本算出来的 `expected` 与
    `run_selftests.py` 核对时用的口径是同一份代码，不存在两套正则漂移。
    """
    mod_path = ROOT / "run_selftests.py"
    spec = importlib.util.spec_from_file_location("repo_runner", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.count_checks


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _measure(cmd, count_checks, timeout: int = 600) -> dict:
    """跑一次自检并计数。返回实测字典。"""
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
           "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           encoding="utf-8", errors="replace", cwd=str(ROOT),
                           env=env)
    except subprocess.TimeoutExpired:
        return {"pass": 0, "fail": 0, "finished": False, "skipped": False,
                "declared_skip": False, "rc": -1, "why": f"timeout >{timeout}s"}
    except Exception as e:  # noqa: BLE001
        return {"pass": 0, "fail": 0, "finished": False, "skipped": False,
                "declared_skip": False, "rc": -1, "why": f"{type(e).__name__}: {e}"}
    out = (p.stdout or "") + "\n" + (p.stderr or "")
    n_pass, n_fail, finished, skipped = count_checks(out)
    tail = "\n".join([ln for ln in (p.stdout or "").splitlines() if ln.strip()][-3:])
    return {"pass": n_pass, "fail": n_fail, "finished": finished,
            "skipped": skipped, "declared_skip": "[SKIP]" in out,
            "rc": p.returncode, "why": "", "tail": tail}


def build() -> dict:
    count_checks = _load_counter()
    entries = []
    problems = []
    for t in TOOLS:
        rel = t["path"]
        src = ROOT / rel
        if not src.exists():
            print(f"[FATAL] 登记表里的路径不存在：{rel}", file=sys.stderr)
            raise SystemExit(2)
        scope = t["scope"]
        entry = {
            "path": rel,
            "role": t["role"],
            "sha256": _sha256(src),
            "bytes": src.stat().st_size,
        }
        if scope == "none":
            # 工具**没有**自检入口 —— 这是验证缺口，必须写清楚，不能装作没事。
            entry["selftest"] = None
            entry["tier"] = "-"
            entry["verification"] = ("⚠ No self-test entry -- verification gap; "
                                     "confirm by hand before relying on it")
            entries.append(entry)
            print(f"  [SKIP] {rel:<44} 无自检入口（验证缺口）")
            continue
        r = _measure([sys.executable, rel] + list(t["args"]), count_checks)
        if scope == "needs-project":
            # 自检需要项目产出数据 ⇒ 本仓库里跑不通 ⇒ 跳过并出声，**不记 PASS 也不记 FAIL**
            entry["selftest"] = {
                "args": list(t["args"]),
                "scope": "needs-project",
                "expected": None,
                "reason": t.get("note", "Self-test needs project data -- skipped here"),
                "measured": r,
            }
            entry["tier"] = "B"
            entries.append(entry)
            print(f"  [SKIP] {rel:<44} 需项目数据（跳过并出声）")
            continue
        # scope == standalone：必须是绿的，否则这个 scope 就是错的
        green = (r["finished"] and not r["skipped"] and r["fail"] == 0 and r["pass"] > 0)
        if not green:
            problems.append(f"{rel}: 声明为 standalone 但实跑不绿 "
                            f"(pass={r['pass']} fail={r['fail']} finished={r['finished']})")
        entry["selftest"] = {
            "args": list(t["args"]),
            "scope": "standalone",
            "expected": r["pass"] if r["finished"] else None,
            "reason": "Runnable standalone inside this repository",
            "measured": r,
        }
        entry["tier"] = "A"
        entries.append(entry)
        print(f"  [{'OK ' if green else 'FAIL'}] {rel:<44} {r['pass']} 项  fail={r['fail']}")
    if problems:
        print()
        for p in problems:
            print(f"  [FAIL] {p}")
        raise SystemExit(1)
    return {
        "package": {
            "name": "geo-measure-toolkit",
            "version": "1.0.0",
            "built_at": _now(),
            "description": ("Measurement and judgment toolkit for Generative Engine Optimization "
                            "(GEO) experiments: power/MDE inversion, statistical lint rules, "
                            "tautology audit, prompt freezing, feature extraction and the "
                            "information-preservation gate."),
        },
        "tools": entries,
    }


def _now() -> str:
    import datetime
    return datetime.datetime.now().astimezone().replace(microsecond=0).isoformat()


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--check" in argv:
        # 只核对：现��� MANIFEST 的 sha256/项数与当前文件/实跑是否一致
        man_path = ROOT / "MANIFEST.json"
        if not man_path.exists():
            print("[FATAL] 缺 MANIFEST.json", file=sys.stderr)
            return 2
        man = json.loads(man_path.read_text(encoding="utf-8"))
        count_checks = _load_counter()
        bad = []
        for t in man["tools"]:
            st = t.get("selftest")
            if not st:
                # 无自检入口的工具：只能核对文件哈希，核对不了项数
                src = ROOT / t["path"]
                if not src.exists():
                    bad.append(f"{t['path']}: 文件不存在")
                elif _sha256(src) != t["sha256"]:
                    bad.append(f"{t['path']}: sha256 与 MANIFEST 不符")
                continue
            src = ROOT / t["path"]
            if not src.exists():
                bad.append(f"{t['path']}: 文件不存在")
                continue
            if _sha256(src) != t["sha256"]:
                bad.append(f"{t['path']}: sha256 与 MANIFEST 不符")
            r = _measure([sys.executable, t["path"]] + list(st["args"]), count_checks)
            exp = st.get("expected")
            if exp is None:
                continue
            if r["pass"] < exp and not r["declared_skip"]:
                bad.append(f"{t['path']}: 实测 {r['pass']} < 登记 {exp} 且未声明缺席 —— 静默失检")
            elif r["pass"] != exp:
                print(f"  [NOTE] {t['path']}: 实测 {r['pass']} vs 登记 {exp}"
                      f"{'（已声明缺席）' if r['declared_skip'] else ''}")
        if bad:
            for b in bad:
                print(f"  [FAIL] {b}")
            print(f"\n{len(bad)} 处不一致")
            return 1
        print("✅ MANIFEST 与仓库当前状态一致")
        return 0
    man = build()
    out = ROOT / "MANIFEST.json"
    out.write_text(json.dumps(man, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n写入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
