#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""set_repo_url.py -- 把 GitHub 账号一次性写进所有需要它的地方

为什么需要它
------------
仓库地址会出现在**三处**：`CITATION.cff` 的 `repository-code`、
README 顶部的 CI badge、以及 SKILL.md 里的仓库引用。
手改三处 = 迟早漏一处，而漏掉的那处**不报错**，只是 badge 变灰、引用 404。

用法
----
    python scripts/set_repo_url.py --owner YourHandle
    python scripts/set_repo_url.py --url https://github.com/YourHandle/geo-measure-toolkit

🔴 三条纪律
------------
1. **不猜**：没给地址就退出码 2，什么都不改。
2. **只改本仓自己的链接**：🔴 本仓里还有**别人**的仓库链接 ——
   `examples/08_reproduce_geo_bench.py` 指向官方基准仓
   `github.com/GEO-optim/GEO`。第一版我写的是「替换所有 github.com URL」，
   那会把**官方基准链接**改成用户自己的仓 —— 工具会去错误的地址下载数据，
   而且**不报错**。⇒ 现在只替换**本仓的精确 URL** + `__REPO_OWNER__` 占位符。
3. **改完验金丝雀**：官方基准链接必须原封不动（见 `_CANARY`）。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OWNER_TOK = "__REPO_OWNER__"
REPO_NAME = "geo-measure-toolkit"
TARGETS = ("CITATION.cff", "README.md", "SKILL.md")

#: 金丝雀：本仓引用**别人**的仓库地址。替换脚本**绝不许**碰它。
#: 它一被改坏，08 号脚本就会去错误的地址拉官方基准 —— 静默的、后果严重的错。
_CANARY = ("examples/08_reproduce_geo_bench.py", "github.com/GEO-optim/GEO")


def parse_url(url: str) -> str:
    """从 URL 里取 owner；仓库名必须与本仓一致（否则是在指向别的仓）。"""
    m = re.match(r"^https?://github\.com/([^/\s]+)/([^/\s#?]+?)(?:\.git)?/?$", url.strip())
    if not m:
        raise ValueError(f"不是合法的 GitHub 仓库地址：{url!r}")
    owner, name = m.group(1), m.group(2)
    if name.lower() != REPO_NAME.lower():
        raise ValueError(f"仓库名必须是 {REPO_NAME}，收到 {name!r}")
    return owner


def current_owner() -> str | None:
    p = ROOT / "CITATION.cff"
    if not p.exists():
        return None
    m = re.search(r"^repository-code:\s*[\"']?https?://github\.com/([^/\"\s]+)/",
                  p.read_text(encoding="utf-8"), re.M)
    if not m:
        return None
    o = m.group(1)
    return None if OWNER_TOK in o else o


def canary_ok() -> bool:
    rel, needle = _CANARY
    p = ROOT / rel
    return p.exists() and needle in p.read_text(encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="set_repo_url.py")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--owner")
    g.add_argument("--url", help=f"https://github.com/<owner>/{REPO_NAME}")
    a = ap.parse_args(argv)

    try:
        owner = parse_url(a.url) if a.url else (a.owner or "").strip()
    except ValueError as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        return 2
    if not owner or OWNER_TOK in owner:
        print(f"[FATAL] 需要一个真实的 GitHub 账号名（--owner <账号>）", file=sys.stderr)
        return 2

    cur = current_owner()
    changed = []
    for rel in TARGETS:
        p = ROOT / rel
        if not p.exists():
            continue
        t = p.read_text(encoding="utf-8")
        o = t
        # ① 占位符
        t = t.replace(OWNER_TOK, owner)
        # ② 本仓的**精确**旧 URL（重指向用）。只针对本仓，绝不泛化匹配 github.com。
        if cur and cur != owner:
            t = t.replace(f"github.com/{cur}/{REPO_NAME}",
                          f"github.com/{owner}/{REPO_NAME}")
        if t != o:
            p.write_text(t, encoding="utf-8")
            changed.append(rel)

    if not changed:
        print(f"仓库地址已经是 https://github.com/{owner}/{REPO_NAME} —— 无需改动")
        return 0
    print(f"仓库地址：https://github.com/{owner}/{REPO_NAME}")
    print(f"已更新 {len(changed)} 个文件：{', '.join(changed)}")

    # 金丝雀：官方基准链接必须原封不动
    if not canary_ok():
        print(f"❌ 金丝雀被破坏：{_CANARY[1]} 不再出现在 {_CANARY[0]} —— "
              f"官方基准链接被误改了，请检查", file=sys.stderr)
        return 1
    print(f"✅ 金丝雀完好：{_CANARY[1]} 未被触碰")
    return 0


if __name__ == "__main__":
    sys.exit(main())
