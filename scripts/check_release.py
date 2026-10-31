#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_release.py -- 发布前的卫生检查

它守的是**「该做的做了没有」**这一类问题 —— 这类问题的特点是：
不做不会报错，只是东西悄悄地对不上。

检查项
------
1. `CITATION.cff` 存在，且含 CFF 必需字段；
2. 仓库地址**不是占位符**（`__REPO_OWNER__` / `__REPO_NAME__`）；
3. `CITATION.cff` 的 `version` 与 `MANIFEST.json` 的版本**一致**；
4. `CITATION.cff` 声明的 license 与仓库里的 `LICENSE` 文件**对得上**；
5. README 的 CI badge 地址与 `CITATION.cff` 的 `repository-code` **同一个仓**；
6. `LICENSE` 里的版权行不是占位符；
7. `README.md` 与 `SKILL.md` 里**本仓自己**的每一处地址都指向同一个仓，且无占位符。

用法：
    python scripts/check_release.py
退出码：0 = 干净；1 = 有问题（逐条列出）。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
# ── 控制台编码兜底 ────────────────────────────────────────────────
# Windows 控制台默认 GBK 码页，打印 ✅/⚠/❌/🔴/⇒/² 这类字符会 UnicodeEncodeError 崩溃。
# 这里把「无法编码」改成「替换显示」而不是抛异常；UTF-8 环境（Linux/macOS/CI）下是空操作。
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
OWNER_TOK = "__REPO_OWNER__"
NAME_TOK = "__REPO_NAME__"
#: 本仓的仓库名 —— 只有引用**这个**名字的地址才需要与 repository-code 一致；
#: 别人的仓（官方 GEO-bench 等）合法，不查。
REPO_NAME = "geo-measure-toolkit"

CFF_REQUIRED = ("cff-version", "message", "title", "authors", "version",
                "date-released", "license", "repository-code")

#: 金丝雀：本仓引用**别人**的仓库地址。它一被改坏，08 号脚本就会去错误的地址
#: 拉官方基准 —— 静默的、后果严重的错。所以发布前必须确认它原封不动。
CANARY = ("examples/08_reproduce_geo_bench.py", "github.com/GEO-optim/GEO")


def _repo_refs(text: str) -> set[tuple[str, str]]:
    """取出文本里所有 github.com/<owner>/<repo> 形式的仓库引用（去掉 .git 后缀）。"""
    out: set[tuple[str, str]] = set()
    pat = r"github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)"
    for owner, name in re.findall(pat, text):
        out.add((owner, name[:-4] if name.endswith(".git") else name))
    return out


def _cff_get(text: str, key: str) -> str | None:
    """取**顶层标量**字段的值。CFF 是 YAML，但本仓不引入 yaml 依赖（纯标准库），
    所以用受限的正则取这几个已知字段。列表字段（authors）取不到值，
    那是正常的 —— 判断它们在不在要看 `_cff_has`。"""
    m = re.search(rf"^{re.escape(key)}:\s*[\"']?([^\"'\n]+)[\"']?\s*$",
                  text, re.M)
    return m.group(1).strip() if m else None


def _cff_has(text: str, key: str) -> bool:
    """顶层**键是否存在**（不管它是标量还是列表）。

    🔴 第一版只用 `_cff_get` 判必需字段，于是 `authors:`（列表）被判成缺失 ——
       假警报比漏报更糟：它会让人以为文件坏了，去改一个没坏的东西。
    """
    return bool(re.search(rf"^{re.escape(key)}\s*:", text, re.M))


def main(argv=None) -> int:
    problems: list[str] = []
    # 🔴 成功输出必须能回答"检查了几个" —— 0 个引用也算"通过"是最危险的假绿。
    refs_checked = 0
    files_scanned = 0

    cff_path = ROOT / "CITATION.cff"
    if not cff_path.exists():
        problems.append("缺 CITATION.cff —— 别人没法引用你")
        cff_text = ""
    else:
        cff_text = cff_path.read_text(encoding="utf-8")

    for key in CFF_REQUIRED:
        if cff_text and not _cff_has(cff_text, key):
            problems.append(f"CITATION.cff 缺必需字段：{key}")

    repo_code = _cff_get(cff_text, "repository-code") or ""
    if OWNER_TOK in repo_code or NAME_TOK in repo_code:
        problems.append(
            f"CITATION.cff 的 repository-code 还是占位符（{repo_code}）—— "
            f"跑 `python scripts/set_repo_url.py --owner <你的 GitHub 账号>`")
    if repo_code and "github.com" not in repo_code:
        problems.append(f"repository-code 不是 GitHub 地址：{repo_code}")

    # 版本一致性
    man_path = ROOT / "MANIFEST.json"
    if man_path.exists():
        man = json.loads(man_path.read_text(encoding="utf-8"))
        mver = (man.get("package") or {}).get("version")
        cver = _cff_get(cff_text, "version")
        if mver and cver and mver != cver:
            problems.append(f"版本不一致：MANIFEST={mver} 而 CITATION.cff={cver}")

    # license 一致性
    lic_path = ROOT / "LICENSE"
    if lic_path.exists():
        lic = lic_path.read_text(encoding="utf-8")
        clf = (_cff_get(cff_text, "license") or "").upper()
        if clf and clf not in lic.upper():
            problems.append(f"CITATION.cff 声明 license={clf}，但 LICENSE 文件里没有它")
        m = re.search(r"Copyright \(c\)\s*(\d{4})?\s*(.*)", lic)
        if m and m.group(2).strip() in ("", "一左一右", "TODO", "XXX"):
            problems.append("LICENSE 的版权行还是占位符")

    # badge 与 repository-code 同仓
    readme = ROOT / "README.md"
    if readme.exists():
        rt = readme.read_text(encoding="utf-8")
        if OWNER_TOK in rt or NAME_TOK in rt:
            problems.append("README 里还有仓库地址占位符（badge 会指向不存在的仓库）")
        badges = re.findall(r"github\.com/([^\s/]+)/([^\s/]+)/actions/workflows", rt)
        if badges:
            cparts = re.sub(r"^https?://github\.com/", "", repo_code).strip("/")
            for owner, name in sorted(set(badges)):
                if f"{owner}/{name}" != cparts:
                    problems.append(
                        f"README 的 badge 指向 {owner}/{name}，"
                        f"而 CITATION.cff 指向 {cparts} —— 不是同一个仓")

    # 本仓地址一致性（README 与 SKILL.md）
    #
    # 🔴 为什么要连 `SKILL.md` 一起查：`set_repo_url.py` 写**三处**
    #    （CITATION.cff / README / SKILL.md），而上面那块只验了前两处。
    #    ⇒ SKILL.md 指向一个不存在的仓时，**没有任何检查会报错**，
    #      只是别人 clone 不到 —— 这正是"静默地对不上"那一类问题。
    # 🔴 只查**仓库名等于本仓名**的引用：README 里指向官方 GEO-bench 的链接
    #    是合法的，不该被误报（假警报比漏报更坏）。
    # 🔴 上面那块 badge 检查**必须保留**：badge 指向的仓名可能整个换掉，
    #    那时它就不再是"本仓的引用"，下面的通用检查会跳过它 ⇒ 会漏报。
    cparts_all = ""
    if repo_code:
        cparts_all = re.sub(r"^https?://github\.com/", "", repo_code).strip("/")
    for rel in ("README.md", "SKILL.md"):
        fp = ROOT / rel
        if not fp.exists():
            continue
        t = fp.read_text(encoding="utf-8")
        files_scanned += 1
        if OWNER_TOK in t or NAME_TOK in t:
            problems.append(
                f"{rel} 里还有仓库地址占位符（{OWNER_TOK} / {NAME_TOK}）")
        if not cparts_all:
            continue
        for owner, name in sorted(_repo_refs(t)):
            if name != REPO_NAME:
                continue  # 别人的仓（官方基准等），合法
            refs_checked += 1
            if f"{owner}/{name}" != cparts_all:
                problems.append(
                    f"{rel} 里的本仓地址指向 {owner}/{name}，"
                    f"而 CITATION.cff 指向 {cparts_all} —— 不是同一个仓")
        if f"github.com/{cparts_all}" not in t:
            problems.append(
                f"{rel} 里找不到本仓地址 github.com/{cparts_all} —— 可能被整个改成了别的仓")

    # 金丝雀：别人仓库的链接必须原封不动
    rel, needle = CANARY
    cp = ROOT / rel
    if not cp.exists():
        problems.append(f"金丝雀文件不存在：{rel}")
    elif needle not in cp.read_text(encoding="utf-8"):
        problems.append(
            f"金丝雀被破坏：{rel} 里找不到 {needle} —— "
            f"官方基准链接被误改了（08 号脚本会去错误的地址下载）")

    if problems:
        print("❌ 发布卫生检查未通过：")
        for p in problems:
            print(f"   [FAIL] {p}")
        return 1
    print("✅ 发布卫生检查通过：CITATION / license / 版本 / badge 地址一致；"
          "官方基准链接金丝雀完好"
          f"（读了 {files_scanned} 个文件，核了 {refs_checked} 处本仓地址）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
