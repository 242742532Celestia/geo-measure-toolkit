#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_deps.py -- 仓库内依赖闭包审计

为什么需要它
------------
🔴 一次真实的翻车记录（本仓库的立身之本）：
   `examples/23_e4_sequential.py` 用 `importlib.util.spec_from_file_location`
   **动态**加载 `examples/19_paired_analysis.py`。这个依赖**不在**任何静态
   import 语句里，所以「按 import 语句扫描」的工具看不见它 ——
   结果打包时把依赖漏掉，包内一跑就 `FileNotFoundError` 崩在 import 阶段。

   ⇒ 静态扫描不够。**动态依赖用「.py 字符串字面量」找**：
      凡是被当作路径用的 `.py` 字面量，都可能是动态 import 的目标。

它做什么
--------
1. 扫每个 `.py` 文件的 AST：
   - 静态 import（`import X` / `from X import ...`）；
   - **动态**依赖：所有以 `.py` 结尾的字符串字面量（`spec_from_file_location`
     的典型用法就是传一个路径字符串）。
2. 把每个依赖解析到**仓库内**的具体文件；
3. 解析不到的 → 报 FAIL（那个文件不在本仓库里，跑起来就会崩）；
4. 顺带报出**孤儿**（仓库里没有任何文件依赖它的文件）—— 提示可能多带了。

用法：
    python scripts/check_deps.py          # 审计
    python scripts/check_deps.py --quiet  # 只报问题
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STDLIB_OK = True  # 标准库不查


def py_files() -> list:
    return sorted(p for p in ROOT.rglob("*.py")
                  if "__pycache__" not in p.parts and ".git" not in p.parts)


def literal_py_paths(tree: ast.AST) -> list:
    """所有以 .py 结尾的字符串字面量 —— 动态 import 的载体。"""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            v = node.value.strip()
            # 排除模式串（`*.py`）与占位符（`<x>.py`）—— 它们不是依赖
            if (v.endswith(".py") and len(v) < 120 and "\n" not in v
                    and "*" not in v and "<" not in v and ">" not in v
                    and len(v) > 3):   # 排除裸的 ".py"（那是后缀，不是路径）
                out.append(v)
    return out


def static_imports(tree: ast.AST) -> list:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.append(node.module)
    return out


def resolve(name: str, src: Path) -> Path | None:
    """把一个依赖名解析到仓库内的文件；解析不到返回 None。"""
    cands = []
    if os.path.isabs(name):
        cands.append(Path(name))
    else:
        cands += [
            src.parent / name,
            ROOT / name,
            ROOT / "tools" / name,
            ROOT / "examples" / name,
            ROOT / "scripts" / name,
        ]
    for c in cands:
        try:
            if c.exists() and c.is_file():
                return c.resolve()
        except OSError:
            continue
    return None


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    quiet = "--quiet" in argv
    files = py_files()
    deps: dict[Path, set] = {}
    missing: list = []
    local_mods = {p.stem for p in files}

    for f in files:
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        except SyntaxError as e:
            print(f"  [FAIL] {f.relative_to(ROOT).as_posix()}: 语法错误 {e}")
            missing.append((f, f"<syntax error>"))
            continue
        found = set()
        for lit in literal_py_paths(tree):
            r = resolve(lit, f)
            if r is not None:
                if r != f.resolve():        # 自己引用自己不算依赖
                    found.add(r)
            elif Path(lit).name in local_mods or lit.endswith(".py"):
                # 以 .py 结尾却解析不到 —— 多半是**漏带的依赖**
                missing.append((f, lit))
        for mod in static_imports(tree):
            if mod.split(".")[0] in local_mods:
                stem = mod.split(".")[0]
                for p in files:
                    if p.stem == stem:
                        found.add(p.resolve())
                        break
        deps[f.resolve()] = found

    # 反向索引：谁依赖我
    depended = set()
    for s in deps.values():
        depended |= s

    print(f"== 依赖闭包审计：{len(files)} 个 .py 文件 ==")
    for f in files:
        ds = sorted(deps.get(f.resolve(), set()))
        names = ", ".join(d.relative_to(ROOT).as_posix() for d in ds) or "—"
        orphan = "  ⚠ 孤儿（无人依赖）" if f.resolve() not in depended else ""
        if not quiet or ds:
            print(f"  {f.relative_to(ROOT).as_posix():<48} -> {names}{orphan}")

    print()
    if missing:
        print(f"❌ 解析不到的依赖 {len(missing)} 处 —— 这些文件在本仓库里不存在，"
              f"跑到就会崩：")
        for f, name in missing:
            print(f"   [FAIL] {f.relative_to(ROOT).as_posix()}: {name}")
        return 1
    print("✅ 依赖闭包完整：所有动态/静态依赖都能在仓库内解析")
    return 0


if __name__ == "__main__":
    sys.exit(main())
