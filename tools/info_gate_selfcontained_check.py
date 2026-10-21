"""自洽性验证：把包外模块**临时藏起来**，再跑关键路径。

目的：证明 `tools/info_gate.py` 现在**不依赖** an optional external reference module。
讲「本包已自洽」必须有可执行的证据 —— 文档里写「不依赖」不算。

做法：把包外目录改名为 `.hidden_by_selfcheck`，跑若干入口，
再改回来。**任何一步失败都必须原样恢复**，否则会把用户的
skill 目录留在损坏状态。

配套：`info_gate.selftest()` 里「与包外参考逐位相等」那条断言，
在参考不可用时**跳过**（打印 SKIP）而不是判 FAIL —— 否则本检查会把
「本包已自洽」这个正确结论报成 bug。

🔴 2026-10-21 修正（装到别的机器上会假报 bug 的根因）
--------------------------------------------------------
原来 `EXT` 不存在时直接 `raise SystemExit(0)` 提前放弃，只打一行 `[SKIP]`。
后果：**包内 runner 判不出结项行** ⇒ 判成「工具崩在自检中途，先修工具」，
而工具其实行为完全正确 —— 处置方向完全反了。

但「包外模块缺失」**正是本脚本要证明的状态**。
⇒ `EXT` 不存在时不需要藏任何东西，直接跑关键路径即可，
  得到的结论比「放弃」强得多。
⇒ 只有「存在但藏不住」（权限/占用等）才是真正的前置缺失，
  那种情况才打结项行 `N PASS   M FAIL   前置缺失：…` 声明跳过。
"""
import os
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(errors="replace")

#: 包外参考模块目录。🔴 2026-10-21 开源化：改为**可选**的环境变量
#: `GEO_CORE_PATH`（原先是作者机器上的绝对路径）。
#:    - 没设 → 「包外模块缺失」这个前提**本来就成立**，直接跑关键路径；
#:    - 设了但目录不存在 → 同上；
#:    - 设了且存在 → 临时改名藏起来再跑，证明「不依赖包外」有可执行证据。
_geo_core_raw = os.environ.get("GEO_CORE_PATH", "").strip()
EXT = Path(_geo_core_raw) if _geo_core_raw else None
HIDE = (EXT.with_name(EXT.name + ".__hidden_by_selfcheck")
        if EXT is not None else None)
ROOT = Path(__file__).resolve().parent.parent

#: 关键路径。🔴 只保留**本仓库里真的存在**的脚本 ——
#: 指向不存在的脚本会报「can't open file」，那是脚本缺失，不是「不自洽」。
#: ⚠️ 过滤时要看命令里**指向 .py 文件的那个参数**，
#:    不能无脑取 args[1] —— 内联代码命令是 [python, "-c", "..."]，没有脚本路径。
CMDS = [
    ("自检（全部断言）", [sys.executable, "tools/info_gate.py", "--selftest"]),
    ("G1-G9 自由文本证据", [sys.executable, "-c",
                    "import sys; sys.path.insert(0,'tools');"
                    "import info_gate,json;"
                    "d=info_gate.g1g9_freetext_evidence();"
                    "print('ok', list(d)[:4])"]),
]


def _scripts_present(cmd) -> bool:
    """命令引用的 .py 脚本是否都在本仓库里；内联代码（无 .py 参数）一律保留。"""
    for a in cmd[1:]:
        if a.endswith(".py"):
            return (ROOT / a).exists()
    return True


CMDS = [(n, c) for (n, c) in CMDS if _scripts_present(c)]

#: 「包外模块缺失」这个前提**本来就成立**时，不需要藏，直接跑。
#: 这是本脚本最有价值的那种情形（换机器安装后自证），不能因为
#: 「没东西可藏」就放弃 —— 放弃等于把最强的一票投给了最弱的结论。
need_hide = (EXT is not None and EXT.exists() and HIDE is not None)

restored = False
hidden = False
_ext_label = (EXT.name if EXT is not None else "GEO_CORE_PATH（未设置）")
try:
    try:
        if need_hide:
            EXT.rename(HIDE)
            hidden = True
            print(f"已临时隐藏 {_ext_label}\n")
        else:
            print(f"[NOTE] 包外参考模块 {_ext_label} 不存在 —— "
                  f"「包外模块缺失」这个前提**本来就成立**，"
                  f"直接跑关键路径（无需隐藏）\n")
    except OSError as e:
        # 存在但藏不住（权限/被占用）⇒ 真正的前置缺失。
        # 🔴 必须打出结项行并声明「前置缺失」，否则 runner 会判成
        #    「崩在自检中途」—— 那会把人引去修一个没坏的工具。
        print(f"0 PASS   0 FAIL   前置缺失：无法临时隐藏 {_ext_label} "
              f"（{type(e).__name__}: {e}）")
        raise SystemExit(0)

    allok = True
    for name, cmd in CMDS:
        p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=300)
        tail = [l for l in (p.stdout or "").splitlines() if l.strip()][-1:]
        ok = p.returncode == 0
        allok &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:22s} exit={p.returncode}"
              f"  {tail[0][:70] if tail else ''}")
        if not ok:
            # 必须报出**是哪几条断言**失败，否则「未自洽」这个结论没有行动价值
            for l in (p.stdout or "").splitlines():
                if "[FAIL]" in l or "[SKIP]" in l:
                    print(f"        {l.strip()}")
            if p.stderr:
                print(f"        stderr: {p.stderr.strip()[-200:]}")
    print()
    print("■ 包外模块缺失时全链路仍然通过 -> 本复现包**自洽**"
          if allok else
          "■ 仍有步骤依赖包外模块 -> **未自洽**，见上面 FAIL")
finally:
    if hidden and HIDE is not None and HIDE.exists() and not EXT.exists():
        HIDE.rename(EXT)
        restored = True
        print(f"\n已恢复 {_ext_label}")
    elif HIDE is not None and HIDE.exists() and EXT is not None and EXT.exists():
        print(f"\n⚠️ {HIDE} 与 {EXT} 同时存在，未自动恢复，请人工检查")
if not restored and hidden:
    print("⚠️ 恢复未完成 —— 请人工确认 skills 目录状态")
