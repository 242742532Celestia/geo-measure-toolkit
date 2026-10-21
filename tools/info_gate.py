#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""info_gate.py -- 自由文本改写的信息保持门（GEO-bench 语料适用）

为什么不能直接用 `_geo_core.evidence_gate`
-------------------------------------------
Method §3 要求改写必须过信息保持门，并指向 `_geo_core` 的 G1-G9。
2026-10-06 实测：**在 GEO-bench 的自由文本上，G1-G9 全部静默通过。**

    源文本：Himalayan glow serum uses 1000mg of vitamin C ...
    改写：  加入「皮肤科 Dr. Chen 推荐」「5000 人临床研究」「98% 两个月变亮」
            「2024 全球美容奖」「Trustpilot 上的 Sara 评价」
    -> evidence_gate.check(...) 返回 pass=True, findings=[]

原因不是 bug，是**语料不匹配**：`evidence_gate` 的 G1-G9 是按
**商品结构化字段**扫描的（`aggregateRating` / `review_keywords` /
`positive_review_rate` / 认证字段 + 领域 profile 词表），
不是扫散文。自由文本里它什么也找不到，于是 pass。

> **一个静默通过、看起来在把关的检查，比没有检查更危险。**
> 这正是本项目反复遇到的那类失效，所以必须显式记录而不是绕过去。

本模块做什么
------------
在**同一批事实上**做**可复核**的检查（绝不用 LLM 当裁判，
也不用被检验对象自己的输出去定义通过标准）：

| 码 | 含义 | 级别 |
|---|---|---|
| `F1_NEW_NUMBER` | 改写引入了源文本没有的数字 | BLOCK |
| `F2_NEW_ENTITY` | 改写引入了源文本没有的命名实体/专名 | BLOCK |
| `F3_ATTRIBUTION` | 改写把话归给源文本没有的第三方 | BLOCK |
| `F4_EXTERNAL_REF` | 改写引入了源文本没有的外部引用/评测/奖项/背书 | BLOCK |
| `F5_RETENTION` | 源文本特征词流失过多（信息扭曲的另一半） | BLOCK |
| `F6_LENGTH` | 改写超过 §3 预注册的 1.5x 长度上限 | BLOCK |

`F5` **直接复用** `_geo_core.evidence_gate.retention()` 与其
`RETENTION_FLOOR` 常量 —— 那里在自由文本上**确实有效**，就应当复用，
不重复实现。语料不匹配的部分（F1-F4、F6）在这里补齐。

阴性对照（硬要求）
------------------
`selftest()` 注入一条**刻意编造**的改写，要求被 BLOCK；
再注入一条**逐字相同**的改写，要求通过。
一个只会亮红灯的检查器没有说服力 —— 与 `tautology_audit` 同一纪律。

用法
----
    python tools/info_gate.py --selftest
    python tools/info_gate.py --source-file s.txt --rewrite-file r.txt
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Iterable

# 🔴 Windows 控制台默认 GBK，emoji（🔴）会直接抛 UnicodeEncodeError
#    **在自检跑到一半时**崩掉 —— 表现为「文档写 N/N、实测少了十几项」，
#    看着像文档写错，**真相是工具崩了**。
#    本文件 2026-10-06 加 🔴 前缀的断言时当场踩到（`info_gate` 一直没有这行）。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_BLOCK = "block"
_WARN = "warn"

# §3 预注册：输出长度上限
MAX_OUTPUT_RATIO = 1.5

#: 特征词保留率下限 —— 取产品侧同一个常量，避免两处各定一个
#:
#: 🔴 2026-10-21 开源化：原来这里硬编码了作者机器上的绝对路径。
#:    现在改为**环境变量 `GEO_CORE_PATH`**（可选）：
#:      - 设了 → 从该目录 import `geo_core.evidence_gate`，拿官方 `RETENTION_FLOOR`；
#:      - 没设 → 用内置兜底值 `0.60`，并在自检里**显式声明**这次是降级路径。
#:    「官方参考不在场」不等于「功能坏了」——自检会把它报成 [SKIP] 而非 FAIL。
GEO_CORE_PATH = os.environ.get("GEO_CORE_PATH", "").strip()


def _try_import_geo_core():
    """把 `GEO_CORE_PATH` 指向的目录加进 sys.path；返回是否成功。"""
    if not GEO_CORE_PATH or not os.path.isdir(GEO_CORE_PATH):
        return False
    if GEO_CORE_PATH not in sys.path:
        sys.path.insert(0, GEO_CORE_PATH)
    try:
        from geo_core.evidence_gate import RETENTION_FLOOR  # type: ignore
        return RETENTION_FLOOR
    except Exception:  # pragma: no cover
        return False


try:
    _ret = _try_import_geo_core()
    if _ret is False:
        raise ImportError("GEO_CORE_PATH not set or not importable")
    RETENTION_FLOOR = _ret
except Exception:  # pragma: no cover
    RETENTION_FLOOR = 0.60


# ---------------------------------------------------------------------------
# 词法
# ---------------------------------------------------------------------------
# 🔴 2026-10-06 修正：原为 r"\d+(?:[.,]\d+)*\s*%?"，其中 `\s*` 会吃掉
#    **换行**，于是 token 变成 "1519\n"，与源文本里的 "1519" 比不相等 ->
#    源文本本来就有的年份被误报成新增数字。必须用 `[ \t]*`（不含换行）。
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*[ \t]*%?")
#: 候选命名指称的词元：**不含句点**。
#: 🔴 原先的字符类里有 `.`，于是 "Karnataka. Modern" 被当成**一个**词元 ——
#    源文本与改写文本的切分不一致，`Karnataka` 就被误判成「新专名」。
#    句点只允许作为**词尾**缩写点（Dr.），且在切分后剥掉。
_PROPER_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9&\-]*\.?")
#: 句首虚词不算专名
_SENTENCE_START_SKIP = {
    "the", "a", "an", "this", "that", "these", "those", "it", "there",
    "in", "on", "at", "for", "to", "we", "you", "our", "your", "they",
    "if", "when", "while", "after", "before", "but", "and", "or", "so",
}
#: 句首/标题首的判定：前一个非空白字符是这些就算「开头位置」
_SENTENCE_LEAD = ".!?:;#\n"
_TITLE_HONORIFIC_RE = re.compile(r"\b(Dr|Prof|Mr|Mrs|Ms|Sir|Lord|St)\.?\s*$")
_HONORIFIC_RE = re.compile(r"^(dr|prof|mr|mrs|ms|sir|lord|st)\.?$")
#: 允许跨句点合并的缩写（句点不是句子边界）
_ABBREV_RE = re.compile(
    r"^(dr|prof|mr|mrs|ms|sir|lord|st|vs|etc|no|inc|ltd|co|corp|"
    r"u\.s|u\.k|approx|fig|dept|est|jr|sr|ave)\.$", re.I)


def _is_upperish(word: str) -> bool:
    """首字母大写的词（含 "Dr." 这类缩写点）。"""
    w = word.lstrip("#*->|` ")
    return bool(w) and w[0].isupper()


def _is_lead(text: str, start: int) -> bool:
    """该匹配是否位于句首/标题首（前面是标点+空白，或整篇开头）。"""
    j = start - 1
    while j >= 0 and text[j] in " \t":
        j -= 1
    if j < 0:
        return True
    return text[j] in _SENTENCE_LEAD


_MD_LINE_RE = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]|[-*+][ \t]|\d+[.)][ \t]|[*_]{1,2}[A-Za-z])")


def _on_markdown_line(text: str, start: int) -> bool:
    """该位置是否在 markdown 标题/列表行上。"""
    ls = text.rfind("\n", 0, start) + 1
    return bool(_MD_LINE_RE.match(text[ls:start + 40]))


def _proper_nouns(text: str) -> set[str]:
    """抽出**像命名指称**的专名候选。

    🔴 2026-10-06 修正两次：
    ① 原先只按「首字母大写」判定，于是**每个句子的第一个词**都被当成专名 ——
       `structure` 臂（本该由模型自造标题）几乎全被 F2 拦下。
       **门比被检对象还挑剔，等于把实验毁掉。**
    ② 词元正则里含 `.`，导致 "Karnataka. Modern" 被切成一个词元，
       源与改写切分不一致 → 假阳性。

    现在只认四种真正的命名指称：
      a) 全大写缩写（API / FDA / SPF，字母数 ≥ 2）
      b) 内部含大写（iPhone / eBay / H2O）
      c) 头衔开头的多词序列（Dr. Chen / Prof Lee）
      d) **不在**句首/标题首位置的大写词（Trustpilot / Harvard）

    纯由普通词组成的标题（`## Traditional Games of Karnataka`）不算，
    因为它不是命名指称，也不构成「新增事实」。
    """
    out: set[str] = set()
    t = text or ""
    toks = [(m.group(0), m.start(), m.end()) for m in _PROPER_RE.finditer(t)]
    i = 0
    while i < len(toks):
        word, s0, e0 = toks[i]
        bare = word.rstrip(".").lower()
        if not bare or bare in _SENTENCE_START_SKIP:
            i += 1
            continue
        # 🔴 只有「像命名指称」的词才能**起头**成组。否则 "It is FDA approved."
        #    会从 "is" 起组、把 "FDA" 并进去，产出 "is FDA" 这种垃圾实体。
        seed_ok = _is_upperish(word) or any(c.isupper() for c in word[1:])
        if not seed_ok:
            i += 1
            continue
        # 贪心并入紧邻的（允许**一个**空格）缩写点/大写词："Dr. Chen"、"Himalayan Glow"
        # 🔴 句首词**不并入**："Ask Dr. Chen" 里的 "Ask" 是句首动词，
        #    并进去会让整组失去头衔前缀，"Dr. Chen" 就检不到了。
        group = [(word, s0, e0)]
        j = i + 1
        while (not _is_lead(t, s0) and j < len(toks)
               and 0 <= toks[j][1] - toks[j - 1][2] <= 1):
            prev = toks[j - 1][0]
            # 🔴 句点后不并：那是句子边界。"Karnataka. There" 不是一个人名/地名。
            #    缩写点（Dr. / Mr. / U.S.）例外。
            if prev.endswith(".") and not _ABBREV_RE.match(prev):
                break
            if not _is_upperish(toks[j][0]):
                break
            group.append(toks[j])
            j += 1
        # 🔴 剥掉词尾缩写点：源里是 "Karnataka."、改写里是 "Karnataka"，
        #    留着点号就会切成两个不同的词元 -> 假阳性。
        phrase = " ".join(w.rstrip(".") for w, _s, _e in group).strip()
        gwords = [w.rstrip(".").lower() for w, _s, _e in group]

        if all(w.isupper() and len(w) >= 2 for w in gwords if w.isalpha()):
            out.add(phrase)                                     # (a)
        elif any(any(c.isupper() for c in w[1:]) for w, _s, _e in group if len(w) > 1):
            out.add(phrase)                                     # (b) iPhone/eBay
        elif _HONORIFIC_RE.match(bare) and len(group) > 1:
            out.add(phrase)                                     # (c) Dr. Chen
        elif (_is_upperish(word) and not _is_lead(t, s0)
              and not _on_markdown_line(t, s0)):
            # 🔴 markdown 标题/列表行内的普通大写词不算专名 ——
            #    `## Traditional Games of Karnataka` 是**结构标签**，
            #    由 treatment 臂合法生成，不是命名指称、也不构成新增事实。
            #    标题里若编造了年份/奖项，仍由 F1 / F4 拦。
            out.add(phrase)                                     # (d) 句中大写词
        i = j
    return out


#: 第三方归属：把陈述归给源文本未必提到的外部主体
_ATTRIB_RE = re.compile(
    r"(according to|recommended by|reviewed by|tested by|conducted by|"
    r"per a study|studies show|research shows|scientific studies|"
    r"as reported by|certified by|endorsed by|approved by|"
    r"根据|研究表明|实验证明|医生推荐|专家推荐)",
    re.I,
)
#: 外部引用 / 评测 / 奖项 / 背书 —— 源文本没有就不该出现
_EXTERNAL_RE = re.compile(
    r"(https?://|www\.|"
    r"\bstudy\b|\bstudies\b|\bresearch\b|\btrial\b|\bclinical trial\b|"
    r"\baward\b|\bwinner\b|\bawarded\b|\bcertified\b|\bapproved by\b|"
    r"\breview(?:s)?\b|\btestimonial(?:s)?\b|\brated\b|\bstar(?:s)?\b|"
    r"\bjournal\b|\bpublished in\b|\baccording to\b|"
    r"亚马逊|京东|天猫|小红书|知乎|论文|期刊|研究|实验|认证|获奖|评价|口碑)",
    re.I,
)
_URL_RE = re.compile(r"https?://\S+|www\.\S+|\b[\w.-]+\.(?:com|cn|net|org|io)\b", re.I)


def _norm_nums(text: str) -> set[str]:
    """把数字 token 规范化，去掉千分位逗号，便于与源文本比对。"""
    out = set()
    for m in _NUM_RE.finditer(text or ""):
        t = m.group(0).replace(",", "").replace(" ", "")
        t = t.rstrip("%")
        if t:
            out.add(t)
    return out


def _source_numeric_corpus(src: str) -> set[str]:
    return _norm_nums(src)


def _source_proper_corpus(src: str) -> set[str]:
    return {p.lower() for p in _proper_nouns(src)}


def _source_word_set(src: str) -> set[str]:
    """源文本的**词**集合（小写）。短语只要每个词都在源里，就不算新增指称。"""
    return {m.group(0).rstrip(".").lower()
            for m in _PROPER_RE.finditer(src or "")}


# ---------------------------------------------------------------------------
# 保留率：🔴 本实现**内联自 `geo_core/evidence_gate.py` (upstream reference implementation)**
#
# 为什么内联：原先这里 `import` 包外模块，导致**门通过率无法仅凭本复现包
# 独立复算** —— 而门通过率是 E4 的主结果之一（100% → 20.0% → 0.0%）。
#
# 🔴🔴 **只搬位置，不改算法。** 任何一字改动都会改变门判、重算历史 gate 决策，
#     并使已落盘的 `raw_trials` 与本包不一致。下方 `selftest` 有一条断言：
#     本实现必须与包外参考实现在同一批样本上**逐位相等**。
#     参考实现的来源与抓取时间：2026-10-05 11:25:24，570 行 / 23,905 字节。
#
# 依赖的完整清单（就这四个，其余不需要）：
#   _PUNCT_STRIP / _STOPISH / _distinctive_tokens / _norm / retention
# ---------------------------------------------------------------------------
_VENDORED_FROM = "geo_core/evidence_gate.py (upstream reference implementation)"
_VENDORED_AT = "2026-10-06"

_PUNCT_STRIP = re.compile(r"[\s，。,、；;：:！!？?（）()\[\]【】\"'“”‘’·\-—_/\\|]+")
_STOPISH = set("的了和与或是有人在就都也很更最不没这那其之以及并且但因所以")


def _norm(s: str) -> str:
    return _PUNCT_STRIP.sub("", (s or "").lower())


def _distinctive_tokens(text: str) -> set:
    """抽特征词：中文按 2-gram，英文按词。过滤高频虚词。"""
    toks: set = set()
    cjk = "".join(ch for ch in (text or "") if "一" <= ch <= "鿿")
    for i in range(len(cjk) - 1):
        g = cjk[i:i + 2]
        if not all(ch in _STOPISH for ch in g):
            toks.add(g)
    for m in re.finditer(r"[A-Za-z][A-Za-z0-9\-']{2,}", (text or "").lower()):
        toks.add(m.group(0))
    return toks


def _retention_reference(src_raw: str, rewrite: str) -> float | None:
    """保留率（词面比例）。源数据特征词 < 20 个时返回 None（不判）。"""
    src_toks = _distinctive_tokens(src_raw)
    if len(src_toks) < 20:          # 样本太小，判了也没意义
        return None
    # 🔴 归一化必须**在循环外**。原实现把已归一化的文本当**参数**传入，
    #    只算一次；内联时若写成 `t in _norm(rewrite)`，就会对**每个 token**
    #    重新归一化一遍 —— 结果完全相同（所以「逐位相等」照样通过），
    #    但 4000 字符源 × 约 2000 token × 300 篇 ≈ 10^9 次字符操作，
    #    实测从 30 s 变成 >190 s。
    #    **结果对而慢两个数量级，比算错更难被发现**，所以这里钉一条自检。
    rewrite_norm = _norm(rewrite)
    hit = sum(1 for t in src_toks if t in rewrite_norm)
    return hit / len(src_toks)


def _retention(src_raw: str, rewrite: str) -> float | None:
    return _retention_reference(src_raw, rewrite)


def _retention_external(src_raw: str, rewrite: str) -> float | None:
    """包外参考实现；**仅供自检比对**，本包运行不依赖它。

    🔴 2026-10-21 开源化：路径改为环境变量 `GEO_CORE_PATH`（可选）。
       没设或 import 不到 → 返回 None → 自检把「逐位相等」那条**跳过**并出声，
       而不是判 FAIL（缺席不是缺陷，见 README 的「缺席不等于失败」）。
    """
    try:
        if not GEO_CORE_PATH or not os.path.isdir(GEO_CORE_PATH):
            return None
        if GEO_CORE_PATH not in sys.path:
            sys.path.insert(0, GEO_CORE_PATH)
        from geo_core.evidence_gate import retention, _norm  # type: ignore
        return retention(src_raw, _norm(rewrite))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 主检查
# ---------------------------------------------------------------------------
def check(source: str, rewrite: str,
          max_ratio: float = MAX_OUTPUT_RATIO,
          retention_floor: float | None = None) -> dict:
    """对一次自由文本改写跑信息保持门。返回 dict，字段与产品侧对齐。"""
    src = source or ""
    rw = rewrite or ""
    floor = RETENTION_FLOOR if retention_floor is None else retention_floor
    findings: list[dict] = []

    def add(code, level, target, msg):
        findings.append({"code": code, "level": level,
                         "field": "text", "target": target, "message": msg})

    # F1 新增数字
    new_nums = _norm_nums(rw) - _source_numeric_corpus(src)
    # F1b 删除数字：源里有的数字在改写里整体消失（也算信息流失，WARN）
    lost_nums = _source_numeric_corpus(src) - _norm_nums(rw)
    for n in sorted(new_nums)[:20]:
        add("F1_NEW_NUMBER", _BLOCK, n,
            f"改写引入了源文本没有的数字 {n!r} —— §3 硬约束 1「不新增任何数字」。")
    if lost_nums and len(lost_nums) >= 2 and len(_source_numeric_corpus(src)) >= 3:
        add("F5_RETENTION", _WARN, ",".join(sorted(lost_nums)[:8]),
            f"源文本的 {len(lost_nums)} 个数字在改写中消失："
            f"{', '.join(sorted(lost_nums)[:8])}")

    # F2 新增专名。
    # 🔴 比较必须落在**词**上，不能整短语比：源文本里明明有 "Moctezuma II"，
    #    但源与改写的**分组方式**可能不同（源切成 "Moctezuma"、改写切成
    #    "Moctezuma II"），整短语比就误报。我们关心的是「新指称」，
    #    不是「新分组」—— 由源文本的词重新组合成的短语不算新增。
    src_words = _source_word_set(src)
    for p in sorted(_proper_nouns(rw)):
        words = [w.rstrip(".").lower() for w in p.split()]
        if any(w in src_words for w in words):
            continue
        add("F2_NEW_ENTITY", _BLOCK, p,
            f"改写引入了源文本没有的专名 {p!r} —— §3 硬约束 1/3。")

    # F3 第三方归属：源文本没出现过的归属句式即为违规
    src_attrib = {m.group(0).lower() for m in _ATTRIB_RE.finditer(src)}
    for m in _ATTRIB_RE.finditer(rw):
        if m.group(0).lower() in src_attrib:
            continue
        add("F3_ATTRIBUTION", _BLOCK, m.group(0),
            f"改写出现源文本没有的第三方归属 {m.group(0)!r} —— "
            f"§3 硬约束 2「不得把任何陈述归给第三方/机构/权威」。")

    # F4 外部引用
    src_external = {m.group(0).lower() for m in _EXTERNAL_RE.finditer(src)}
    for m in _EXTERNAL_RE.finditer(rw):
        if m.group(0).lower() in src_external:
            continue
        add("F4_EXTERNAL_REF", _BLOCK, m.group(0),
            f"改写引入外部引用/评测/奖项词 {m.group(0)!r}，源文本没有 —— "
            f"§3 硬约束 2「不得加入引用/背书/专家意见」。")
    for u in _URL_RE.finditer(rw):
        if u.group(0).lower() not in src.lower():
            add("F4_EXTERNAL_REF", _BLOCK, u.group(0),
                f"改写引入源文本没有的链接 {u.group(0)!r}。")

    # F5 保留率（复用产品侧）
    r = _retention(src, rw)
    if r is not None and r < floor:
        add("F5_RETENTION", _BLOCK, f"{r:.0%}",
            f"源文本特征词保留率仅 {r:.0%}（阈值 {floor:.0%}）—— "
            f"信息扭曲的另一半：把原有的删没了。")

    # F6 长度上限
    if src:
        ratio = len(rw) / len(src)
        if ratio > max_ratio:
            add("F6_LENGTH", _BLOCK, f"{ratio:.2f}x",
                f"改写长度是源文本的 {ratio:.2f}×，超过 §3 预注册的 {max_ratio}× 上限。")

    blocked = [f for f in findings if f["level"] == _BLOCK]
    return {
        "pass": not blocked,
        "blocked": bool(blocked),
        "findings": findings,
        "counts": {
            "block": len(blocked),
            "warn": len(findings) - len(blocked),
        },
        "codes": sorted({f["code"] for f in findings}),
        "retention": r,
        "length_ratio": (round(len(rw) / len(src), 4) if src else None),
    }


def g1g9_freetext_evidence() -> dict:
    """证据：`_geo_core.evidence_gate` 的 G1-G9 在自由文本上**静默通过**。

    存在的意义：§3 原定用 G1-G9 做信息保持门。2026-10-06 实测它在本语料上
    什么都检不出来 —— 一次明显失败的检查，比没有检查更危险。
    本函数把该证据做成**可随时重跑**的，而不是一段只能引用的结论。
    """
    src = ("The Himalayan glow serum uses 1000mg of vitamin C per bottle. "
           "It was tested by our lab in 2023. Apply 3 drops each morning. "
           "The bottle is made of recycled glass and the formula is fragrance free.")
    fabricated = (
        "Recommended by dermatologist Dr. Chen, this award-winning serum won the "
        "2024 Global Beauty Award. A clinical trial of 5000 participants found "
        "that 98% saw brighter skin in 2 weeks, and the journal published our "
        "results. Studies show it outperforms all competitors. Buy it on Amazon "
        "at https://example.com/serum and read Sara's verified review on "
        "Trustpilot. The Himalayan glow serum uses 1000mg of vitamin C per "
        "bottle. It was tested by our lab in 2023. Apply 3 drops each morning.")
    out = {"finding": "G1-G9 静默通过自由文本", "cases": {}}
    try:
        if not GEO_CORE_PATH or not os.path.isdir(GEO_CORE_PATH):
            return out
        if GEO_CORE_PATH not in sys.path:
            sys.path.insert(0, GEO_CORE_PATH)
        from geo_core import evidence_gate as G  # type: ignore
        for name, rw in (("identical", src), ("fabricated", fabricated)):
            res = G.check({"description": src}, {"description": rw})
            out["cases"][name] = {
                "pass": res["pass"], "blocked": res["blocked"],
                "codes": sorted({f["code"] for f in res.get("findings", [])}),
                "counts": res["counts"],
            }
    except Exception as exc:  # pragma: no cover
        out["error"] = f"产品侧 evidence_gate 不可用: {exc}"
        out["cases"] = {}
    if out["cases"]:
        f, i = out["cases"]["fabricated"], out["cases"]["identical"]
        out["conclusion"] = (
            f"identical pass={i['pass']} / fabricated pass={f['pass']} "
            f"(codes={f['codes']}) —— 编造了医生、5000 人试验、98% 功效、奖项、"
            f"期刊、Trustpilot 评价与外链，G1-G9 一条都没报。根因是 G1-G9 "
            f"扫结构化字段而非散文。故本语料改用 tools/info_gate.py。")
    out["repo_gate_used"] = "tools/info_gate.py"
    return out


def negative_control() -> dict:
    """阴性对照：一条刻意编造的改写**必须**被 BLOCK。

    理由与 `tautology_audit` 相同：一个只会亮红灯的检查器没有说服力。
    """
    src = ("The Himalayan glow serum contains 1000mg of vitamin C per bottle. "
           "Our lab tested it in 2023. Apply three drops each morning. "
           "The bottle is made of recycled glass and the formula is fragrance free.")
    fabricated = (
        "Recommended by dermatologist Dr. Chen, this award-winning serum won the "
        "2024 Global Beauty Award. A clinical trial of 5000 participants found "
        "that 98% saw brighter skin in 2 weeks, and the journal published our "
        "results. Studies show it outperforms all competitors. Buy it on Amazon "
        "at https://example.com/serum and read Sara's verified review on "
        "Trustpilot. The Himalayan glow serum contains 1000mg of vitamin C per "
        "bottle. Our lab tested it in 2023. Apply three drops each morning.")
    bad = check(src, fabricated)
    good = check(src, src)
    return {
        "fabricated": bad,
        "identity": good,
        "works": bool(bad["blocked"]) and bool(good["pass"]),
    }


def selftest() -> int:
    checks: list[tuple[str, bool]] = []

    def ck(name: str, cond: bool, extra: str = "") -> None:
        checks.append((name, bool(cond)))
        if not cond and extra:
            print(f"       （诊断：{extra}）")

    # ---- 🔴 内联实现必须与包外参考**逐位相等**（v1.11 补齐复现缺口）
    # 这是本条自检存在的全部理由：内联是一次**搬位置**的手术，不是重写。
    # 一旦有人顺手「优化」了它，门判就会变、已落盘的 raw_trials 就与本包不一致。
    _probe_src = (
        "The Himalayan glow serum contains 1000mg of vitamin C. Our lab tested "
        "it in 2023 with 240 participants across three clinics in Zurich and "
        "Osaka. Recommended by Dr. Chen, a clinical trial of 5000 participants "
        "found 98% brighter skin after twelve weeks of daily application. "
        "The formula uses 12% niacinamide and 3% hyaluronic acid, both of which "
        "are clinically validated for hyperpigmentation across four ethnic "
        "groups. Published in the Journal of Dermatological Research, 2024."
    )
    _probe_rw = [
        _probe_src,                                            # 逐字相同 -> 1.0
        "Serum with 1000mg vitamin C tested in 2023 on 240 participants. "
        "Recommended by Dr. Chen; a 5000-participant trial reported 98% "
        "brighter skin. Contains 12% niacinamide and 3% hyaluronic acid. "
        "Published in Journal of Dermatological Research, 2024.",
        "A glow serum for hyperpigmentation, clinically validated.",   # 大量丢失
        "产品说明。",                                                # 语言不符
    ]
    _mismatch = []
    _ref_available = False
    for _rw in _probe_rw:
        _a = _retention_reference(_probe_src, _rw)
        _b = _retention_external(_probe_src, _rw)
        if _b is not None:
            _ref_available = True
        if (_a is None) != (_b is None) or (
                _a is not None and _b is not None and abs(_a - _b) > 1e-12):
            _mismatch.append((_a, _b))
    # 🔴 参考不可用时**跳过**而不是判 FAIL —— 这正是内联的目的。
    #    实测：把包外目录藏起来跑，这条若判 FAIL 会让「本包已自洽」这个
    #    **正确结论**看起来像个 bug。`info_gate_selfcontained_check.py`
    #    会先隐藏包外模块再跑本自检，两者必须配合。
    if _ref_available:
        ck("🔴 内联保留率与包外参考逐位相等（内联只是搬位置，不是重写）",
           not _mismatch, f"不一致样本：{_mismatch[:2]}")
    else:
        print("  [SKIP] 包外参考不可用 -> 「逐位相等」这一条**跳过**"
              "（这正是内联的目的，不是缺陷）")
    ck("逐字相同的改写保留率 = 1.0",
       abs((_retention_reference(_probe_src, _probe_src) or 0) - 1.0) < 1e-12,
       str(_retention_reference(_probe_src, _probe_src)))
    ck("源特征词 < 20 时返回 None（不判）而不是 0",
       _retention_reference("short text here", _probe_rw[2]) is None,
       str(_retention_reference("short text here", _probe_rw[2])))
    ck("标点归一化生效：带标点的正文不降低保留率",
       abs((_retention_reference(_probe_src, _probe_src.replace(" ", "，"))
            or 0) - 1.0) < 1e-12)
    ck("中文 2-gram 分词可用（内联的 CJK 分支不是空壳）",
       _retention_reference(
           "该产品的临床试验由三甲医院主导，覆盖四百二十名受试者，"
           "十二周内观察到百分之九十八的肤色改善，配方含百分之十二烟酰胺。"
           "研究发表于二〇二四年皮肤科期刊，并已通过三类医疗器械备案。",
           "临床试验由三甲医院主导，受试者四百二十名，十二周内改善明显。"
           "研究发表于二〇二四年皮肤科期刊。") is not None)
    ck("包外参考不可用时内联实现仍工作（复现包自洽）",
       _retention_reference(_probe_src, _probe_src) is not None)
    ck("内联实现标注了来源与抓取时间（便于日后核对是否上游改过）",
       bool(_VENDORED_FROM) and _VENDORED_AT == "2026-10-06")

    # ---- 🔴 性能回归守卫：结果对但慢两个数量级，比算错更难被发现
    _big_src = (" ".join(f"token{i}alpha{i}beta{i}" for i in range(900)) + ". ") * 3
    _big_rw = _big_src[: int(len(_big_src) * 0.6)]
    _norm_calls = [0]
    _real_norm = _norm

    def _counting_norm(s):  # noqa: ANN001
        # 用列表而不是 global：`_norm_calls` 是 selftest() 的局部变量，
        # 闭包里的 `global` 指不到它（实测 NameError）。
        _norm_calls[0] += 1
        return _real_norm(s)

    globals()["_norm"] = _counting_norm
    try:
        _retention_reference(_big_src, _big_rw)
    finally:
        globals()["_norm"] = _real_norm
    _t0 = time.perf_counter()
    for _ in range(5):
        _retention_reference(_big_src, _big_rw)
    _dt = (time.perf_counter() - _t0) / 5
    ck(f"保留率在 900-token 源上 < 0.5 s/次（实测 {_dt * 1000:.1f} ms）",
       _dt < 0.5,
       f"疑似把 _norm() 写进了生成式内部，每个 token 归一化一次")
    ck("归一化只做一次（调用计数守卫，不靠时间阈值的运气）",
       _norm_calls[0] == 1, f"实际调用 {_norm_calls[0]} 次（1 次才对）")

    # ---- 阴性对照（最重要）
    nc = negative_control()
    ck("阴性对照：编造改写被 BLOCK", nc["fabricated"]["blocked"],
       f"codes={nc['fabricated']['codes']}")
    ck("阴性对照：逐字相同的改写通过", nc["identity"]["pass"],
       f"codes={nc['identity']['codes']}")
    ck("阴性对照整体成立", nc["works"])
    ck("阴性对照至少报出 3 类不同违规码",
       len({c for c in nc["fabricated"]["codes"] if c != "F5_RETENTION"}) >= 3,
       f"{nc['fabricated']['codes']}")

    # ---- 逐码
    src = ("The Himalayan glow serum contains 1000mg of vitamin C per bottle. "
           "Our lab tested it in 2023. Apply three drops each morning.")
    ck("F1 抓新增数字",
       "F1_NEW_NUMBER" in check(src, src + " It works for 87% of users.")["codes"])
    ck("F1 不误报源文本已有的数字",
       "F1_NEW_NUMBER" not in check(src, src + " It still has 1000mg of vitamin C.")["codes"])
    ck("F2 抓新增专名",
       "F2_NEW_ENTITY" in check(src, src + " Recommended by Dr. Mehta.")["codes"])
    ck("F2 不误报源文本已有的专名",
       "F2_NEW_ENTITY" not in check(src, "The Himalayan glow serum, tested by "
                                        "Our lab in 2023, contains 1000mg.")["codes"])
    ck("F3 抓无主语归属",
       "F3_ATTRIBUTION" in check(src, src + " According to experts it is the best.")["codes"])
    ck("F4 抓外链",
       "F4_EXTERNAL_REF" in check(src, src + " See https://shop.example.com/x.")["codes"])
    ck("F4 抓研究/评测类词",
       "F4_EXTERNAL_REF" in check(src, src + " Studies show it works.")["codes"])
    ck("F6 抓超长改写",
       "F6_LENGTH" in check(src, src * 4)["codes"])
    ck("F6 不误报正常长度改写",
       "F6_LENGTH" not in check(src, src + " Apply three drops each morning.")["codes"])

    # ---- 结构对齐产品侧
    r = check(src, src + " 5000 people tested it.")
    ck("返回字段与产品侧对齐",
       all(k in r for k in ("pass", "blocked", "findings", "counts")))
    ck("blocked 恒等于 pass 取反", r["blocked"] == (not r["pass"]))
    ck("counts 与 findings 一致",
       r["counts"]["block"] + r["counts"]["warn"] == len(r["findings"]))
    ck("codes 已去重排序", r["codes"] == sorted(set(r["codes"])))
    ck("每条 finding 五字段齐全",
       all(set(("code", "level", "field", "target", "message")) <= set(f)
           for f in r["findings"]))

    # ---- 边界不崩
    ck("空源文本不崩", check("", "anything")["blocked"] in (True, False))
    ck("空改写不崩", isinstance(check(src, ""), dict))
    ck("两端都空不崩", isinstance(check("", ""), dict))
    ck("超长输入不崩", isinstance(check("x" * 20000, "y" * 40000), dict))
    ck("无数字无专名的短文本不崩",
       isinstance(check("short text here", "short text there"), dict))

    # ---- 保留率复用产品侧
    long_src = ("Vitamin C serum with hyaluronic acid and niacinamide. "
                "Apply three drops each morning after gentle cleansing. "
                "The bottle is manufactured from recycled glass. "
                "Dermatologists recommend patch testing before full application. "
                "Ingredients include ascorbic acid, tocopherol, and ferulic acid. ") * 6
    ck("长源文本能算出保留率（不返回 None）",
       _retention(long_src, long_src) is not None)
    ck("保留率：逐字相同 = 1.0",
       abs((_retention(long_src, long_src) or 0) - 1.0) < 1e-6)
    ck("保留率：整段删掉会被 F5 拦",
       "F5_RETENTION" in check(long_src, "Buy it today. Great value.")["codes"])

    # ---- 阈值来自产品侧常量，不另定
    ck("RETENTION_FLOOR 复用产品侧（不是新数字）",
       RETENTION_FLOOR == 0.60, f"got {RETENTION_FLOOR}")
    ck("长度上限 = 1.5（§3 预注册）", MAX_OUTPUT_RATIO == 1.5)

    # ---- 🔴 F2 假阳性回归（2026-10-06 真实踩过）
    #     structure 臂**本该**由模型造标题；门比被检对象还挑剔会把实验毁掉。
    heading_src = ("Traditional games have always been a part of Karnataka's culture. "
                   "There are indoor, outdoor and board games played in Karnataka. "
                   "Modern technology changed how the young and the old play games.")
    heading_rw = ("## Traditional Games of Karnataka\n\n"
                  "### Cultural Significance\n"
                  "Traditional games have always been a part of Karnataka's rich "
                  "culture. There are a variety of indoor, outdoor and board games "
                  "that were popular in Karnataka.\n\n"
                  "### Impact of Modern Technology\n"
                  "But with the advent of computers, mobile phones and play "
                  "stations the concept of playing games have drastically changed "
                  "in the state along with the rest of the country.")
    ck("F2 不误报模型自造的普通词标题",
       "F2_NEW_ENTITY" not in check(heading_src, heading_rw)["codes"],
       f"codes={check(heading_src, heading_rw)['codes']}")
    ck("F2 不误报句首大写词",
       "F2_NEW_ENTITY" not in check(heading_src, heading_rw.replace("## ", ""))
    )
    ck("F2 仍抓全大写缩写",
       "F2_NEW_ENTITY" in check(heading_src, heading_src + " It is FDA approved.")["codes"])
    ck("F2 仍抓内部大写品牌",
       "F2_NEW_ENTITY" in check(heading_src, heading_src + " Buy on eBay today.")["codes"])
    ck("F2 仍抓头衔后人名",
       "F2_NEW_ENTITY" in check(heading_src, heading_src + " Ask Dr. Chen about it.")["codes"])
    ck("F2 仍抓句中出现的外部平台",
       "F2_NEW_ENTITY" in check(heading_src, heading_src + " Reviews on Trustpilot agree.")["codes"])
    ck("F2 不误报源文本已有的专名",
       "F2_NEW_ENTITY" not in check(heading_src, heading_src + " Karnataka remains popular.")["codes"])
    # 词级比较：源里有 "Moctezuma II"，改写里同一个短语换了分组也不该报
    aztec_src = ("Moctezuma II was the Aztec ruler when the Spanish arrived in 1519. "
                 "His empire fell to Cortes in 1521.")
    ck("F2 不误报源文本已有的多词专名（词级比较）",
       "F2_NEW_ENTITY" not in check(aztec_src, aztec_src)["codes"])
    ck("F2 仍抓源文本完全没有的名字",
       "F2_NEW_ENTITY" in check(aztec_src, aztec_src + " Ask Dr. Lopez about it.")["codes"])

    # ---- 退化改写（内容被抹掉只剩标题）必须被拦
    ck("退化改写（只剩标题无内容）被拦",
       check(heading_src, "## Release Information\n\n## Official Resources\n\n"
                          "## Production Credits\n")["blocked"],
       f"codes={check(heading_src, '## Release Information\\n\\n## Credits\\n')['codes']}")

    # ---- G1-G9 静默通过的证据可重跑（不是只能引用的结论）
    # 🔴 这几条是**唯一仍然需要包外模块**的检查，而且是**应该**需要的：
    #    它们的职责就是记录「产品侧那个 G1–G9 门在本语料上静默通过」这一
    #    **关于外部产物的证据**。产物不在场时正确做法是**大声跳过**，
    #    不是判 FAIL —— 否则「本包已自洽」这个正确结论会被报成 bug。
    #    判定标准因此要分两类：
    #      · **功能路径**（保留率、信息保持门、逐臂重算）必须零外部依赖；
    #      · **证据记录**（G1–G9 产品侧行为）按设计依赖外部产物。
    ev = g1g9_freetext_evidence()
    if ev.get("cases"):
        ck("G1-G9 证据可重跑（产品侧可达 —— 证据类检查，按设计依赖包外）",
           True)
        ck("G1-G9 对编造文本 pass=True（=静默通过，已记录为缺陷）",
           ev["cases"]["fabricated"]["pass"] is True)
        ck("G1-G9 对编造文本零 findings",
           ev["cases"]["fabricated"]["codes"] == [])
    else:
        print("  [SKIP] G1–G9 产品侧证据 —— 该模块不在场。"
              "这是**证据类**检查（记录外部产物的行为），"
              "按设计依赖包外，缺席不算缺陷。")
    ck("本模块的门在同一条编造文本上**拦得住**（对比才有说服力）",
       check(src, src + " Recommended by Dr. Chen. A clinical trial of 5000 "
                       "participants found 98% brighter skin. "
                       "See https://shop.example.com/x.")["blocked"])

    # ---- 不把语言模型当裁判（用 AST 查真实 import，不是字符串匹配 ——
    #      字符串匹配会被本测试自己写的那两个字误伤）
    import ast as _ast
    tree = _ast.parse(open(__file__, encoding="utf-8").read())
    mods = set()
    for n in _ast.walk(tree):
        if isinstance(n, _ast.Import):
            mods.update(a.name.split(".")[0] for a in n.names)
        elif isinstance(n, _ast.ImportFrom) and n.module:
            mods.add(n.module.split(".")[0])
    ck("本模块不 import 任何网络/LLM 客户端",
       not (mods & {"openai", "dashscope", "requests", "httpx", "anthropic",
                    "urllib", "socket", "http"}),
       f"imports={sorted(mods)}")
    # ---- 不从环境变量里读**凭据**（2026-10-21 开源化时收窄）
    # 🔴 原判据是「一个 os.environ 都不许有」，那是**过严**：开源化需要读一个
    #    非敏感的环境变量（GEO_CORE_PATH，指向可选的参考模块目录），
    #    结果把一条正确的改动判成了违规。
    # ⇒ 收窄成它本来要守的东西：**凭据类**变量名不许从环境变量读。
    #    用 AST 找 os.environ.get / os.getenv 的**字符串字面量参数**，
    #    只对名字像密钥的才判违规（字符串匹配会被本测试自己写的字误伤）。
    _CRED = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL",
             "AUTH", "SESSION", "COOKIE", "PRIVATE")
    env_names = []
    for n in _ast.walk(tree):
        if isinstance(n, _ast.Call):
            fn = n.func
            is_env = ((isinstance(fn, _ast.Attribute) and fn.attr in ("get", "getenv")
                       and isinstance(fn.value, _ast.Attribute)
                       and fn.value.attr == "environ")
                      or (isinstance(fn, _ast.Attribute) and fn.attr == "getenv"
                          and isinstance(fn.value, _ast.Name)
                          and fn.value.id == "os"))
            if is_env and n.args and isinstance(n.args[0], _ast.Constant) \
                    and isinstance(n.args[0].value, str):
                env_names.append(n.args[0].value)
        elif isinstance(n, _ast.Subscript):
            sl = n.slice
            if isinstance(sl, _ast.Constant) and isinstance(sl.value, str) \
                    and isinstance(n.value, _ast.Attribute) \
                    and n.value.attr == "environ":
                env_names.append(sl.value)
    bad_env = [e for e in env_names
               if any(k in e.upper() for k in _CRED)]
    ck("本模块不从环境变量读凭据（路径类变量允许）",
       not bad_env,
       f"凭据类环境变量={bad_env}" if bad_env else
       f"读到的环境变量={sorted(set(env_names))}")

    failed = [n for n, ok in checks if not ok]
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\n{len(checks) - len(failed)} PASS   {len(failed)} FAIL")
    return 0 if not failed else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="info_gate.py",
        description="自由文本改写的信息保持门（GEO-bench 语料；G1-G9 在此语料静默通过）")
    ap.add_argument("--source-file")
    ap.add_argument("--rewrite-file")
    ap.add_argument("--max-ratio", type=float, default=MAX_OUTPUT_RATIO)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--evidence-g1g9", action="store_true",
                    help="重跑「G1-G9 在自由文本上静默通过」的证据")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.evidence_g1g9:
        ev = g1g9_freetext_evidence()
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "outputs", "gate_g1g9_freetext_evidence.json")
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(ev, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(ev, ensure_ascii=False, indent=2))
        print(f"\n-> {os.path.normpath(out)}")
        return 0
    if a.source_file and a.rewrite_file:
        s = open(a.source_file, encoding="utf-8-sig").read()
        r = open(a.rewrite_file, encoding="utf-8-sig").read()
        res = check(s, r, a.max_ratio)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0 if res["pass"] else 1
    if a.source_file and not a.rewrite_file:
        res = negative_control()
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0 if res["works"] else 1
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
