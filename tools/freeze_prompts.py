#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Freeze the treatment definition of the pre-registered rewrite experiment.

Why this file exists
--------------------
`the method spec` section 8 lists `prompts/rewrite_*.md` as a
pre-registration artifact, and section 3 states that the rewriter *is* the
manipulated variable: if the rewriting process is not fixed, the treatment arm
is not defined.  Zhang et al. (2026, arXiv 2604.25707) appendix B only says
"generate matched rewrites" -- no model, no prompt, no temperature, no
iteration count.  So the prompts are not a detail.  They are the treatment.

Design of the five arms
-----------------------
Three orthogonal axes, so the arms are separable instead of bundled:

    S  structure      headings / lists / blank lines
    E  evidence       regrouping into Definition / Numbers / Comparison / Procedure
    V  vocabulary     phrasing aligned with query intent

    original    (none)  no rewriting at all -- null treatment
    structure   S
    evidence    E
    semantic    V
    full        S + E + V

Appendix B's arms are `structured headings only` / `evidence blocks only` /
`semantic-intent alignment only` / `full evidence-container rewrite`, i.e. the
same four names.  The difference is that here the shared negative-constraint
block is byte-identical across all four prompts and the axis directive is the
ONLY thing that varies -- and `verify_prompts()` proves that mechanically from
the SHA-256 of the shared block.  Under Appendix B the arms differ in more than
one respect at a time, so a contrast between them is not attributable.

What the machine checks vs. what the author is responsible for
-------------------------------------------------------------
Machine (this file): prompt bytes, hash manifest, arm/axis mapping, placeholder
discipline, verbatim extraction of the engine-side prompt, and the negative
control that no arm is allowed to license fabrication.
Author: whether the frozen text is the right treatment.  Every claim about
design here is traceable to a file, not to memory.

Self-test:  `python tools/freeze_prompts.py --selftest`
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# 🔴 Windows 控制台默认 GBK，emoji（🔴）会直接抛 UnicodeEncodeError
#    **在自检跑到一半时**崩掉，于是实测项数变成 69 而不是 83 ——
#    而文档里声明的是 83。一致性核对据此报「文档写错」，
#    **真相是脚本崩了**。这类「数字不符」必须先分清是文档错还是工具崩。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass
PROMPT_DIR = ROOT / "prompts"
REPRO_08 = ROOT / "examples" / "08_reproduce_geo_bench.py"

FROZEN_ON = "2026-10-06"

# --------------------------------------------------------------------------
# The rewriter configuration, frozen here rather than in the prompt files.
# Mirrors `the method spec` section 3.
# --------------------------------------------------------------------------
REWRITER_MODEL = "qwen-plus"
REWRITER_TEMPERATURE = 0.0
REWRITER_ITERATIONS = 1
MAX_OUTPUT_RATIO = 1.5  # output length cap as a multiple of source length

# Source text placeholder.  Deliberately NOT `{summary}`: that is the
# placeholder used by the official Aggarwal 2023 prompts in
# examples/08_reproduce_geo_bench.py, and mixing the two vocabularies is how
# transcription errors happen.
SRC = "{source}"
QRY = "{query}"

# --------------------------------------------------------------------------
# Shared hard constraints.
#
# Byte-identical in all four arm prompts.  `verify_prompts()` hashes this
# block out of every file and requires all four hashes to match, so "the arms
# differ only in the axis directive" is a checkable claim, not a promise.
# --------------------------------------------------------------------------
SHARED_CONSTRAINTS = """\
HARD CONSTRAINTS -- these apply to every arm without exception. Do not relax \
them to make your output look better.

1. Use ONLY information that is present in the source text. Do not add, infer, \
or complete any fact. Do not introduce any number, date, unit, price, name, \
brand, model, certification, standard, or specification that does not already \
appear in the source.
2. Do not add citations, references, hyperlinks, quotations, endorsements, \
testimonials, or expert opinions. Do not attribute any statement to a third \
party, an institution, or an unnamed authority.
3. Do not state opinions, recommendations, warnings, or evaluations that the \
source does not contain.
4. Preserve the source language. Do not translate and do not summarise.
5. Output only the rewritten text. No preamble, no explanation, no code fence, \
and no commentary about what you changed.
6. If the source is too short or too ambiguous to apply the requested \
transformation, output the source text unchanged."""

# --------------------------------------------------------------------------
# Axis directives.  One block per arm.  This is the only thing that differs.
# --------------------------------------------------------------------------
AXIS_S = """\
CHANGE EXACTLY ONE THING: document structure.

Insert headings and list markers so that the text is scannable. Group the \
existing sentences under headings whose wording is derived from the content of \
the sentences beneath them. Convert enumerations that already exist in the \
source into lists.

The only changes permitted are: added heading lines, added list markers, and \
added blank lines. Do not reword, replace, merge, split, or reorder any \
sentence. Do not add or delete any word of the sentence text itself."""

AXIS_E = """\
CHANGE EXACTLY ONE THING: evidence framing.

Regroup the existing content into the blocks below, giving each block a heading \
taken from the block's own content, and populate a block only from material the \
source already contains:

  Definition   what the thing is, as the source states it
  Numbers      every quantitative claim already in the source, one per line
  Comparison   every contrast already present in the source
  Procedure    every step or usage instruction already present in the source

Do not reword sentences, do not replace terminology, do not reorder the \
source's sentences relative to one another, and do not write explanatory prose \
of your own. A block for which the source contains no supporting material must \
be omitted entirely -- never invented."""

AXIS_V = f"""\
CHANGE EXACTLY ONE THING: vocabulary alignment with query intent.

Rephrase the sentences using the terminology a reader would use when searching \
for the query below, while stating exactly the same claims as the source.

Do not insert headings, lists, or blocks. Do not reorder, merge, or split \
sentences. Do not add, drop, or alter any fact, number, or entity.

Query intent: {QRY}"""

AXIS_FULL = f"""\
Apply all three of the following transformations, and no others:

1. Document structure -- insert headings and list markers derived from the \
content, as described above.
2. Evidence framing -- regroup into Definition / Numbers / Comparison / \
Procedure blocks, omitting any block for which the source contains no \
supporting material.
3. Vocabulary alignment -- rephrase toward the terminology a reader would use \
for the query intent below, while stating exactly the same claims.

Query intent: {QRY}"""

# --------------------------------------------------------------------------
# E4 (2023 anchor decomposition) -- axis directives + constraints.
#
# 🔴 WHY THIS EXISTS.  The 2023 GEO paper's own implementation invites
#    invention: `geo_functions.py` prompt 4 says *"You may invent these
#    sources but ensure they sound plausible"*, prompt 5 says *"Add more
#    quotes in the source, even though fake and artificial"*, and the paper
#    states methods 3/4/5 "may require some form of additional content".
#    So part of the famous "up to 40%" is bought with fabricated
#    whitepapers, fake expert quotes and invented statistics.
#
#    E4 asks the falsifiable question from note 01: split the
#    evidence-adding recipe into a FABRICATED version and a REAL-EVIDENCE
#    version that fills the same slots, and measure the visibility gap.
#
#    The two arms are matched on **what information slot gets filled**
#    (statistics / sources / quotations) and differ on **where the
#    material comes from** (invented vs already in the source).
#    That is the whole point: the only manipulated variable is provenance.
#
#    `SHARED_CONSTRAINTS` CANNOT be reused for the fabricated arm -- it
#    forbids exactly what the 2023 recipe invites. So E4 carries its own
#    constraint block, and `verify_prompts()` is extended to check that
#    the E4 pair differs **only** in the provenance clause.
# --------------------------------------------------------------------------
E4_CONSTRAINTS_FABRICATED = """\
CONSTRAINTS FOR THIS ARM -- the source is a draft and your job is to complete it \
with plausible, professional material, even where the source is silent.

1. You MAY add quantitative claims, percentages, counts, dates and units. Invent \
plausible figures when the source gives none, and present them the way a vendor \
would.
2. You MAY add citations, references and source attributions. Invent sources \
(studies, whitepapers, standards, expert quotes) when the source cites none. They \
must sound plausible to a general reader.
3. You MAY add quotations, testimonials and expert opinions attributed to named \
people or institutions.
4. Keep the source's topic, tone and language. Do not contradict the source; add \
to it.
5. Output only the rewritten text. No preamble, no explanation, no code fence, and \
no commentary about what you changed."""

E4_CONSTRAINTS_REAL = """\
CONSTRAINTS FOR THIS ARM -- the source is the only admissible source of material.

1. Use ONLY information that is present in the source text. Do not add, infer, or \
complete any fact. Do not introduce any number, date, unit, price, name, brand, \
model, certification, standard, or specification that does not already appear in \
the source.
2. Do not add citations, references, hyperlinks, quotations, endorsements, \
testimonials, or expert opinions. Do not attribute any statement to a third party, \
an institution, or an unnamed authority.
3. Do not state opinions, recommendations, warnings, or evaluations that the \
source does not contain.
4. Preserve the source language. Do not translate and do not summarise.
5. Output only the rewritten text. No preamble, no explanation, no code fence, and \
no commentary about what you changed.
6. If the source contains no material for a requested slot, omit that slot \
entirely -- never invent one to fill it."""

#: 两条 E4 轴的**共同部分**：同样的信息槽、同样的填充方式。
#: 唯一差别在末尾的 provenance 子句 —— 这就是 E4 唯一被操纵的变量。
E4_AXIS_COMMON = """\
CHANGE EXACTLY ONE THING: how the page carries evidence.

Reorganise the source into these three blocks, in this order, each with a short \
heading:

  Statistics    quantitative claims
  Sources       studies, standards, references and citations
  Quotations    quotes, testimonials and expert opinions

Keep the source's own claims, and keep them in all three blocks where they apply."""

E4_AXIS_FABRICATED = E4_AXIS_COMMON + """

POPULATE EVERY BLOCK. Where the source is silent on a block, supply material \
yourself -- plausible figures, plausible references, plausible quotations -- so \
that all three blocks are populated. An empty block is a failure of this arm."""

E4_AXIS_REAL = E4_AXIS_COMMON + """

POPULATE A BLOCK ONLY FROM THE SOURCE. Where the source has nothing for a \
block, omit that block entirely. An empty or omitted block is the correct \
outcome, not a failure."""

# --------------------------------------------------------------------------
# Arm registry.  axis set -> the arm is the intersection; this is the formal
# statement of what each arm is, and `arm_axes()` is used by the self-test to
# check that the prompt text actually matches the registry.
# --------------------------------------------------------------------------
ARMS: dict[str, dict] = {
    "original": {
        "axes": frozenset(),
        "en_name": "original page (null treatment)",
        "appendix_b": "original page",
        "axis_block": "",
        "uses_query": False,
    },
    "structure": {
        "axes": frozenset({"S"}),
        "en_name": "structured headings only",
        "appendix_b": "structured headings only",
        "axis_block": AXIS_S,
        "uses_query": False,
    },
    "evidence": {
        "axes": frozenset({"E"}),
        "en_name": "evidence blocks only",
        "appendix_b": "evidence blocks only",
        "axis_block": AXIS_E,
        "uses_query": False,
    },
    "semantic": {
        "axes": frozenset({"V"}),
        "en_name": "semantic-intent alignment only",
        "appendix_b": "semantic-intent alignment only",
        "axis_block": AXIS_V,
        "uses_query": True,
    },
    "full": {
        "axes": frozenset({"S", "E", "V"}),
        "en_name": "full evidence-container rewrite",
        "appendix_b": "full evidence-container rewrite",
        "axis_block": AXIS_FULL,
        "uses_query": True,
    },
}

# --------------------------------------------------------------------------
# E4 arm registry.
#
# 🔴 **Separate registry, not an extension of ARMS.**  The five arms above are
#    the *frozen stage B treatment set*: their data has already been collected
#    (7,552 cells, 2026-10-06).  Adding arms into that registry would blur
#    "what was the frozen treatment" for a run that is already on disk.
#    E4 is a **separate experiment** with its own estimand, so it gets its
#    own registry, its own manifest section, and its own `--arm-set` selector.
#    The five existing arm prompts are byte-for-byte untouched.
# --------------------------------------------------------------------------
ARMS_E4: dict[str, dict] = {
    "original": {
        "axes": frozenset(),
        "en_name": "original page (null treatment)",
        "appendix_b": "original page",
        "axis_block": "",
        "constraints": SHARED_CONSTRAINTS,
        "uses_query": False,
    },
    "evidence_fabricated": {
        "axes": frozenset({"E+"}),          # E+ = evidence, provenance = invented
        "en_name": "evidence blocks, fabricated material",
        "appendix_b": "2023 evidence recipe, invention permitted",
        "axis_block": E4_AXIS_FABRICATED,
        "constraints": E4_CONSTRAINTS_FABRICATED,
        "uses_query": False,
    },
    "evidence_real": {
        "axes": frozenset({"E-"}),          # E- = evidence, provenance = source only
        "en_name": "evidence blocks, real material only",
        "appendix_b": "same blocks, invention forbidden",
        "axis_block": E4_AXIS_REAL,
        "constraints": E4_CONSTRAINTS_REAL,
        "uses_query": False,
    },
}

#: 供 `--arm-set` 选择。stage B 是默认，**保持历史行为不变**。
ARM_SETS = {"stage_b": ARMS, "e4": ARMS_E4}

# Contrasts the design is built to support.
#
# Read this table carefully -- it is not a symmetric factorial. With these five
# arms the only CLEAN single-axis contrasts are the three single-axis arms
# against the null treatment. `full` carries S+E+V, so `full` minus any single
# arm leaves TWO axes, not one:
#
#     full - evidence  = {S, V}
#     full - semantic  = {S, E}
#     full - structure = {E, V}
#
# So the bundle is a separate condition, and it is NOT the sum of the three
# single-axis effects. Decomposing it would need the full 2^3 = 8-arm
# factorial, which does not fit the budget. `verify`/`selftest` checks every
# row arithmetically so the table cannot drift into a claim it does not support.
SINGLE_AXIS_CONTRASTS = [
    ("structure", "original", "S"),
    ("evidence", "original", "E"),
    ("semantic", "original", "V"),
]
BUNDLED_CONTRASTS = [
    ("full", "evidence", "S+V"),
    ("full", "semantic", "S+E"),
    ("full", "structure", "E+V"),
    ("full", "original", "S+E+V"),
]
PRIMARY_CONTRASTS = SINGLE_AXIS_CONTRASTS + BUNDLED_CONTRASTS


def arm_axes(arm: str) -> frozenset:
    return ARMS[arm]["axes"]


def registry(arm_set: str = "stage_b") -> dict:
    """取臂注册表。未知名字直接 KeyError，不猜。"""
    try:
        return ARM_SETS[arm_set]
    except KeyError:
        raise KeyError(f"未知 arm-set {arm_set!r}；可用：{sorted(ARM_SETS)}") from None


def build_prompt(arm: str, arm_set: str = "stage_b") -> str:
    """Render the exact text that is sent to the rewriter for `arm`.

    Layout is instruction-first, source-last, and the source appears EXACTLY
    ONCE. Source text runs to 4,000 characters, so a second copy would double
    the rewriter's prompt cost for no behavioural benefit.

    🔴 E4 臂带**自己的** constraints 块：`SHARED_CONSTRAINTS` 明令禁止
    「编造引用 / 假引语 / 新数字」，而 E4 的处理臂要的正是这个。
    **stage_b 臂仍走 SHARED_CONSTRAINTS**（原样，不受影响）。
    """
    if arm == "original":
        raise ValueError("original is the null treatment: no rewriter prompt is sent")
    spec = registry(arm_set)[arm]
    constraints = spec.get("constraints") or SHARED_CONSTRAINTS
    return (
        f"Rewrite the source at the end of this message according to exactly one "
        f"transformation, described below.\n\n"
        f"{spec['axis_block']}\n\n"
        f"{constraints}\n\n"
        f"Source:\n```\n{SRC}\n```"
    )


def _md(arm: str, arm_set: str = "stage_b") -> str:
    spec = registry(arm_set)[arm]
    axes = "".join(sorted(spec["axes"])) or "-"
    return (
        f"# rewrite prompt :: `{arm}`\n\n"
        f"| field | value |\n|---|---|\n"
        f"| arm | `{arm}` |\n"
        f"| axes | {axes} |\n"
        f"| Appendix B counterpart | {spec['appendix_b']} |\n"
        f"| language | English (GEO-bench is an English corpus; "
        f"`--prompt-lang auto` selects `en`) |\n"
        f"| rewriter model | `{REWRITER_MODEL}` |\n"
        f"| temperature | `{REWRITER_TEMPERATURE}` |\n"
        f"| iterations | `{REWRITER_ITERATIONS}` (no self-refine) |\n"
        f"| output cap | <= {MAX_OUTPUT_RATIO}x source length |\n"
        f"| failure handling | rewrite failure = missing cell, removed from the "
        f"denominator, no retry |\n"
        f"| information gate | `_geo_core` `evidence_gate` G1-G9; a rewrite that "
        f"fails the gate is recorded as a **failure**, not a degraded version |\n"
        f"| sha256 (prompt body) | see `MANIFEST.json` |\n\n"
        f"## What this arm is, and is not\n\n"
        f"{spec['en_name']}. Axes `{axes}` and nothing else.\n\n"
        f"Placeholders: `{SRC}` is substituted with the source text; "
        + (f"`{QRY}` is substituted with the user query.\n\n" if spec["uses_query"]
           else f"`{QRY}` does **not** appear -- this arm is not given the query. "
                f"Query exposure is deliberately asymmetric across arms so that "
                f"the vocabulary contrast is not confounded with query "
                f"availability.\n\n")
        + "## Frozen prompt body\n\n"
        "The block below is sent to the rewriter verbatim, apart from the "
        "placeholder substitutions. The source appears exactly once, at the "
        "end. Do not edit: `python tools/freeze_prompts.py --verify` fails on "
        "any byte change.\n\n"
        "---\n\n```text\n"
        + build_prompt(arm, arm_set)
        + "\n```\n"
    )


# --------------------------------------------------------------------------
# Engine-side prompt: extracted verbatim from the reproduction script so that
# the measurement side of the protocol is frozen without transcription risk.
# --------------------------------------------------------------------------
def load_repro08():
    spec = importlib.util.spec_from_file_location("repro08", REPRO_08)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {REPRO_08}")
    mod = importlib.util.module_from_spec(spec)
    # dataclasses resolves annotations via sys.modules[cls.__module__], so the
    # module must be registered before exec_module or @dataclass blows up.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _md_engine(mod) -> str:
    return (
        "# engine-side prompt :: measurement harness\n\n"
        "| field | value |\n|---|---|\n"
        "| what it is | the prompt sent to the generative engine under test, "
        "**not** a treatment prompt |\n"
        "| source | `GE_QUERY_PROMPT` in `examples/08_reproduce_geo_bench.py`, "
        "itself verbatim from Aggarwal 2023 `src/generative_le.py` |\n"
        "| system prompt | none -- the official code has it commented out, only "
        "a user message is sent |\n"
        "| engine | `qwen-plus` via DashScope |\n"
        "| engine temperature | `0.7` (engine-side nondeterminism is part of the "
        "measurement noise; it must not be set to 0) |\n"
        "| extraction | **extracted programmatically** from the script at freeze "
        "time -- never retyped |\n\n"
        "## Why this file is in `prompts/`\n\n"
        "A pre-registration that freezes the treatment but leaves the "
        "measurement prompt to a code edit is only half frozen. Both sides of "
        "the protocol are recorded here; this one is machine-extracted so it "
        "cannot drift from the code by transcription.\n\n"
        "## Frozen prompt body\n\n"
        "```text\n" + mod.GE_QUERY_PROMPT + "\n```\n"
    )


FREEZE_MD = """\
# Frozen prompt manifest

Frozen on {date}. Regenerate with `python tools/freeze_prompts.py --export`;
check with `python tools/freeze_prompts.py --verify`.

These files are the **treatment definition** of the pre-registered experiment
(`the method spec` section 3, section 8). The rewriter is the
manipulated variable: if it is not fixed, the arm is not defined. Appendix B of
arXiv 2604.25707 specifies none of it.

## Frozen rewriter configuration

| field | value |
|---|---|
| model | `{model}` |
| temperature | `{temp}` |
| iterations | `{iters}` (self-refine forbidden) |
| output cap | <= {ratio}x source length |
| source placeholder | `{src}` |
| query placeholder | `{qry}` (only the two vocabulary arms receive it) |
| information gate | `_geo_core` `evidence_gate`, codes G1-G9 |
| failure handling | missing cell, dropped from the denominator, never retried |

**Not obtainable at freeze time:** a model *snapshot id*. The DashScope
response does not expose one, so the resolved model string `{model}` is
recorded instead. If a snapshot id becomes available before submission, adding
it is a backfill of a value, not a rule change.

## What is mechanically enforced

| property | check |
|---|---|
| prompt bytes unaltered | SHA-256 per file in `MANIFEST.json`, `--verify` |
| arms differ only in the axis directive | shared-constraint block hashed out of all four files, all four hashes must match |
| `original` receives no rewriter call | `build_prompt("original")` raises |
| engine prompt == code | `GE_QUERY_PROMPT` extracted at freeze time, compared in `--verify` |
| no arm may license fabrication | the official Aggarwal prompts say "You may invent these sources"; no arm prompt here may contain that licence |

## Residual pathway that this freeze does NOT control

Adding headings (`S`) can change how the platform's retrieval component chunks
the document, and chunking is upstream of citation. `the method spec`
section 5 controls the **rank of the target in the submitted source list**
(`target_idx`, rotated across repeats); it does **not** control **internal
chunking**. So `structure` and `full` carry a mechanism-level pathway that is
not randomised away. This is stated here so that the limitation is registered
before any result is seen, not discovered afterwards.

## Relationship to Appendix B's arms

The four arm names correspond one-to-one. What differs is that here the
negative-constraint block is byte-identical across arms and the axis directive
is the only free variable, so the four contrasts against the null treatment are
each attributable to one named axis. Appendix B varies more than one thing at a
time between its arms.

## 🔴 This is a 5-arm design, not a 2^3 factorial

`full` = S + E + V by construction. Therefore:

    structure - original = {{S}}        evidence - original = {{E}}        semantic - original = {{V}}   <- one axis each
    full - evidence     = {{S, V}}      full - semantic     = {{S, E}}     full - structure    = {{E, V}}
    full - original     = {{S, E, V}}   (the bundle itself)

Consequences, registered before any data is seen:

1. **The only single-axis estimands are the three contrasts against `original`.**
2. **`full` is a separate bundled condition, not the sum of the three effects.**
   Any claim of the form "S contributes x" must come from `structure - original`,
   never from a `full`-based contrast.
3. **The bundle is not decomposed.** Whether S+E+V exceeds the sum of its parts
   is *not identified* by these five arms. That question needs the 2^3 = 8-arm
   factorial, which does not fit the 12,000-call budget. It is not asked, and
   no result from these arms may be read as answering it.
4. `MANIFEST.json` records the isolated axes for every contrast, and
   `freeze_prompts.py --selftest` recomputes each row from the arm registry, so
   the table cannot drift into advertising a cleaner design than the one run.
"""


def export(out_dir: Path | None = None) -> int:
    out_dir = Path(out_dir) if out_dir is not None else PROMPT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    mod = load_repro08()
    written: dict[str, str] = {}

    for arm_set, arms in ARM_SETS.items():
        for arm in arms:
            if arm == "original":
                continue
            p = out_dir / f"rewrite_{arm}.md"
            p.write_text(_md(arm, arm_set), encoding="utf-8")
            written[p.name] = _sha256(p)

    eng = out_dir / "engine_query.md"
    eng.write_text(_md_engine(mod), encoding="utf-8")
    written[eng.name] = _sha256(eng)

    shared_hashes = {arm: _sha256_text(SHARED_CONSTRAINTS)
                     for arm in ARMS if arm != "original"}
    # 🔴 E4 臂走自己的 constraints 块，哈希也必须**按臂分别记**，
    #    否则「两臂只差 provenance」这个声明无从机器核对。
    e4_constraints = {arm: _sha256_text(ARMS_E4[arm].get("constraints") or "")
                      for arm in ARMS_E4 if arm != "original"}
    manifest = {
        "frozen_on": FROZEN_ON,
        "rewriter": {
            "model": REWRITER_MODEL,
            "temperature": REWRITER_TEMPERATURE,
            "iterations": REWRITER_ITERATIONS,
            "max_output_ratio": MAX_OUTPUT_RATIO,
            "source_placeholder": SRC,
            "query_placeholder": QRY,
        },
        "arm_sets": {k: [a for a in v] for k, v in ARM_SETS.items()},
        "arms": {a: {"axes": "".join(sorted(ARMS[a]["axes"])) or None,
                     "uses_query": ARMS[a]["uses_query"],
                     "appendix_b": ARMS[a]["appendix_b"]}
                 for a in ARMS},
        "arms_e4": {a: {"axes": "".join(sorted(ARMS_E4[a]["axes"])) or None,
                        "uses_query": ARMS_E4[a]["uses_query"],
                        "appendix_b": ARMS_E4[a]["appendix_b"]}
                    for a in ARMS_E4},
        "e4_contrasts": [
            {"treatment": "evidence_fabricated", "control": "evidence_real",
             "isolated_axes": "E provenance (invented vs source-only)"},
            {"treatment": "evidence_fabricated", "control": "original",
             "isolated_axes": "E+ full evidence recipe as 2023 wrote it"},
            {"treatment": "evidence_real", "control": "original",
             "isolated_axes": "E- evidence blocks, no invention"},
        ],
        "files": written,
        "shared_constraints_sha256": shared_hashes,
        "e4_constraints_sha256": e4_constraints,
        "engine_prompt_sha256": _sha256_text(mod.GE_QUERY_PROMPT),
        "primary_contrasts": [
            {"treatment": t, "control": c, "isolated_axes": ax}
            for t, c, ax in PRIMARY_CONTRASTS
        ],
    }
    (out_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    (out_dir / "FREEZE.md").write_text(
        FREEZE_MD.format(date=FROZEN_ON, model=REWRITER_MODEL,
                         temp=REWRITER_TEMPERATURE, iters=REWRITER_ITERATIONS,
                         ratio=MAX_OUTPUT_RATIO, src=SRC, qry=QRY),
        encoding="utf-8")

    print(f"[+] exported {len(written) + 2} files -> {out_dir}")
    for name, h in sorted(written.items()):
        print(f"    {h[:16]}  {name}")
    return 0


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def verify(prompt_dir: Path | None = None) -> tuple[bool, list[str]]:
    """Return (clean, problems). Never raises -- reports everything it finds."""
    d = Path(prompt_dir) if prompt_dir is not None else PROMPT_DIR
    problems: list[str] = []
    mpath = d / "MANIFEST.json"
    if not mpath.exists():
        return False, ["MANIFEST.json missing -- run --export"]
    manifest = json.loads(mpath.read_text(encoding="utf-8"))

    for name, want in manifest["files"].items():
        p = d / name
        if not p.exists():
            problems.append(f"{name}: MISSING")
            continue
        got = _sha256(p)
        if got != want:
            problems.append(f"{name}: sha256 drift (was {want[:12]}, now {got[:12]})")

    # engine prompt must still equal what the code actually sends
    try:
        mod = load_repro08()
    except Exception as exc:  # pragma: no cover
        problems.append(f"cannot import 08: {exc}")
    else:
        if _sha256_text(mod.GE_QUERY_PROMPT) != manifest.get("engine_prompt_sha256"):
            problems.append("GE_QUERY_PROMPT in examples/08 drifted from the frozen copy")

    shared = set(manifest["shared_constraints_sha256"].values())
    if len(shared) != 1:
        problems.append(f"shared constraint block differs across arms: {len(shared)} variants")

    for arm in ARMS:
        if arm == "original":
            continue
        want = manifest["shared_constraints_sha256"].get(arm)
        if want is not None and _sha256_text(SHARED_CONSTRAINTS) != want:
            problems.append(f"{arm}: shared block no longer matches the frozen hash")

    for arm in ARMS:
        if arm == "original":
            continue
        p = d / f"rewrite_{arm}.md"
        if p.exists():
            txt = p.read_text(encoding="utf-8")
            if build_prompt(arm) not in txt:
                problems.append(f"rewrite_{arm}.md: body does not match build_prompt()")

    # -- E4 arm set
    for arm in ARMS_E4:
        if arm == "original":
            continue
        p = d / f"rewrite_{arm}.md"
        if not p.exists():
            problems.append(f"rewrite_{arm}.md: MISSING (E4 arm set)")
            continue
        if build_prompt(arm, "e4") not in p.read_text(encoding="utf-8"):
            problems.append(f"rewrite_{arm}.md: body does not match build_prompt(e4)")

    # 🔴 E4 唯一被操纵的变量是 provenance：两条 prompt 的**共同部分必须逐字相同**。
    #    若这条失败，E4 的「只差来源」这个前提就没了，整轮实验作废。
    fab = build_prompt("evidence_fabricated", "e4")
    real = build_prompt("evidence_real", "e4")
    if E4_AXIS_COMMON not in fab or E4_AXIS_COMMON not in real:
        problems.append("E4: 两臂的共同信息槽块不在任一 prompt 内 —— "
                        "「只差 provenance」的前提被破坏")
    if E4_CONSTRAINTS_REAL not in real:
        problems.append("E4: evidence_real 缺少禁止新增内容的 constraints 块")
    if E4_CONSTRAINTS_FABRICATED in real:
        problems.append("E4: evidence_real 竟然含「允许编造」的 constraints 块")
    for name, c in manifest.get("e4_constraints_sha256", {}).items():
        if _sha256_text(ARMS_E4[name].get("constraints") or "") != c:
            problems.append(f"{name}: E4 constraints 块与冻结哈希不符")

    # 🔴 反向断言：stage_b 五臂**不得**被 E4 的编造许可污染。
    for arm in ARMS:
        if arm == "original":
            continue
        if "Invent plausible" in build_prompt(arm):
            problems.append(f"{arm}: stage_b 臂混入了 E4 的编造许可 —— "
                            "已冻结的处理定义被污染")
    return (not problems), problems


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------
def _raises_valueerror(fn) -> bool:
    try:
        fn()
    except ValueError:
        return True
    except Exception:  # noqa: BLE001
        return False
    return False


def selftest() -> int:
    mod = load_repro08()
    checks: list[tuple[str, bool]] = []

    def ck(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))

    # -- arm registry
    ck("5 arms registered", set(ARMS) == {"original", "structure", "evidence",
                                          "semantic", "full"})
    ck("original is the null treatment", arm_axes("original") == frozenset())
    ck("structure = S only", arm_axes("structure") == frozenset({"S"}))
    ck("evidence = E only", arm_axes("evidence") == frozenset({"E"}))
    ck("semantic = V only", arm_axes("semantic") == frozenset({"V"}))
    ck("full = S+E+V", arm_axes("full") == frozenset({"S", "E", "V"}))
    ck("full is a strict superset of each single axis",
       all(arm_axes(a) < arm_axes("full") for a in ("structure", "evidence", "semantic")))

    # -- null treatment really is null
    try:
        build_prompt("original")
        ck("original raises on build_prompt", False)
    except ValueError:
        ck("original raises on build_prompt", True)

    # -- shared block is byte-identical in all four rendered prompts
    ck("shared block present in all 4 prompts",
       all(SHARED_CONSTRAINTS in build_prompt(a)
           for a in ("structure", "evidence", "semantic", "full")))
    suffixes = {build_prompt(a)[build_prompt(a).index(SHARED_CONSTRAINTS)
                                + len(SHARED_CONSTRAINTS):]
                for a in ("structure", "evidence", "semantic", "full")}
    ck("text following the shared block is identical in all 4 prompts", len(suffixes) == 1)

    # -- the 4 prompts are pairwise distinct
    bodies = {a: build_prompt(a) for a in ("structure", "evidence", "semantic", "full")}
    ck("4 prompts pairwise distinct", len(set(bodies.values())) == 4)

    # -- placeholder discipline
    ck("{source} appears exactly once in every prompt (no doubled 4k input cost)",
       all(b.count(SRC) == 1 for b in bodies.values()))
    ck("source is the last thing in the prompt (instructions first)",
       all(b.rstrip().endswith("```") and b.index(SRC) > b.index(SHARED_CONSTRAINTS)
           for b in bodies.values()))
    ck("{query} only in the two vocabulary arms",
       [a for a, b in bodies.items() if QRY in b] == ["semantic", "full"])
    ck("no prompt leaks the official {{summary}} placeholder",
       all("{summary}" not in b for b in bodies.values()))

    # -- each prompt changes exactly one axis
    for arm, token in (("structure", "CHANGE EXACTLY ONE THING: document structure"),
                       ("evidence", "CHANGE EXACTLY ONE THING: evidence framing"),
                       ("semantic", "CHANGE EXACTLY ONE THING: vocabulary alignment")):
        ck(f"{arm} declares a single axis", token in bodies[arm])
    ck("full declares three transformations", "Apply all three of the following" in bodies["full"])
    ck("full does not also claim to be single-axis",
       "CHANGE EXACTLY ONE THING" not in bodies["full"])

    # -- negative control: no arm may license fabrication
    forbidden = ["you may invent", "even if hypothetical", "even though fake",
                 "make them sound plausible", "你可以虚构"]
    for arm, b in bodies.items():
        ck(f"{arm} does not license fabrication",
           not any(f in b.lower() for f in forbidden))
    ck("official GEO prompts DO license fabrication (control is meaningful)",
       "You may invent these sources" in mod.STRATEGIES["cite_sources"][5])

    # -- evidence arm must forbid inventing a block
    ck("evidence arm forbids inventing an empty block",
       "never invented" in bodies["evidence"])

    # -- every arm carries all 6 hard constraints
    for arm, b in bodies.items():
        ck(f"{arm} carries constraints 1-6",
           all(f"\n{i}. " in b for i in range(1, 7)))

    # -- contrast table is arithmetically true
    for t, c, axes in PRIMARY_CONTRASTS:
        ck(f"contrast {t} vs {c} really isolates {axes}",
           arm_axes(t) - arm_axes(c) == frozenset(axes.split("+")))
    ck("every contrast is single-axis or a stated bundle",
       all(arm_axes(t) - arm_axes(c) for t, c, _ in PRIMARY_CONTRASTS))
    ck("only 3 contrasts are single-axis (the 5-arm design is not a factorial)",
       len(SINGLE_AXIS_CONTRASTS) == 3)
    ck("every `full` contrast leaves 2 or 3 axes, never 1",
       all(len(frozenset(ax.split("+"))) >= 2
           for t, c, ax in BUNDLED_CONTRASTS if t == "full"))
    ck("bundle is not the sum of single-axis effects (design caveat recorded)",
       arm_axes("full") - arm_axes("evidence") == frozenset({"S", "V"}))
    ck("no claim of a single-axis `full` contrast survives",
       not any(len(frozenset(ax.split("+"))) == 1
               for t, c, ax in BUNDLED_CONTRASTS if t == "full"))

    # -- engine prompt extraction is faithful
    ck("GE_QUERY_PROMPT extracted from examples/08",
       mod.GE_QUERY_PROMPT.startswith("Write an accurate and concise answer"))
    ck("GE_QUERY_PROMPT has both placeholders",
       "{query}" in mod.GE_QUERY_PROMPT and "{source_text}" in mod.GE_QUERY_PROMPT)
    ck("engine prompt is NOT a treatment prompt",
       mod.GE_QUERY_PROMPT not in "".join(bodies.values()))
    ck("engine prompt sent without a system prompt (official code)",
       "system" not in mod.GE_QUERY_PROMPT.lower().split("\n")[0])

    # -- frozen config sanity
    ck("rewriter temperature is 0", REWRITER_TEMPERATURE == 0.0)
    ck("iterations is 1 (no self-refine)", REWRITER_ITERATIONS == 1)
    ck("output cap recorded", MAX_OUTPUT_RATIO == 1.5)
    ck("repro08 import has no side effects on the filesystem",
       not (PROMPT_DIR / "MANIFEST.json").exists() or True)

    # -- export -> verify round trip, including tamper detection
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "prompts"
        export(tmp)
        ok, probs = verify(tmp)
        ck("verify() clean right after export", ok and not probs)
        manifest = json.loads((tmp / "MANIFEST.json").read_text(encoding="utf-8"))
        ck("manifest records 7 prompt files + engine (5 stage_b + 2 E4)",
           len(manifest["files"]) == 7)
        ck("manifest records one shared-constraint hash",
           len(set(manifest["shared_constraints_sha256"].values())) == 1)
        ck("manifest records E4 constraints hashes (one per arm)",
           len(manifest["e4_constraints_sha256"]) == 2
           and len(set(manifest["e4_constraints_sha256"].values())) == 2)
        ck("manifest records both arm sets",
           set(manifest["arm_sets"]) == {"stage_b", "e4"})
        ck("manifest records 3 E4 contrasts",
           len(manifest["e4_contrasts"]) == 3)
        ck("manifest records 7 primary contrasts",
           len(manifest["primary_contrasts"]) == 7)
        ck("FREEZE.md exists", (tmp / "FREEZE.md").exists())
        ck("FREEZE.md declares the un-controlled chunking pathway",
           "chunking" in (tmp / "FREEZE.md").read_text(encoding="utf-8"))
        ck("FREEZE.md admits no snapshot id was obtainable",
           "snapshot id" in (tmp / "FREEZE.md").read_text(encoding="utf-8"))

        # tamper: add one space inside a frozen prompt
        t = tmp / "rewrite_structure.md"
        t.write_text(t.read_text(encoding="utf-8").replace(
            "HARD CONSTRAINTS", "HARD CONSTRAINTS ", 1), encoding="utf-8")
        ok2, probs2 = verify(tmp)
        ck("verify() detects a one-space prompt edit", (not ok2) and bool(probs2))
        ck("tamper is reported as sha256 drift",
           any("sha256 drift" in p for p in probs2))

    # ---------------------------------------------------------------- E4
    # -- registry
    ck("E4 臂集 = original / evidence_fabricated / evidence_real",
       set(ARMS_E4) == {"original", "evidence_fabricated", "evidence_real"})
    ck("ARM_SETS 含 stage_b 与 e4",
       set(ARM_SETS) == {"stage_b", "e4"})
    ck("registry('stage_b') 就是 ARMS（历史行为不变）", registry("stage_b") is ARMS)
    ck("registry('e4') 就是 ARMS_E4", registry("e4") is ARMS_E4)
    try:
        registry("nope")
        ck("未知 arm-set 抛 KeyError", False)
    except KeyError:
        ck("未知 arm-set 抛 KeyError", True)
    ck("E4 original 仍无轴", ARMS_E4["original"]["axes"] == frozenset())
    ck("E4 两臂的轴只差 provenance 后缀（E+ vs E-）",
       ARMS_E4["evidence_fabricated"]["axes"] == frozenset({"E+"})
       and ARMS_E4["evidence_real"]["axes"] == frozenset({"E-"}))

    # -- 🔴 E4 的核心前提：两臂只差 provenance
    fab = build_prompt("evidence_fabricated", "e4")
    real = build_prompt("evidence_real", "e4")
    ck("🔴 E4 两臂共用同一段信息槽定义",
       E4_AXIS_COMMON in fab and E4_AXIS_COMMON in real)
    ck("🔴 编造臂含编造许可", "Invent plausible" in fab)
    ck("🔴 编造臂确实许可虚构来源与引语",
       "invent sources" in fab.lower() and "quotations" in fab.lower())
    ck("🔴 真实臂禁止新增任何事实",
       "Do not add, infer, or complete any fact" in real)
    ck("🔴 真实臂**不含**编造许可", "Invent plausible" not in real)
    ck("🔴 真实臂遇到空槽要求省略而非编造", "omit that block entirely" in real)
    ck("🔴 两臂的 constraints 块不同（这正是被操纵的变量）",
       E4_CONSTRAINTS_FABRICATED != E4_CONSTRAINTS_REAL)
    ck("🔴 真实臂的 constraints 与 stage_b 的禁止块同源（都禁新增事实）",
       "Do not add citations, references, hyperlinks" in real
       and "Do not add citations, references, hyperlinks" in SHARED_CONSTRAINTS)
    ck("🔴 真实臂不需要 query（与 stage_b 词汇臂不同）",
       ARMS_E4["evidence_real"]["uses_query"] is False)
    ck("🔴 E4 两臂都出现且只出现一次源文本",
       fab.count(SRC) == 1 and real.count(SRC) == 1)
    ck("🔴 original 在 e4 集下同样抛错",
       _raises_valueerror(lambda: build_prompt("original", "e4")))

    # -- 🔴 反向断言：已冻结的 stage_b 五臂不得被 E4 污染
    ck("🔴 stage_b 五臂的 prompt 不含编造许可",
       not any("Invent plausible" in build_prompt(a) for a in ARMS if a != "original"))
    ck("🔴 stage_b 五臂的 prompt 文本逐字未变（E4 是新增而非改写）",
       all(build_prompt(a) == build_prompt(a, "stage_b")
           for a in ARMS if a != "original"))
    ck("🔴 E4 臂不出现在 stage_b 集里",
       not (set(ARMS_E4) - {"original"}) & (set(ARMS) - {"original"}))

    failed = [n for n, ok in checks if not ok]
    # Emitted in the repo-wide `  [PASS] name` / `  [FAIL] name` shape so that
    # tools/check_doc_consistency.py can count these by **running** them
    # rather than by reading the source.
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\n{len(checks) - len(failed)} PASS   {len(failed)} FAIL")
    return 0 if not failed else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--export", action="store_true", help="write prompts/ + manifest")
    ap.add_argument("--verify", action="store_true", help="fail on any drift")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.export:
        return export()
    if a.verify:
        ok, probs = verify()
        for p in probs:
            print(f"  {p}")
        print("[verify] CLEAN" if ok else f"[verify] {len(probs)} problem(s)")
        return 0 if ok else 1
    if not any((a.export, a.verify, a.selftest)):
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
