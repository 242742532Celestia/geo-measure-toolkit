# geo-measure-toolkit

[![selftest](https://github.com/242742532Celestia/geo-measure-toolkit/actions/workflows/selftest.yml/badge.svg)](https://github.com/242742532Celestia/geo-measure-toolkit/actions/workflows/selftest.yml)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Measurement and judgment tooling for **Generative Engine Optimization (GEO)** experiments —
power/MDE inversion, statistical lint rules, tautology auditing, prompt freezing,
feature extraction, and an information-preservation gate.

Everything here is **pure standard library**. No numpy, no scipy, no matplotlib,
no network access required for the core self-test.

```bash
git clone <this-repo>
cd geo-measure-toolkit
python run_selftests.py      # -> 17 PASS  0 FAIL
```

That is the whole install. The tools are scripts, not a library.

---

## Why this exists

GEO experiments fail in a specific, repeatable way: **a result is reported that the
design could never have produced.** Not because the analysis was sloppy, but because
the design quietly could not detect the effect it claimed to find.

This toolkit is built around making that failure **detectable and checkable**:

| Problem | Tool | What it does |
|---|---|---|
| "Is this effect size even reportable?" | `statpower.py` | Inverts power/MDE — given the number a paper reported, computes the smallest effect it could actually have detected |
| "Does this null stand up?" | `geo_stat_lint.py` | Six lint rules (L1–L6) over a filled-in result card |
| "Did the treatment write the answer into the outcome?" | `tautology_audit.py` | Leakage score L(T,Y) with four severity bands |
| "Can I rebuild the exact same arms later?" | `freeze_prompts.py` | Snapshot + sha256 of every prompt, plus `--verify` |
| "Is this extractor secretly producing a score?" | `geo_page_features.py` | Candidate features only — a self-test recursively scans the output keys to guarantee no score/weight/grade ever appears |
| "Did the rewrite fabricate facts?" | `info_gate.py` | Information-preservation gate G1–G9 over free text |

Plus the analysis drivers under `examples/` that apply them to real runs.

---

## The four judgment states (read this before trusting a red line)

The self-test runner does **not** collapse everything into pass/fail. A tool can be
perfectly healthy and still not run — and confusing those cases sends you to fix
the wrong thing.

| State | What the tool looks like | Runner says | What you should do |
|---|---|---|---|
| **Precondition missing** | prints a finish line declaring `前置缺失` / precondition missing | `SKIP` | fix the environment, or use it elsewhere |
| **Declared absence** | runs fewer checks than registered, but printed `[SKIP]` explaining why | `SKIP` | fix the environment (e.g. provide the optional reference module) |
| **Silent loss of checks** | runs fewer checks and says *nothing* | `FAIL` | **fix the tool** |
| **Crashed** | never printed a finish line | `FAIL` | **fix the tool** |

"Evidence is not present" and "the function is missing" are different findings and
must never be reported as the same thing. The runner enforces this distinction
mechanically: a tool-level skip is never counted as a failure, and the failure
counter is cross-checked against the printed failure lines before the runner exits.

Running **more** checks than registered is also not a failure — it means the registry
is stale; re-run `python tools/build_manifest.py` to refresh it.

---

## Layout

```
run_selftests.py              # self-test entry point (also the CI entry point)
MANIFEST.json                 # machine-readable registry: measured counts + sha256
tools/
  statpower.py                # power / MDE inversion kernel
  geo_stat_lint.py            # lint rules L1-L6
  tautology_audit.py          # leakage score L(T,Y)
  freeze_prompts.py           # prompt freezing + --verify
  geo_page_features.py        # candidate feature extractor (no scores)
  info_gate.py                # information-preservation gate G1-G9
  info_gate_selfcontained_check.py
  build_manifest.py           # regenerates MANIFEST.json by measurement
examples/
  08_reproduce_geo_bench.py   # reproduce the official GEO-bench harness
  10_citation_noise.py        # run-to-run noise on the citation-rate outcome
  11_construct_validity.py    # construct validity
  12_probe_site_facts.py      # probe site-level facts
  13_infrastructure_survey.py # infrastructure layer vs method layer
  14_stage_a_variance.py      # variance decomposition sigma_b / sigma_e
  17_extend_geobench_cache.py # extend the query cache without changing the filter
  19_paired_analysis.py       # paired analysis (Holm correction)
  20_rewriter_noise.py        # noise introduced by a rewriting system
  23_e4_sequential.py         # sequential stopping rules S1/S2/S3
  24_e4_noise_floor.py        # variance decomposition + arithmetic lower bound on MDE
scripts/
  check_deps.py               # dependency-closure audit (see below)
```

### Why there are 18 tool files but 15 tools

`MANIFEST.json` marks **15** as independently runnable. Three more are shipped because
other tools **dynamically load** them:

```
examples/23_e4_sequential.py  ->  examples/19_paired_analysis.py  ->  examples/08_reproduce_geo_bench.py
examples/24_e4_noise_floor.py ->  examples/23_e4_sequential.py
tools/freeze_prompts.py       ->  examples/08_reproduce_geo_bench.py
examples/17_extend_geobench_cache.py -> examples/08_reproduce_geo_bench.py
examples/14_stage_a_variance.py -> examples/08_reproduce_geo_bench.py
```

These are loaded with `importlib.util.spec_from_file_location`, so **no static import
statement mentions them**. A dependency scan that only reads `import` lines misses
them entirely — which is exactly how they went missing once already.

`scripts/check_deps.py` closes that hole by scanning every `.py` **string literal**
that looks like a path, not just the import statements:

```bash
python scripts/check_deps.py
# -> ✅ dependency closure complete
```

---

## Optional external reference module

`info_gate.py` can cross-check its own retention implementation against an upstream
reference implementation. That module is **optional**:

```bash
export GEO_CORE_PATH=/path/to/geo_core        # optional
```

* **not set** → the gate uses its built-in fallback constant, and the cross-check
  assertions are reported as `[SKIP]` (declared absence, not a failure);
* **set** → the cross-check runs and the self-test grows from 49 to 53 assertions.

The self-test count therefore depends on the environment. That is intentional and
handled: the runner accepts *more* checks than registered, and fewer checks are only
a failure when the tool did **not** declare the absence.

Other optional inputs, all environment variables with CLI overrides:

| Variable | Used by | Purpose |
|---|---|---|
| `GEO_CORE_PATH` | `info_gate.py`, `11_construct_validity.py`, `info_gate_selfcontained_check.py` | optional upstream reference module |
| `GEO_RAW_TRIALS_CSV` | `10_citation_noise.py` | per-cell trial results |
| `GEO_SITE_FACTS_JSON` | `13_infrastructure_survey.py` | probed site facts |
| `GEO_REMEASURE_JSON` | `11_construct_validity.py` | re-measurement payload |

### `info_gate_selfcontained_check.py` has a side effect

It **temporarily renames** the directory `GEO_CORE_PATH` points at, re-runs the key
paths with the external module hidden, then renames it back. That is the only way to
*prove* "does not depend on the external module" with executable evidence — a docstring
claiming independence proves nothing.

Do not kill the process halfway: the rename and the restore both live in a `finally`
block, and a hard kill leaves the directory renamed. If that happens, look for
`<name>.__hidden_by_selfcheck` next to the original.

---

## Tools that need data you have to supply

### Synthetic-data fallback paths

Two tools need inputs that only exist inside a real project. Rather than shipping
them unverified, both have a **synthetic-data path** that exercises the same code
without the real artefacts — and both **say so out loud** when they fall back:

* `examples/14_stage_a_variance.py` — the "4b per-arm MDE" render path needs a gate
  report. When the real one is absent it renders against a synthetic gate report
  (three arms with deliberately spread pass rates) and prints
  `[SKIP] 真实门报告不在场 … 合成数据只证明代码路径，不证明真实通过率`.
  The synthetic data proves the **code path**; the real pass rates still have to come
  from real data.
* `examples/19_paired_analysis.py` — reads an optional query-exclusion registry
  (`GEO_QUERY_EXCLUSIONS`, default `data/query_exclusions.json`). When that registry
  is absent, the four assertions that would cross-check it are **skipped with a
  declaration** rather than failed, so the tool self-verifies either way.

One tool has **no self-test entry at all**: `examples/08_reproduce_geo_bench.py`.
It is a verification gap and is flagged as such in `MANIFEST.json` — confirm it by
hand before relying on it.

---

## Keeping the registry honest

`MANIFEST.json` holds each tool's measured self-test count and file hash. Those
numbers are **measured, never hand-written** — `tools/build_manifest.py` runs every
tool and imports the *same* `count_checks()` implementation the runner uses, so the
packaging side and the checking side cannot drift apart.

```bash
python tools/build_manifest.py           # regenerate MANIFEST.json
python tools/build_manifest.py --check   # verify the registry matches reality
```

`--check` fails when a file's hash no longer matches the registry, or when a tool
silently lost checks.

---

## Requirements

* Python 3.9+ (developed and tested on 3.14; CI runs 3.9 / 3.11 / 3.13 / 3.14)
* No third-party packages for the core tools
* Network access only for `examples/12_probe_site_facts.py` (site probing) and
  `examples/17_extend_geobench_cache.py` (downloading the official benchmark cache)

## Citing

If you use this toolkit in published work, please cite it — the metadata lives in
[`CITATION.cff`](CITATION.cff):

```bibtex
@software{geo-measure-toolkit,
  title  = {geo-measure-toolkit},
  author = {一左一右},
  year   = {2026},
  version = {1.0.0},
  license = {MIT}
}
```

### Before you publish this repo

The GitHub account is the one thing this repository cannot know. It is left as a
single placeholder (`242742532Celestia`) in `CITATION.cff` and in the README badge, so
it can be filled in exactly once, in exactly one place:

```bash
python scripts/set_repo_url.py --owner YourGitHubHandle
python scripts/check_release.py     # must pass
```

`check_release.py` verifies that the citation metadata, the licence, the registry
version and the badge all point at the same repository — and that the link to the
**official** GEO-bench repository has not been collaterally edited.

## License

MIT — see [LICENSE](LICENSE).
