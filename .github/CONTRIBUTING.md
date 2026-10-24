# Contributing to geo-measure-toolkit

This repository is **measurement and judgment tooling** for GEO experiments: power /
MDE inversion, six statistical lint rules, a leakage audit, prompt freezing, feature
extraction, and an information-preservation gate. It ships **no data** and no
dependencies beyond the Python standard library.

Everything here is built around one idea: **a result that the design could never have
produced should be detectable, not merely unlikely.** A contribution is welcome when
it makes some failure mode *falsifiable* that was not falsifiable before.

---

## The contract a PR must keep green

A pull request is not done until all four of these pass locally:

```bash
python run_selftests.py                  # 17 PASS  0 FAIL
python tools/build_manifest.py --check   # registry matches reality
python scripts/check_deps.py             # dependency closure complete
python scripts/check_release.py          # CITATION / license / badge consistency
```

CI runs the same checks on Python 3.9 / 3.11 / 3.13 / 3.14.

**Why `build_manifest.py --check` is part of the contract:** `MANIFEST.json` holds each
tool's measured self-test count and sha256. Those numbers are **measured, never
hand-written** — the builder imports the *same* `count_checks()` implementation the
runner uses, so the packaging side and the checking side cannot drift apart. If you
touch a tool, regenerate the registry (`python tools/build_manifest.py`) and commit the
result. Do not edit `MANIFEST.json` by hand.

---

## Before you file a bug: classify the red

A red line is not one thing. Reporting "it's broken" when the tool is fine sends
everyone to fix the wrong thing. There are four states, and the runner already
distinguishes them:

| State | What the tool looks like | Runner says | What to do |
|---|---|---|---|
| **Precondition missing** | prints a finish line declaring the precondition is absent | `SKIP` | fix the environment, or use the tool elsewhere |
| **Declared absence** | runs fewer checks than registered, but printed `[SKIP]` explaining why | `SKIP` | fix the environment (e.g. provide the optional reference module) |
| **Silent loss of checks** | runs fewer checks and says *nothing* | `FAIL` | **fix the tool** |
| **Crashed** | never printed a finish line | `FAIL` | **fix the tool** |

"Evidence is not present" and "the function is missing" are different findings. When
you file an issue, say which state you are in — the issue template asks for it.

Two more rules that keep a report actionable:

* **Paste the finish line, not a wall of output.** The finish line is the tool's own
  verdict: how many checks it ran, how many failed, and why it stopped. Everything
  else is context.
* **A wrong number is a bug; a missing input is not.** If a tool needs your own
  experiment output, it says so and exits with an error. That is the contract working.

---

## Adding a tool

1. Put it under `tools/` (a reusable kernel) or `examples/` (an analysis driver).
2. Give it a self-test entry that **prints a finish line** with the check count. The
   runner and the manifest builder both count from that line; without it, the tool
   cannot be registered.
3. Register it in `tools/build_manifest.py` (`TOOLS`) with an honest `scope`:
   * `standalone` — the self-test runs with no project data, no network, and no
     out-of-package module. `build_manifest.py` **fails the build** if a tool
     declared `standalone` does not actually come back green.
   * `needs-project` — it needs this project's `outputs/`. It will be skipped
     **loudly** in this repo (neither PASS nor FAIL).
   * `none` — no self-test entry at all. This is a **verification gap**; it will be
     flagged in `MANIFEST.json` and must be confirmed by hand.
4. Run `python tools/build_manifest.py` and commit the regenerated `MANIFEST.json`.
5. Run `python scripts/check_deps.py`. If your tool loads another module dynamically
   with `importlib.util.spec_from_file_location`, the closure audit will find it —
   but write the path so the audit can resolve it.

---

## House style

* **English** for code, docstrings, comments, commit messages and documentation.
* **No third-party packages** for the core tools. The standard library is not a
  limitation to work around; it is what lets someone read the implementation and
  check the arithmetic line by line. If a tool genuinely needs a dependency, it goes
  under `examples/` and says so in its `--help`.
* **Every statistic is hand-written.** Same reason.
* **Windows consoles are GBK.** Printing `✅` / `⚠` / `❌` / `🔴` / `⇒` / `²` crashes
  with `UnicodeEncodeError` on a default Chinese-locale console. Every script in this
  repo opens with:

  ```python
  try:
      sys.stdout.reconfigure(errors="replace")
  except Exception:
      pass
  ```

  Do not change the console code page — that turns the whole Chinese output into
  mojibake. "Replace what cannot be encoded" is the correct trade.
* **Read and write files as UTF-8 explicitly.** `open(..., encoding="utf-8")` on both
  sides; the locale encoding is not a file format.
* **Commit messages**: one line, imperative, prefixed by area where it helps —
  `fix:`, `docs:`, `chore:`. Explain *why* in the body when the why is not obvious
  from the diff.

---

## Data policy

**Do not commit datasets.** This repository ships code and a registry, nothing else.
If a tool needs data, it takes a path or a URL and says where it came from — see
"Tools that need data you have to supply" in the README.

If you contribute a script that reads a public dataset, record the source (URL,
licence, retrieval date) in the script's docstring. A dataset that cannot be
re-fetched is not reproducible, only remembered.

---

## One source of truth

Do not copy these tools into another repository, a skill directory, or a vendored
folder. A second copy is a second source of truth, and the drift between them is
**silent**: the MDE in your paper was computed by one copy while the reviewer's run
uses the other, and nothing reports the disagreement.

Point at this repository instead:

```bash
git clone https://github.com/242742532Celestia/geo-measure-toolkit.git
```

If you need a tool to live somewhere else, open an issue and say why — the answer may
be a published package rather than a copy.

---

## What a good PR looks like

* It states which failure mode becomes **detectable** that was not detectable before.
* It keeps the four commands above green, and shows the output.
* If it changes a tool's behaviour, it changes the manifest with it.
* If it changes a rule, a threshold or a default, it says so in the commit body —
  those are pre-registration decisions, not implementation details.
* If it touches the optional external reference module, it keeps the
  "declared absence" path intact: absent module → loud `SKIP`, never silent success.
