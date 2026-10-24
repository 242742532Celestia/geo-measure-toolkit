# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| `v1.0.0` (current) | ✅ |

The repository is versioned with the tool registry: `MANIFEST.json` records the
sha256 of every tool file, so a given version is exactly reproducible.

## Reporting a vulnerability

**Do not open a public issue for a security problem.**

Email **242742532@qq.com** with:

* what the problem is, and the exact file and line if you have it;
* the input or command that triggers it;
* what you expected and what happened instead;
* your Python version and operating system.

You will get an acknowledgement within a few days. If a fix is warranted, it will be
committed, the version bumped, and the advisory disclosed together — the reporter is
credited unless you prefer otherwise.

## What counts as a security issue here

This repository ships **analysis scripts**, not a service. There is no server, no
authentication, no database. So the bar is specific:

**In scope**

* Code execution or arbitrary file access triggered by input a tool processes
  (e.g. a crafted dataset file or a path argument).
* Path traversal or unsafe writes into the user's working tree.
* A dependency-closure hole that pulls in unverified code.
* Anything that makes a tool execute a command it should only have described.

**Out of scope**

* **A wrong or misleading number.** That is a correctness bug, not a security
  vulnerability — and it is a serious one. Please file it as an issue: the point of
  this repository is that such things become detectable, and a report helps.
* Missing input, absent project data, or an unmet precondition. The tools report
  these as `SKIP` with a declared reason; that is the designed behaviour.
* Findings from running the tool on data you do not control, without a concrete
  exploit path.
* Vulnerabilities in Python itself, in the standard library, or in the optional
  external reference module — report those upstream.

## A note on the optional external reference module

`info_gate.py` can cross-check its retention implementation against an upstream
module pointed at by `GEO_CORE_PATH`. That module is **optional** and is not
distributed here. Only point `GEO_CORE_PATH` at code you trust; treat it as you
would any third-party dependency.
