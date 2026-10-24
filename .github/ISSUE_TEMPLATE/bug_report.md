---
name: Tool or documentation bug
description: A tool behaves incorrectly, or the documentation disagrees with the code
title: "[bug] "
labels: bug
body:
  - type: markdown
    attributes:
      value: |
        Thanks for taking the time. Two notes before you fill this in:

        1. **Paste the finish line, not a wall of output.** The finish line is the
           tool's own verdict (how many checks it ran, how many failed, why it
           stopped). Everything else is context.
        2. **"Evidence is not present" and "the function is missing" are different
           findings.** Please pick the state below — it decides whether anyone
           should change code or change their environment.

  - type: textarea
    id: tool
    attributes:
      label: Which tool, and the exact command
      description: For example `python tools/statpower.py mde-r --n 112`
      placeholder: python tools/statpower.py mde-r --n 112
    validations:
      required: true

  - type: textarea
    id: finishline
    attributes:
      label: The finish line
      description: The last line the tool printed — check count, fail count, and any `[SKIP]` reason.
      render: text
      placeholder: "all checks passed (33 checks, 0 failures)"
    validations:
      required: true

  - type: dropdown
    id: state
    attributes:
      label: Which state is this?
      description: See "The four judgment states" in the README.
      options:
        - Precondition missing — the tool declared that something is absent and exited cleanly
        - Declared absence — it ran fewer checks than registered and printed `[SKIP]` explaining why
        - Silent loss of checks — it ran fewer checks and said nothing (this is the dangerous one)
        - Crashed — it never printed a finish line
        - Not a self-test problem — the tool ran fine but the result is wrong
    validations:
      required: true

  - type: textarea
    id: expected
    attributes:
      label: What did you expect, and what happened instead?
      description: If a number is wrong, give the number you got, the number you expected, and how you computed the expected one.
    validations:
      required: true

  - type: textarea
    id: data
    attributes:
      label: Did you supply the data yourself?
      description: |
        Some tools read your own experiment output or a public dataset. Say which
        path, dataset or URL you used — and if you fell back to a built-in demo
        or synthetic data, say that too. A demo run is not an experiment.
      placeholder: "`--dataset geobench`, cached at data/geo-bench/ …"
    validations:
      required: false

  - type: input
    id: env
    attributes:
      label: Python version and OS
      placeholder: 3.14.7 on Windows 10 (cp936 console)
    validations:
      required: true

  - type: textarea
    id: repochecks
    attributes:
      label: Do the repository's own checks pass for you?
      description: |
        ```
        python run_selftests.py
        python tools/build_manifest.py --check
        python scripts/check_deps.py
        python scripts/check_release.py
        ```
        Paste the summary line of each. If one of them fails, that may be the
        actual bug — and "which one and why" is worth more than a description.
      render: text
    validations:
      required: false

  - type: textarea
    id: docs
    attributes:
      label: If the documentation is what disagrees with the code
      description: Quote the README / SKILL.md line and say what the code actually does.
    validations:
      required: false
