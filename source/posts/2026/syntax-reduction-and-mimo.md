---
title: On Syntax-Guided Program Reduction
date: 2026-06-01
tags: ["Python", "LLM", "program-reduction"]
---

Program reduction is an interesting problem: given an objective (e.g., a unit test), can we automatically create the minimal viable program by deleting unnecessary code?

This repo ([nappe](https://github.com/NoRaincheck/nappe)) and [PyPI package](https://pypi.org/project/nappe/) is an implementation of an existing approach with extensions to solve this. As this problem is NP-hard, it's not surprising that it is a tad 'expensive' to do this well. However, it is interesting as it tackles things like removing dead code which perhaps tools like `ruff` don't succeed with.

On the implementation side, this was also an opportunity to explore the MiMo coding agent (free limits) and overall it has worked as advertised (probably Sonnet level) with very generous free limits. I've only had one instance where it fell into an infinite repeat thinking loop. For problems with a well-defined and testable input/output, these setups are perfect for re-implementing research papers and having it in a state where it can be used.
