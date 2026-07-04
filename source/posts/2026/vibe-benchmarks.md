---
title: Vibe Benchmarks
date: 2026-07-04
tags: ["LLM", "Coding", "Benchmarks"]
---

## Vibe Benchmarks

_July 2026_

A quick round of informal benchmarking across a few local coding models.
Nothing rigorous — just a few problems run through each model and seeing
how they handled it. The goal was to get a sense of the trade-offs between
quality, speed, and practical usability.

### The models

- **Qwen Coder Next** — the best quality by a fair margin. It understood
  the problem, produced clean code, and generally got it right on the first
  try. The problem is that it's too large and too slow. The latency was
  noticeable, and the larger context windows it supports actually work
  against it — with more context comes more tokens to process, and the
  slowdown compounds.
- **Qwen 35B MoE** — the best on balance. It matched Qwen Coder Next on
  most problems, was noticeably faster, and didn't suffer from the same
  context-window bloat. For practical daily use, this is the sweet spot.
- **Cohere North Mini** — extremely fast. Almost instant responses. But
  it failed to solve several problems that Qwen 35B solved after one or
  two turns. North Mini kept going in circles — same wrong approach,
  repeated, unable to course-correct. Speed is great when it works, but
  not much use if it can't actually solve the problem.
- **Qwen 27B** — too slow for my taste. I'd need to try it again under
  different conditions before forming a firm opinion. It had decent
  quality but the latency was a real drag.
- **Gemma MoE** — okay. Nothing wrong with it, but Qwen models were
  consistently higher quality. Gemma felt like it was trying its best
  but falling short on the harder problems.

### TL;DR

Qwen Coder Next is the best. Qwen 35B MoE is the best on balance. Cohere
North Mini is extremely fast but unreliable on actual problem-solving.
Gemma MoE is fine but Qwen wins on quality. Qwen 27B needs another shot.
