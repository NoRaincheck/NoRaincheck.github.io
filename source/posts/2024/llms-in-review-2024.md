---
title: LLMs - in Review (2024)
date: 2024-12-01
tags: ["LLM", "self-hosting", "production", "Go", "ML", "embeddings", "ONNX"]
---

## LLMs - in Review (2024)

_December 2024_

2024 was the first year where I took LLMs seriously. I successfully hosted a
Llama 70b parameter model in production which was used as with
[continue.dev](https://www.continue.dev/) for a self-hosted co-pilot
replacement, along with a code autocomplete like
[Qwen Coder](https://qwenlm.github.io/blog/qwen2.5-coder-family/) or
[Deepseek](https://deepseekcoder.github.io/), these were fine replacements and
surprisingly robust.
[Huggingface's TGI](https://huggingface.co/docs/text-generation-inference/index)
along with [Triton Server](https://github.com/triton-inference-server/server)
were the main heroes for this project, (Triton was used to serve `onnx` models
for embeddings) though I've yet to find a "good" embedding model. At this stage
in time, most of the vector database solutions "feel" the same and can all
seemingly be trivially hosted via Kubernetes.

Tools used:

- Huggingface TGI (LLMs)
- Triton Inference Server (embeddings/`onnx` models)

Optimising usage of LLMs at scale is still a massive challenge, particularly
when opening up for general access. The space is still new so there are still
lots of patterns especially related to agentic patterns which need to be
explored. On that particular note,
[this particular blog post from Anthropic](https://www.anthropic.com/research/building-effective-agents)
is something I'm looking at working on in the new year. At the same time, I've
been experimenting with [mlflow](https://mlflow.org/docs/latest/llms/index.html)
for LLM evaluations which has worked reasonably well, though I've pretty
disappointed at [mlflow](https://mlflow.org/docs/latest/llms/index.html) from a
self-hosting perspective, especially if only a tracking server is required.
Maybe this is an open source project for the future.
