# 2024

## LLMs - in Review (2024)

> December 2024

2024 was the first year where I took LLMs seriously. I successfully hosted a Llama 70b parameter model in production which was used as with [continue.dev](https://www.continue.dev/) for a self-hosted co-pilot replacement, along with a code autocomplete like [Qwen Coder](https://qwenlm.github.io/blog/qwen2.5-coder-family/) or [Deepseek](https://deepseekcoder.github.io/), these were fine replacements and surprisingly robust. [Huggingface's TGI](https://huggingface.co/docs/text-generation-inference/index) along with [Triton Server](https://github.com/triton-inference-server/server) were the main heroes for this project, (Triton was used to serve `onnx` models for embeddings) though I've yet to find a "good" embedding model. At this stage in time, most of the vector database solutions "feel" the same and can all seemingly be trivially hosted via Kubernetes. 

Tools used:

- Huggingface TGI (LLMs)
- Triton Inference Server (embeddings/`onnx` models)

Optimising usage of LLMs at scale is still a massive challenge, particularly when opening up for general access. The space is still new so there are still lots of patterns especially related to agentic patterns which need to be explored. On that particular note, [this particular blog post from Anthropic](https://www.anthropic.com/research/building-effective-agents) is something I'm looking at working on in the new year. At the same time, I've been experimenting with [mlflow](https://mlflow.org/docs/latest/llms/index.html) for LLM evaluations which has worked reasonably well, though I've pretty disappointed at [mlflow](https://mlflow.org/docs/latest/llms/index.html) from a self-hosting perspective, especially if only a tracking server is required. Maybe this is an open source project for the future.

## Python & TypeScript - in Review (2024)

> December 2024

One thing that I like to stress is the importance of _tooling_ and [relying on defaults](https://en.wikipedia.org/wiki/Convention_over_configuration). By being able to speak consistently within ones own projects or using commonly seen patterns reduces the mental overhead. These could be folder structures or idioms, especially things which permeate across different programming languages or frameworks. 

Here are some of my thoughts on Python and TypeScript; coming from someone who is predominantly a Python developer and does minimal front-end work. 

### Python

As a general trend, I've gone all-in in [astral](https://astral.sh/:

- Moving away from [poetry](https://python-poetry.org/) and onto [uv](https://docs.astral.sh/uv/). I believe [uv](https://docs.astral.sh/uv/) has reached a level of maturity which makes it production ready, especially its relative speed. 
- Similarly moving to [uv](https://docs.astral.sh/uv/), I'm defaulting more and more to [ruff](https://docs.astral.sh/ruff/) as a replacement for [flake8](https://github.com/PyCQA/flake8) and [isort](https://github.com/PyCQA/isort). 

**General setup**

```sh 
brew install uv
uv init <project>
```

with options in `pyproject.toml`

```toml
[tool.ruff]
line-length = 120  # changed from 88

[tool.ruff.lint]
select = ["E4", "E7", "E9", "F", "I001"]  # added isort option
```

In general using `ruff format` and `ruff check --write` make for good developer experiences

On "default" libraries, I do find myself trying to rely more and more on stdlib instead of particular frameworks. This is probably more a reflection of trying out [https://cosmo.zip/](https://cosmo.zip/) for trying out portable scripting as a default. 

Enterprise library recommendations:

* I do still use [pydantic](https://pydantic.dev/) very heavily. One thing that I would like to try is [logfire](https://github.com/pydantic/logfire) which I haven't had an opportunity to test. 
* For web frameworks [fastapi](https://fastapi.tiangolo.com/) is still more than sufficient and provides a good developer experience. I've run into issues with it being _too_ opinionated, and needed to write a fair bit of custom middleware, though nothing insurmountable. [sqlmodel](https://sqlmodel.tiangolo.com/) looks like a good option for what it does though I've personally had mixed experiences working with ORMs and generally have hand-rolled things rather than rely on frameworks. 
* [orjson](https://github.com/ijl/orjson) has been an amazing drop-in replacement for stdlib [json](https://docs.python.org/3/library/json.html) library, and I've witness substatial improvements to latency metrics in production services
* [pytest](https://github.com/pytest-dev/pytest) remains my preferred testing library of choice

**Patterns**

In terms of enterprise building patterns, I still recommend [Cosmic Python](https://www.cosmicpython.com/), in particular working with Domain Driven Design. 

### TypeScript

I'm very new to TypeScript, and have not used it in anger in enterprise settings. Infact modern web development is still very daunting and the tooling still doesn't "make sense" to me. The general webdev space feels like there was a lot of tacit knowledge that was in-built in the ecosystem and non-obvious for how a beginner should proceed. 

- Defaulting to [bun](https://bun.sh/). Maybe I'm just late to the game, and I feel like maybe I can just skip [node](https://nodejs.org/en). This allows me to use [bun test](https://bun.sh/docs/cli/test) as a test framework as well (rather than [jest](https://jestjs.io/) or [mocha](https://mochajs.org/))
- [biome](https://biomejs.dev/) has been amazing for formatting and linting with APIs very similar to [Python's ruff](https://docs.astral.sh/ruff/)
- [rolldown](https://rolldown.rs/), it just "works" for me out of the box and meant I didn't need to concern myself with builds and complex configuration
- [alpine.js](https://alpinejs.dev/) as a front-end javascript library. This may be a somewhat unusual choice, though coming first from Python and moving to TypeScript just means my biases is towards this model of development rather than using [React](https://react.dev/). 

Since I use TypeScript more as a hobbyist, I do find myself gravitating towards patterns which allow me to "deploy" applications straight to static hosting solutions like github pages. In the past, I have tried solutions like [Gatsby](https://www.gatsbyjs.com/docs/glossary/static-site-generator/) [Vite](https://vite.dev/guide/static-deploy) though they haven't quite clicked for me. 

**General Setup**

In general `bun` defaults work well. I would generally push source code to the `src` folder and use `rolldown` to push builds to `dist` with separate `html` files for testing the webserver. Hopefully in the future I get the whole software development cycle seamlessly working (n.b. I know [bun supports watch mode](https://bun.sh/docs/runtime/hot#watch-mode) supports watching)

Hopefully in the new year I get more acquainted with the appropriate patterns for TypeScript development. I may try doing more work with `bun repl` and scripting as a replacement to Python to practise my coding chops. 

