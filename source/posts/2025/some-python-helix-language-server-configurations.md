---
title: Some Python Helix Language Server Configurations
date: 2025-11-01
tags: ["Python"]
---

## Some Python Helix Language Server Configurations

_November 2025_

Getting Helix to play 'nicely' with custom configuration wasn't too
straightforward, specifically wanting ruff formatter to also do auto-fixes (e.g.
fixing imports). I also wanted the binaries to be managed by `uvx` rather than
polluting `$PATH` so needed a bit of customisation there as well.

Doing this ended up to be not too complicated, just need to find the correct
setup.

On MacOS

```toml
[[language]]
name = "python"
language-servers = ["pyrefly"]
auto-formatt = true
formatter = {command="bash", args=["-c", "uvx ruff check --fix - | uvx ruff format -"]}

[language-server.pyrefly]
command = "uvx"
args = ["pyrefly", "lsp"]
```

On Windows it is similar except using `cmd /c` instead of `bash -c`.

To test this you can check:

```sh
hx --health python
```

and in `NORMAL` mode

```sh
:fmt
```

to force formatter to run, allowing you to check that it is configured properly.
