---
title: Maintaining Scripts using Just and Gum
date: 2025-04-01
tags: ["CLI", "shell"]
---

## Maintaining Scripts using Just and Gum

_April 2025_

I've found maintaining shell scripts and aliases to be a bit of a mess. I know
people have their own systems with dotfiles, but it never 'stuck' with me. I'm
trialing a new approach using [gum](https://github.com/charmbracelet/gum/) and
[just](https://github.com/casey/just), both which can be installed via `brew`.
This makes it so there less memorisation of what different programs do, and
their parameters, since you can control and name variables to a flow that you
like. Combined with `just`'s ability to target `justfile` from different
directories, this makes centralising scripts and cli commands very
straightforward.

For example, the following are roughly equivalent:

```
$ just foo/a b
$ (cd foo && just a b)
```

To avoid the `cd`, the annotation `[no-cd]` can be used. Putting this
altogether, I can for example have some "global" scripts that converts video to
audio, or downloads youtube videos using the following recipe:

```makefile
[no-cd]
convert_to_audio:
    FILE=$(gum file)
    ffmpeg -i $FILE -vn -acodec libvorbis -q:a 0 $FILE.ogg

[no-cd]
download_video:
    uvx yt-dlp@latest -S "res:$(gum choose 1080 720 480 360 240 144)" $(gum input)
```

Edit: _August 2025_

Recently when trying to setup API keys and authentication for a locally running
service, I ended up using `gum` to tackle having a 'lightweight' approach to
avoid having credentials in files. In a nutshell it would look something like:

```makefile
run_server:
    API_KEY=$(gum input --password)
    start-server.sh
```
