---
title: Beating FastAPI
date: 2026-07-15
tags: ["Python", "FastAPI", "Performance", "Go"]
---

One thing which Go does a lot better than Python is single binary deployments. As an [experiment](https://github.com/NoRaincheck/gofre) I thought, why not have a way to package up Go as part of a Python package, similar to [maturin](https://www.maturin.rs/) - and also have a way to spin up a Python webserver that is packaged as a Go binary. 

The results were definitely interesting:

* Go binary had 9x more throughput than stdlib Python's http library, and ~5x more throughput than FastAPI
* Idle memroy usage was `3x higher for FastAPI, and the peak memory usage was ~2x higher than Go binary

Of course the pocket python restriction may be too much for more complex applications.

**Next Steps**

Test it out on other patterns and setups, such as with background tasks or with more concurrency to see the ablations

I also need to change the name, since `goforge` is already taken on PyPI.

**Edit**

Because `goforge` was already a taken name, I have changed it to `gofre`. In particular I have added benchmarks for web development which appear to be broadly promising. There are more interesting extensions that we can use, one of them is to take the `mypyc` approach of automatically creating C extensions off a typed subset of the code. Hopefully we can make it more ergonomic since `mypyc` doesn't seem to be widely used. Ideally something that does:

```sh
gofre compile /path/to/file.py  # or equivalent
```

which will then build appropriate bindings that are optimised assuming typing checks are met (use `ty` for this)
