---
title: Syncify Python Async Functions
date: 2025-04-01
tags: ["Python", "async"]
---

## Syncify Python Async Functions

_April 2025_

There's a lot to unpack with Python shenanigans. Including why
[nest-asyncio](https://pypi.org/project/nest-asyncio/) even exists. Here is yet
another pattern (note: you really should look into https://asyncer.tiangolo.com/
if it fits for you).

```py
def syncify(func, *args, **kwargs):
    def wrapper(coro):
        return asyncio.run(coro)

    with ThreadPoolExecutor(max_workers=1) as executor:
        result = next(executor.map(wrapper, [func(*args, **kwargs)]))
    return result
```

Which delegates/moves the execution to a threadpool to bypass potential issues.
This is definitely not a 'performant' option, but it may save your skin.

I'll point out in general, where your event loop is not nested, you could just
use `asyncio.run`.
