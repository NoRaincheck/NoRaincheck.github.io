---
title: Python Debugger and VSCode
date: 2026-03-01
tags: ["Python"]
---

## Python Debugger and VSCode

_March 2026_

I find myself copy + pasting (or ChatGPT) the debugging configuration more often
than I would like specifically for `pytest` so replicating it here.

```json
{
  "configurations": [
    {
      "name": "$NAME",
      "type": "debugpy",
      "request": "launch",
      "module": "pytest",
      "args": ["tests/test_name_of_file.py::test_name_of_func"],
      "justMyCode": false
    }
  ]
}
```
