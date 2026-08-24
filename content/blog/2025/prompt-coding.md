---
title: Prompt Coding
date: 2025-07-01
tags: ["Python"]
---

## Prompt Coding

_July 2025_

Perhaps another paradigm that is worth considering is just prompt coding,
whereby a single prompt produces a script end to end, no edits, no rework. This
means that you are just saving and rerunning the prompts rather than relying on
(perhaps flaky and difficult to reproduce) diffs and iterations. This way also
removes the reliance on AI IDEs, and you can use (free) chat interfaces to
create code.

Approaching coding in this way changes the flow. One item on my mind is thinking
through architecturally how we should structure software to faciliate this.
Doing things in this way naturally 'scales' since a single prompt or file leads
to modifying a single prompt/file. You won't have agentic setups which may
interact with each other.

As a related code templating pattern (specifically for Python), we should
probably focus on having code which contains both tests and code in the same
file. To achieve this, you can use the stdlib `unittest` library with custom
pattern for discoverability to solve it. For example:

```py
"""
An example of running unittest with the function definition. Run this via:

    python -m unittest discover -p "*.py"
"""

import unittest


def adder(left: float, right: float) -> float:
    return left + right


class AdderTestCase(unittest.TestCase):
    def test_it_works(self):
        result = adder(2, 2)
        self.assertEqual(result, 4)
```

Which follows a pattern whereby the unit tests are placed underneath the core
library code (similar to Rust). Note that for performance critical applications
this is a relatively bad idea since the tests are then loaded in the package. In
practise tests are not distributed in packages (the easy way is to put it in a
different folder that is not including in the Python packaging setup).
