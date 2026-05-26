---
title: Setting SQL for Feature Transformations as a Standard
date: 2025-03-01
tags: ["production", "Python", "ML"]
---

## Setting SQL for Feature Transformations as a Standard

_March 2025_

So Spark's `SQLTransformer` is probably the first (and only?) documented,
formal, specification for doing SQL transformations specifically for machine
learning preprocessing. I think that is interesting for a variety of reasons,
with the biggest one being opportunities for standardisation, in particular for
real-time flows. Having a well-optimised SQL transformation engine could be
immensely valuation. The reason why this hasn't occured is probably because
industry standard today still relies on Python as the execution engine, however
this because untenable in scenarios where Python is an inappropriate production
programming language choice. At the same time, Spark is typically too expensive
of a dependency to justify low latency workflows. Nevertheless lets quickly look
at the specification and considerations for using the language with an
alternative computation backend.

1. `SQLTransformer` supports Spark SQL syntax, because that is what it is built
   on.
2. Since this is strictly for feature transformation we **probably** should
   ignore all filtering, i.e. only supports
   `SELECT ... from __THIS__ where __THIS__`
3. The only table name is `__THIS__`

All of these are slightly unusual quirks for `SQLTransformation` which devolves
building a computation backend to tackle `SQL` expression-like transformations,
which is much more tenable.

Some unusual parts of `SQLTransformer` is the use of vector notation. It's
really interesting to see converters like
[PMML](https://github.com/jpmml/jpmml-sparkml) format support this, but not have
this present in more "modern" formats. Its probably a gap that should be solved
in the future.
