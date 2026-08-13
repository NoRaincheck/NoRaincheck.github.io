# AGENTS.md

When generating or updating YAML frontmatter tags in blog posts, always use a list of strings format:

```yaml
tags: ["ai", "agentic-coding", "pi", "skills", "tooling", "reflection"]
```

Do **not** use the bare list format:

```yaml
tags: [ai, agentic-coding, pi, skills, tooling, reflection]
```

Each tag must be a quoted string inside a YAML list.
