---
title: Self-Hosted Agentic Engineering
date: 2026-09-13
tags: ["home-lab", "agents", "self-hosting", "local-first", "agentic-coding", "code-review"]
---

Recently I've been playing around a lot more with local-first software from [kenn-io](https://github.com/kenn-io). I have been enjoying it a lot, and it has made me think about the viability of single-software-engineer 'teams'.

The key parts are:

* `agentsview` is tremendously helpful in measuring and assessing the quality of agentic sessions, which is important when optimising your harness and the tooling around it (e.g. if you are using the Pi harness).
* `roborev` is incredible as a local-first tool. I now essentially have it running on all my personal projects as an async (in the background) way of ensuring code review is done on my commits. Even when working with a team, this is an amazing piece of software since it allows/enforces reviews before pushing/working on a PR in your team's centralised version control.
* `kata` is an interesting alternative to beads. Although I haven't used it in anger, I can definitely see its applicability, particularly in managing issues and the like locally as opposed to being dependent on a SaaS platform. Even using it as a local Trello (where your agents can access tasks and receive feedback) is useful enough as a tool.

Outside of the [kenn-io](https://github.com/kenn-io) tooling, I've also been looking at [LeafWiki](https://leafwiki.com/) as a simple locally hosted wiki. Combined with a simple Go server that allows for scaling to zero (e.g. self-hosting the wiki or an MCP server), it has really allowed local development to take off without worrying too much about local resource consumption.
