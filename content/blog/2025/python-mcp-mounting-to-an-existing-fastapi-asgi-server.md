---
title: Python MCP - Mounting to an Existing FastAPI ASGI Server
date: 2025-04-01
tags: ["Python", "async", "FastAPI", "MCP"]
---

## Python MCP - Mounting to an Existing FastAPI ASGI Server

_April 2025_

You can mount the SSE server to an existing FastAPI ASGI server dynamically. The
way I've made it work is as follows:

```py
sse = SseServerTransport(...)

...
app.router.routes.append(Mount("/messages", app=sse.handle_post_message))

@app.get("/sse")
async def handle_mcp_connection(request: Request):
    async with sse.connect(sse(request.scope, request.receive, request.send) as (
        read_stream,
        write_stream,
    ):
        await mcp._mcp_server.run(
            read_stream,
            write_stream,
            mcp._mcp_server.create_initialization_options(),
        )
```

This seemed to work pretty well, along with having Redis as an intermediary
broker for dealing with state.
