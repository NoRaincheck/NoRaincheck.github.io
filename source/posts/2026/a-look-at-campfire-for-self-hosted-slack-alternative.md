---
title: A look at Campfire for self-hosted Slack alternative
date: 2026-03-01
tags: ["SQLite", "Docker", "FastAPI"]
---

## A look at Campfire for self-hosted Slack alternative

_March 2026_

Campfire is not really a commonly suggested alternative to Slack/Discord.
Afterall its missing various features including voice chat, threads or other
features. It makes up for it by being incredibly easy to self-host and easy to
write a bot that can interact in such an interface.

Campfire can be hosted from a single Dockerfile using sqlite as its database.
For majority of low-numbered user-count situations this is more than sufficient.

```yaml
services:
  campfire:
    image: ghcr.io/basecamp/once-campfire:latest
    volumes:
      - ./campfire_storage:/rails/storage
    restart: always
    ports:
      - "3000:3000" # or whatever is appropriate
    environment: # probably other envvars to setup if you are actually exposing this via tailscale etc..
      SECRET_KEY_BASE: "salt"
      DISABLE_SSL: "true"
```

Writing a bot for this is likewise very straightforward. It uses a webhook with
a set format. Just make sure your response is correct format. You can also send
images or video with the appropriate MIME type. For example using
FastAPI/Starlette it may look something like this:

```py
@app.post(
    "/debug",
)
def debug_request(request: WebhookRequest):
    return Response(
        content=DEFAULT_MESSAGE_TEMPLATE.render(debug=request, msg="Debug Request Output"),
        media_type="text/html",
    )
```

If this isn't set appropriately, the message would be treated as a `json`
attachment, which is probably not what you're after.
