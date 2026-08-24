---
title: Starting Up Matrix with Docker Compose
date: 2026-03-01
tags: ["self-hosting", "Docker", "CLI", "Matrix"]
---

## Starting Up Matrix with Docker Compose

_March 2026_

Matrix has a lot of complicated self-hosting docs, mainly to do with federation
and hosting. If you're just self-hosting and not worried about setting up
ingresses/federating, its actually fairly simple to setup Matrix. Here is a
`docker-compose.yml` file that will do it. The only bit of configuration is to
pull up the `element_config.json` though that can be left as default values (you
can override them later)

```yaml
# Conduit
version: "3"

services:
  homeserver:
    image: matrixconduit/matrix-conduit:latest
    restart: unless-stopped
    ports:
      - 8448:6167
    volumes:
      - db:/var/lib/matrix-conduit/
    environment:
      CONDUIT_SERVER_NAME: changeme # EDIT THIS
      CONDUIT_DATABASE_PATH: /var/lib/matrix-conduit/
      CONDUIT_DATABASE_BACKEND: rocksdb
      CONDUIT_PORT: 6167
      CONDUIT_MAX_REQUEST_SIZE: 20000000 # in bytes, ~20 MB
      CONDUIT_ALLOW_REGISTRATION: "true"
      CONDUIT_ALLOW_FEDERATION: "false"
      CONDUIT_ALLOW_CHECK_FOR_UPDATES: "false"
      CONDUIT_TRUSTED_SERVERS: "[]"
      CONDUIT_ADDRESS: 0.0.0.0
      CONDUIT_CONFIG: "" # Ignore this

  ### Config-Docs: https://github.com/vector-im/element-web/blob/develop/docs/config.md
  element-web:
    image: vectorim/element-web:latest
    restart: unless-stopped
    ports:
      - 8009:80
    volumes:
      - ./element_config.json:/app/config.json
    depends_on:
      - homeserver

volumes:
  db:
```

To get around the annoying user registration, you may want to manually create it
yourself. It would be something like:

```sh
curl -X POST -d '{"username":"john", "password":"smith", "auth": {"type":"m.login.dummy"}}' "http://localhost:8448/_matrix/client/r0/register"
curl -X POST -d '{"username":"jane", "password":"doe", "auth": {"type":"m.login.dummy"}}' "http://localhost:8448/_matrix/client/r0/register"
```
