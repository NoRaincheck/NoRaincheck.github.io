---
title: LAN Messenger and Agents
date: 2026-08-08
tags: [home-lab, xmpp, p2p, agents]
---

Been thinking about home-lab setups for agents and finally settled on: [pi-msg](https://github.com/NoRaincheck/pi-msg).

Advantages:

* **Serverless.** Uses [XEP-0174](https://xmpp.org/extensions/xep-0174.html) for peer-to-peer messaging — no server required.
* **Simple.** You chat with the bot like you're chatting with someone.
* **Easy to extend.** Create a new agent by pointing to a new project or workspace.

Disadvantages:

* **Limited parsing.** Depends on client support — full HTML rendering has historically been uncommon in XMPP clients.

Still thinking about the approach:

* Should I build my own client to support markdown?

It's been a bit of fun. Looking forward to extending it more — I can see it becoming a powerful, regular tool.
