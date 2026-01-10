## Learning Vim

_WIP notes on learning vim_

**Why `vim`?**

After using lots of co-pilot and cursor, my conclusion is its better to walk away (occassionally) away from those tools,
but when you primarily work in those environments the temptation is very high. Combined with my experience with trying
to use `vscode` on an older PC (which failed miserably), is the motivation for me to finally try and configure `vim`
carefully.

**Getting Setup**

One of the first things I realised when installation `nvim` is that you need a terminal with full colour support,
otherwise everything looks wrong. I've chosen to use:

- terminal app for windows
- ghostty for macos, primarily thanks to its quick and easy color theming which is amazing

**Installation**

Having tried different 'flavours' of `vim` I've arrived at either:

- just starting from `lazy.vim` and building up
- using `astro.vim` to have good defaults

I'm still getting my head around using things like:

```
:Neotree
:sp | terminal
-- LSP plugins
```

So I don't have a good grasp of what _combination_ of things I like yet. That will probably come with time.

**Things to get more comfortable with**

I like having a terminal around to run things, whether it is:

- `cargo check`
- `pytest`
- `just`

etc..., I'm still not comfortable enough to go around doing `<C-w> w` everywhere, nor am I comfortable enough jumping
around words, end of line etc. I need to build up and internalise the shortcuts as I come across them. One could argue,
maybe use tmux and screen, but part of me wants to 'keep things simple', even though I know that `nvim` will add so much
complexity over time. Ironic, isn't it?

Random notes:

- If you're stuck in a terminal window, press `<C-\> <C-n>`
- End of line is just the regex stuff `^` and `$`, and `gg`, `G` for the beginning vs end of document.
- Be more comfortable with `i` vs `a` for append and insert mode, specifically `<S-i>` and `<S-a>` for moving to the
  start/end of line respectively
- Be more comfortable using `<Ctrl d>` and `<Ctrl u>` for moving half screen down/up respectively, and maybe `<Ctrl f>`
  `<Ctrl b>` for forward/back.
- Make use of the terminal shortcuts, i.e. the horizontal terminal is `<leader> th`
