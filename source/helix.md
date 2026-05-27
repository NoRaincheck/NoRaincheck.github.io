---
title: Tutorials - Helix
---

## Moving to Helix

[Helix](https://helix-editor.com/) (`hx`) is a modern modal editor built in
Rust. Unlike Vim/Neovim, Helix ships with a built-in LSP client, tree-sitter
integration, and multiple cursor support out of the box — no plugins required.

### Why Helix over Vim/Neovim?

- **Zero-config LSP** — language servers are detected and configured
  automatically from your project. No more `nvim-lspconfig` or Mason.
- **Tree-sitter natively** — syntax highlighting, text objects, and navigation
  all use tree-sitter. No plugins needed.
- **Multiple cursors** — a first-class feature, not an afterthought.
- **Built-in file picker** — fuzzy finding via `fzf`-like interface, no
  Telescope.
- **Selection-driven editing** — Helix flips the Vim model: motions extend the
  selection first, then you act on it. This is more intuitive once you adjust.

### Key Differences from Vim

| Concept             | Vim / Neovim            | Helix                                  |
| ------------------- | ----------------------- | -------------------------------------- |
| **Normal mode**     | motions act immediately | motions extend the selection           |
| **Insert mode**     | `i` to enter            | `i` enters insert (same)               |
| **Select then act** | `d2w` (delete 2 words)  | `2w` extends selection, `d` deletes it |
| **File explorer**   | Requires plugin         | Built-in `<space>f` or `:open`         |
| **LSP**             | Plugin required         | Built-in, auto-configured              |
| **Config**          | Lua / Vimscript         | `~/.config/helix/config.toml`          |
| **Language config** | Per-plugin setup        | `~/.config/helix/languages.toml`       |

### Getting Started

```sh
# macOS
brew install helix

# Arch
sudo pacman -S helix

# or build from source
git clone https://github.com/helix-editor/helix
cargo install --path helix-term
```

### Configuration

Helix uses `toml` for everything. See the
[official config docs](https://docs.helix-editor.com/configuration.html).

Example `~/.config/helix/config.toml`:

```toml
theme = "catppuccin_macchiato"

[editor]
line-number = "relative"
cursorline = true
color-modes = true
true-color = true

[editor.cursor-shape]
insert = "bar"
normal = "block"
select = "underline"
```

### Language Configuration

Language servers, formatters, and tree-sitter grammars are configured in
`~/.config/helix/languages.toml`. See the
[language config docs](https://docs.helix-editor.com/languages.html).

```toml
[language-server.rust-analyzer]
command = "rust-analyzer"

[[language]]
name = "rust"
language-servers = ["rust-analyzer"]
auto-format = true
```

### Key Mappings

Helix's default keymap is well thought-out. See the
[official keymap docs](https://docs.helix-editor.com/keymap.html).

Notable differences from Vim:

- `<space>` is the **leader key** — opens a menu of common actions (file picker,
  LSP commands, buffers, etc.)
- `v` enters **select mode** (equivalent to Vim's visual mode)
- `C` copies the selection to the clipboard
- `Alt` + `hjkl` moves between **split windows**
- `g` (go-to) + `h` moves to the **start of the line**
- `g` (go-to) + `l` moves to the **end of the line**
- `:` opens the command mode (similar to Vim)

### Switching from Vim

The biggest mental shift is **selection-first** editing. In Vim, you press `d2w`
— a command followed by a motion. In Helix, you press `2w` to extend the
selection to the second word boundary, then `d` to delete the selection.

This means `dd` (delete line) becomes `xd` (extend to full line, then delete).

Vim muscle memory refresher:

| Action                 | Vim                  | Helix           |
| ---------------------- | -------------------- | --------------- |
| Delete line            | `dd`                 | `xd`            |
| Delete 2 words         | `d2w`                | `2wd`           |
| Yank line              | `yy`                 | `xy`            |
| Change inside brackets | `ci(`                | `mi(d`          |
| Comment toggle         | `gc` (with plugin)   | `gc` (built-in) |
| Format file            | `gg=G` (with plugin) | `:fmt`          |
| Find files             | Requires plugin      | `<space>f`      |
| LSP goto def           | `gd` (with plugin)   | `<space>gd`     |
| LSP rename             | `grn` (with plugin)  | `<space>r`      |

### Official Resources

- [Helix Homepage](https://helix-editor.com/)
- [Official Documentation](https://docs.helix-editor.com/)
- [Keymap Reference](https://docs.helix-editor.com/keymap.html)
- [Configuration Guide](https://docs.helix-editor.com/configuration.html)
- [Language Support](https://docs.helix-editor.com/languages.html)
- [GitHub Repository](https://github.com/helix-editor/helix)
- [Migrating from Vim](https://docs.helix-editor.com/faq.html#how-do-i-migrate-from-vim)
