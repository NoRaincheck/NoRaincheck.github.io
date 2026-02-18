## Shell and Fonts

[Iosevka](https://github.com/be5invis/Iosevka) - because I work off my laptop
screen alot, specifically a
[custom variant](https://github.com/HeardACat/Iosevka-Curly/tree/main).

For general "stuff", I currently enjoying the "gothic" router font aesthetic,
which is [used in this blog](https://heardacat.github.io/nationalpark-webfont/).

For my shell:

- [Oh My Zsh](https://ohmyz.sh/#install)
- [Starship](https://starship.rs/)

## Themes

[Catppuccin](https://catppuccin.com/) has started to replace
[Dracula](https://draculatheme.com/) as my daily driver, though I still use some
[Dracula](https://draculatheme.com/) themes like the terminal which I believe to
be superior option still. Alternatives I've considered are
[Rosé Pine](https://rosepinetheme.com/) which I've briefly toyed with, but
haven't taken seriously.

## Text Completion via LLMs

[llama-vscode](https://marketplace.visualstudio.com/items?itemName=ggml-org.llama-vscode)
with
[llama.cpp](https://marketplace.visualstudio.com/items?itemName=ggml-org.llama-vscode).
It's proven to be good enough and gets out of the way (plus I can self-host
this). I've had good success using `Devstral-Small` or even a general LLM like
`gemma-3-12b` variants

```sh
llama-server \
    -m qwen2.5-coder-3b-q8_0.gguf \
    --port 8012 -ngl 99 -fa -ub 1024 -b 1024 \
    --ctx-size 0 --cache-reuse 256
```

I've also started using `aider` for `/ask` commands. Though I generally don't
use local LLMs for agentic code editing as my daily driver.

```sh
uvx --from aider-install aider \
  --model openai/default \
  --openai-api-base http://127.0.0.1:8080/ --openai-api-key NONE \
  --map-tokens 1024 \
  --no-show-model-warnings \
  --no-gitignore
```

## VSCode Settings

I generally like to leave things to default, since moving to other people's
computer and workspaces things don't get confusing. Though the
`keybindings.json` I generally stick to is the terminal focus variant:

```json
[
  "files.exclude": { "**/.git": false },
  "workbench.colorTheme": "Catppuccin Macchiato",
  "editor.fontFamily": "'Iosevka Curly', Menlo, monospace",
  "go.inlayHints.assignVariableTypes": true,
  "go.inlayHints.compositeLiteralFields": true,
  "go.inlayHints.compositeLiteralTypes": true,
  "go.inlayHints.constantValues": true,
  "go.inlayHints.functionTypeParameters": true,
  "go.inlayHints.parameterNames": true,
  "rust-analyzer.cargo.features": ["all"]
]
```

## gitconfig

Stolen from [here](https://blog.gitbutler.com/how-git-core-devs-configure-git/).

```ini
# ~/.gitconfig 
[column]
    ui = auto
[branch]
    sort = -committerdate
[tag]
    sort = version:refname
[init]
    defaultBranch = main
[diff]
    algorithm = histogram
    colorMoved = plain
    mnemonicPrefix = true
    renames = true
[push]
    default = simple
    autoSetupRemote = true
    followTags = true
[fetch]
    prune = true
    pruneTags = true
    all = true

[help]
    autocorrect = prompt
[commit]
    verbose = true
[rerere]
    enabled = true
    autoupdate = true
[core]
    excludesfile = ~/.gitignore
[rebase]
    autoSquash = true
    autoStash = true
    updateRefs = true
```

## GPG

```sh
brew install gpg
gpg --gen-key
# by default for self stuff, use symmetric encryption
```

Ensure in `~/.zshrc`

```
GPG_TTY=$(tty)
export GPG_TTY
```

Example usage:

```sh
echo foo | gpg --symmetric > foo.gpg
gpg --decrypt foo.gpg
# to clipboard
gpg --decrypt foo.gpg | pbcopy
```

## Misc

[stats](https://formulae.brew.sh/cask/stats), because why not?\
[Insomnia](https://github.com/Kong/insomnia), postman, but self-hosted\
[KeePass](https://en.wikipedia.org/wiki/KeePass), self-hosted password manager\
[Obsidian](https://obsidian.md/) for taking notes, with
[Harper LSP Plugin](https://writewithharper.com/docs/integrations/obsidian)

I've been keeping a 10+ year old computer alive.
[Pale Moon](https://www.palemoon.org/) has been amazing for this.
