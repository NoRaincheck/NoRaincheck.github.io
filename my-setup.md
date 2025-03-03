## Shell and Fonts

[Iosevka](https://github.com/be5invis/Iosevka) - because I work off my laptop screen alot, specifically the `slab` variant.

```sh
brew install --cask font-iosevka-slab
```

For general "stuff", I currently enjoying the "gothic" router font aesthetic, which is [used in this blog](https://heardacat.github.io/nationalpark-webfont/).

For my shell:  
- [Oh My Zsh](https://ohmyz.sh/#install)  
- [Starship](https://starship.rs/)  

## Themes

[Dracula](https://draculatheme.com/), good defaults, consistent experience. Alternatives I've considered are [Catppuccin](https://catppuccin.com/) which I actually really, really like, and [Rosé Pine](https://rosepinetheme.com/) which I've briefly toyed with, but haven't taken seriously. 

## Text Completion via LLMs

[llama-vscode](https://marketplace.visualstudio.com/items?itemName=ggml-org.llama-vscode) with [llama.cpp](https://marketplace.visualstudio.com/items?itemName=ggml-org.llama-vscode). It's proven to be good enough and gets out of the way (plus I can self-host this). The medium size variant `ggml-org/Qwen2.5-Coder-3B-Q8_0-GGUF` is good enough for me

```sh
llama-server \
    -m qwen2.5-coder-3b-q8_0.gguf \
    --port 8012 -ngl 99 -fa -ub 1024 -b 1024 \
    --ctx-size 0 --cache-reuse 256
```

## VSCode Settings

I generally like to leave things to default, since moving to other people's computer and workspaces things don't get confusing. Though the `keybindings.json` I generally stick to is the terminal focus variant:

```json
[
    {
        "key": "ctrl+`",
        "command": "workbench.action.terminal.focus"
    },
    {
        "key": "ctrl+`",
        "command": "workbench.action.focusActiveEditorGroup",
        "when": "terminalFocus"
    }
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

[stats](https://formulae.brew.sh/cask/stats), because why not?  
[insomnia](https://github.com/Kong/insomnia), postman, but self-hosted  
[KeePass](https://en.wikipedia.org/wiki/KeePass), self-hosted password manager  
[obsidian](https://obsidian.md/) for taking notes  
