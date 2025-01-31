## Shell and Fonts

[Iosevka](https://github.com/be5invis/Iosevka) - because I work off my laptop screen alot, specifically the two below

```sh
brew install --cask font-iosevka-slab font-iosevka-etoile
```

There is a separate organisation for web fonts: https://github.com/iosevka-webfonts

[Oh My Zsh](https://ohmyz.sh/#install)  
[Starship](https://starship.rs/)  


## Themes

[Dracula](https://draculatheme.com/), good defaults, consistent experience.

## Text Completion via LLMs

[llama-vscode](https://marketplace.visualstudio.com/items?itemName=ggml-org.llama-vscode) with [llama.cpp](https://marketplace.visualstudio.com/items?itemName=ggml-org.llama-vscode). It's proven to be good enough and gets out of the way (plus I can self-host this). The medium size variant `ggml-org/Qwen2.5-Coder-3B-Q8_0-GGUF` is good enough for me

```sh
llama-server \
    -m qwen2.5-coder-3b-q8_0.gguf \
    --port 8012 -ngl 99 -fa -ub 1024 -b 1024 \
    --ctx-size 0 --cache-reuse 256
```

## Misc

[stats](https://formulae.brew.sh/cask/stats), because why not?  
[insomnia](https://github.com/Kong/insomnia), postman, but self-hosted  
[KeePass](https://en.wikipedia.org/wiki/KeePass), self-hosted password manager  
[obsidian](https://obsidian.md/) for taking notes  
