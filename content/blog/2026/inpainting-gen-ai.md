---
title: Inpainting with Generative AI
date: 2026-06-01
tags: ["ML", "image-generation", "CLI"]
---

## Inpainting with Generative AI

If anything, trying to do generative AI (images) via CLI is unusually 'hard'. Mostly because most flows use ComfyUI. I have found ComfyUI to be great when trying things out, or doing things interactively. 

The easiest way to use CLI/scripting has definitely been `stable-diffusion.cpp`: https://github.com/leejet/stable-diffusion.cpp 

For inpainting, it looks like the below

```sh
./bin/sd-cli \
  --diffusion-model flux-2-klein-9b-Q4_0.gguf \
  --vae flux2_dev_diffusion_pytorch_model.safetensors \
  --llm Qwen3-8B-Q3_K_M.gguf \
  --init-img bench.jpg \
  --mask dog-bench-mask.png \
  -p "a lovely dog" \
  --cfg-scale 2 \
  --sampling-method euler \
  -t 24 \
  --color \
  --steps 9 \
  -H 512 \
  -W 512 \
  --vae-tiling \
  --vae-tile-overlap 0.125 \
  -o dog-lovely-bench.png
```

What is important (to me) is that I find that binary masks that 'grows a bit' is better than providing a fuzzy mask, on my hardware this runs reasonably quickly (~15s per step at 512x512). 