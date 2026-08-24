---
title: Notes on Resizing Images
date: 2026-07-18
tags: ["Python", "OpenCV"]
---

## Notes on Resizing Images

_July 2026_

Given the influx of AI and readily available tools for things like super resolution (e.g. RealESRGAN), better ways to enhance images when resizing _downwards_ becomes important as well. 

The broad idea we should use looks like:

```py
orig_w, orig_h = img.size
new_h = round(TARGET_WIDTH * orig_h / orig_w)

# Gaussian blur pre-filter for anti-aliasing
blurred = img.filter(ImageFilter.GaussianBlur(radius=0.5))

# One-step Lanczos downscale
resized = blurred.resize((TARGET_WIDTH, new_h), Image.LANCZOS)

output_path = OUTPUT_DIR / f"page_{page_num:03d}.png"
resized.save(output_path, "PNG", optimize=True)
```

Where setting something like `radius` being related to the size of the image scaled. For example, if the resulting image is half the size log the original image, a `radius=0.5` is probably appropriate. Whereas if it is 1/5 of the size maybe `radius` of 1 to 1.5 is better. If you're after an equation, `radius=math.log(scale)` is pretty close to the target.
