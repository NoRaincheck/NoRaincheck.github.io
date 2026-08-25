---
title: OpenCV 5 Python Bindings Is Here
date: 2026-07-01
tags: ["Python", "OpenCV"]
---

## OpenCV 5 Python Bindings Is Here

_July 2026_

OpenCV 5 for Python has finally been released to `pypi` and comes with QoL for running large models through their pipeline. One interesting way to use this pipeline is to use SAM model with LaMa together in a single code base

```py
sam_result = sam_segment(img)
if sam_result is None:
    print("  SAM failed, aborting")
    return
mask, point_coords = sam_result

inpainted = lama_inpaint(img, mask)
```

Where each of the functions are:

```py

def sam_segment(img, point_coords=None):
    """Segment object using SAM2.1 with point prompts."""
    encoder_path = MODELS_DIR / "sam_encoder_v2.onnx"
    decoder_path = MODELS_DIR / "sam_decoder_v2.onnx"
    if not encoder_path.exists() or not decoder_path.exists():
        print(f"  [!] SAM models not found in {MODELS_DIR}")
        return None

    t0 = time.time()
    h, w = img.shape[:2]

    scale = SAM_IMAGE_SIZE / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_LINEAR)
    padded = np.zeros((SAM_IMAGE_SIZE, SAM_IMAGE_SIZE, 3), dtype=np.uint8)
    padded[:new_h, :new_w] = resized

    pixel_values = padded.astype(np.float32) / 255.0
    pixel_values = (pixel_values - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    pixel_values = pixel_values.transpose(2, 0, 1)[np.newaxis]

    encoder = cv.dnn.readNetFromONNX(str(encoder_path), engine=cv.dnn.ENGINE_NEW)
    encoder.setInput(pixel_values, "image")
    image_embed = encoder.forward("image_embed")
    high_res_0 = encoder.forward("high_res_feats_0")
    high_res_1 = encoder.forward("high_res_feats_1")

    if point_coords is None:
        point_coords = [(w // 2, h // 2)]
    point_labels = [1] * len(point_coords)
    print(f"  Point prompts: {point_coords}")

    scale_x, scale_y = new_w / w, new_h / h
    pts = np.array([[[int(x * scale_x), int(y * scale_y)] for x, y in point_coords]], dtype=np.float32)
    labels = np.array([point_labels], dtype=np.float32)

    mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    has_mask_input = np.array([0.0], dtype=np.float32)

    decoder = cv.dnn.readNetFromONNX(str(decoder_path), engine=cv.dnn.ENGINE_AUTO)
    decoder.setInput(image_embed, "image_embed")
    decoder.setInput(high_res_0, "high_res_feats_0")
    decoder.setInput(high_res_1, "high_res_feats_1")
    decoder.setInput(pts, "point_coords")
    decoder.setInput(labels, "point_labels")
    decoder.setInput(mask_input, "mask_input")
    decoder.setInput(has_mask_input, "has_mask_input")

    masks = decoder.forward("masks")
    iou_predictions = decoder.forward("iou_predictions")

    best_idx, best_score = 0, -1
    for i in range(masks.shape[1]):
        sig = 1.0 / (1.0 + np.exp(-masks[0, i]))
        area_ratio = sig.mean()
        if 0.02 < area_ratio < 0.80:
            score = iou_predictions[0, i]
            if score > best_score:
                best_score = score
                best_idx = i
    if best_score < 0:
        for i in range(masks.shape[1]):
            sig = 1.0 / (1.0 + np.exp(-masks[0, i]))
            area_ratio = sig.mean()
            if area_ratio < 0.95:
                score = iou_predictions[0, i]
                if score > best_score:
                    best_score = score
                    best_idx = i
    if best_score < 0:
        best_idx = int(np.argmax(iou_predictions[0]))

    mask_logits = masks[0, best_idx]
    mask = (1.0 / (1.0 + np.exp(-mask_logits)) > 0.5).astype(np.uint8) * 255
    mask = cv.resize(mask, (w, h), interpolation=cv.INTER_NEAREST)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))
    mask = cv.dilate(mask, kernel, iterations=3)

    elapsed = time.time() - t0
    coverage = (mask > 127).mean() * 100
    print(f"  SAM: IoU={iou_predictions[0, best_idx]:.3f}, coverage={coverage:.1f}%  [{elapsed:.2f}s]")
    return mask, point_coords
```

```py

def lama_inpaint(img, mask):
    """Remove object using LaMa inpainting model."""
    model_path = MODELS_DIR / "lama.onnx"
    if not model_path.exists():
        print(f"  [!] LaMa model not found: {model_path}")
        return None

    t0 = time.time()
    h, w = img.shape[:2]

    resized_img = cv.resize(img, (LAMA_MODEL_SIZE, LAMA_MODEL_SIZE))
    resized_mask = cv.resize(mask, (LAMA_MODEL_SIZE, LAMA_MODEL_SIZE))
    _, resized_mask = cv.threshold(resized_mask, 127, 255, cv.THRESH_BINARY)

    image_blob = cv.dnn.blobFromImage(resized_img, 1 / 255.0, (LAMA_MODEL_SIZE, LAMA_MODEL_SIZE))
    mask_blob = cv.dnn.blobFromImage(resized_mask, scalefactor=1.0, size=(LAMA_MODEL_SIZE, LAMA_MODEL_SIZE), mean=(0,), swapRB=False, crop=False)
    mask_blob = (mask_blob > 0).astype(np.float32)

    net = cv.dnn.readNetFromONNX(str(model_path), engine=cv.dnn.ENGINE_AUTO)
    net.setInput(image_blob, "image")
    net.setInput(mask_blob, "mask")
    output = net.forward()

    result = output[0]
    result = np.transpose(result, (1, 2, 0))
    result = np.clip(result, 0, 255).astype(np.uint8)
    result = cv.resize(result, (w, h))

    elapsed = time.time() - t0
    print(f"  LaMa: inpainting complete  [{elapsed:.2f}s]")
    return result
```


![output](/assets/2026/lama-inpainting-output.jpg)
