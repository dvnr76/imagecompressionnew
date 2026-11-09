# Face-aware WebP converter

`convert-face-webp.js` turns passport photos into ~5 KB, 250×250 WebP images while keeping the **face** as close to the original as possible.

## Setup (once)

```
npm install
```

This adds `onnxruntime-node` (runs the two small AI models locally, no internet needed at run time).
Note: it needs roughly 300 MB of free disk space. After installing you can reclaim ~160 MB by deleting
`node_modules\onnxruntime-node\bin\napi-v6\darwin` and `...\linux` (only `win32` is used on Windows).

The `models` folder must sit next to the script:

- `models/face-detector.onnx` – Ultra-Light-Fast face detector (MIT licence, 1.2 MB)
- `models/u2netp.onnx` – U²-Net-p person segmentation (Apache-2.0 licence, 4.5 MB)

## Run

```
node convert-face-webp.js
```

Defaults: reads `./23Allphotosnew`, writes `./23Allphotosnew_output`.
Other folders: `node convert-face-webp.js <inputDir> <outputDir>`

## What it does to each photo

1. Finds the face and crops a head-and-shoulders square around it, so framing is consistent even when the student isn't centred.
2. Removes the background and replaces it with flat white. This costs almost nothing in the file.
3. Keeps the face and hair pixels untouched, using a high-quality Lanczos resize only.
4. Smooths the clothing with edge-preserving smoothing. Fabric patterns are what eat the bytes.
5. Picks the highest WebP quality that fits in 5 KB. If that quality would drop below 72 (very busy prints), it simplifies the clothes a bit more and tries again, so the face keeps its quality.

## Settings (top of the script, `CONFIG`)

| Setting | Default | Meaning |
|---|---|---|
| `outputSize` | 250 | Output width/height in px |
| `targetKB` | 5 | Max file size |
| `minFaceQuality` | 72 | Below this, clothes get extra simplification |
| `faceScale` | 2.3 | Crop size relative to face height (bigger = more shoulders) |
| `faceCenterY` | 0.42 | Vertical position of the face in the crop |
| `removeBackground` | true | false = keep (smoothed) background |
| `background` | white | Background colour |
| `bodySmoothing` | 3 | 0–3, clothes only |
| `faceSharpen` | false | Mild sharpening on the face (costs bytes) |

## Results on the 1,125 photos in this repo

- Every face was detected (none needed the fallback crop).
- Average file size is 4.9 KB, and every file is at or under 5 KB.
- The typical WebP quality on the face is 75–85. The old script needed far more bytes to look comparable.
- It takes about 0.6 s per photo.
