/**
 * FACE-AWARE PASSPORT PHOTO → WEBP CONVERTER
 * ------------------------------------------
 * Goal: keep the FACE crisp while hitting a small file size (~5 KB).
 *
 * How it works (per image):
 *  1. Detect the face (Ultra-Light face detector, ONNX, ~1 MB).
 *  2. Crop a head-and-shoulders square framed around the detected face.
 *  3. Segment the person from the background (U²-Net-p, ONNX, ~4.5 MB)
 *     and replace the background with flat white (costs ~0 bytes in WebP).
 *  4. Keep face + hair pixels untouched (high-quality Lanczos resize + light
 *     sharpening). Clothing/neck area gets edge-preserving smoothing, which
 *     removes fabric patterns/noise that would otherwise eat the byte budget.
 *  5. Binary-search the WebP quality so each file lands just under the target
 *     size → all spare bytes go to the face.
 *
 * Usage:  node convert-face-webp.js [inputDir] [outputDir]
 * Needs:  npm install sharp onnxruntime-node
 *         models/face-detector.onnx and models/u2netp.onnx next to this script
 */

const fs = require('fs').promises;
const path = require('path');
const os = require('os');
const sharp = require('sharp');
const ort = require('onnxruntime-node');

// === CONFIGURATION ===
const CONFIG = {
  inputDir: process.argv[2] || './23Allphotosnew',
  outputDir: process.argv[3] || './23Allphotosnew_output',
  outputSize: 250,          // output is outputSize × outputSize px
  targetKB: 5,              // max file size per image (KB)
  minQuality: 30,           // never go below this WebP quality
  minFaceQuality: 72,       // if quality would drop below this, clothes get simplified more
  maxQuality: 95,           // never go above this (no visible gain beyond)
  faceScale: 2.3,           // crop side = face height × faceScale (bigger = more shoulders)
  faceCenterY: 0.42,        // face centre sits at this fraction from the top of the crop
  removeBackground: true,   // false = keep background (but it will be smoothed)
  background: { r: 255, g: 255, b: 255 },
  bodySmoothing: 3,         // 0 = off, 1 = light, 2 = medium, 3 = strong (clothes/body only, never the face)
  faceDenoise: 0,           // 0 = off, 1 = very light (removes camera noise only, keeps detail)
  faceSharpen: false,       // mild unsharp mask on the face only (adds bytes; off = most faithful)
  webpOptions: { smartSubsample: true, smartDeblock: true, preset: 'photo' },
  concurrency: Math.max(1, Math.min(4, os.cpus().length - 1)),
  modelDir: path.join(__dirname, 'models'),
};

sharp.cache(false);
ort.env.logLevel = 'error';
const ORT_OPTS = { logSeverityLevel: 3, intraOpNumThreads: 2 };

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------
let faceSession, segSession;
async function loadModels() {
  faceSession = await ort.InferenceSession.create(
    path.join(CONFIG.modelDir, 'face-detector.onnx'), ORT_OPTS);
  if (CONFIG.removeBackground) {
    segSession = await ort.InferenceSession.create(
      path.join(CONFIG.modelDir, 'u2netp.onnx'), ORT_OPTS);
  }
}

// === FACE DETECTION (Ultra-Light RFB-320: input 320×240, RGB, (x-127)/128) ===
async function detectFace(img, W, H) {
  // Letterbox to 4:3 so the face is not distorted
  const DW = 320, DH = 240;
  const scale = Math.min(DW / W, DH / H);
  const rw = Math.round(W * scale), rh = Math.round(H * scale);
  const padX = Math.floor((DW - rw) / 2), padY = Math.floor((DH - rh) / 2);

  const { data } = await sharp(img)
    .resize(rw, rh, { kernel: 'lanczos3' })
    .extend({ left: padX, right: DW - rw - padX, top: padY, bottom: DH - rh - padY,
              background: { r: 127, g: 127, b: 127 } })
    .removeAlpha().raw().toBuffer({ resolveWithObject: true });

  const plane = DW * DH, input = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    input[i] = (data[i * 3] - 127) / 128;
    input[plane + i] = (data[i * 3 + 1] - 127) / 128;
    input[2 * plane + i] = (data[i * 3 + 2] - 127) / 128;
  }
  const out = await faceSession.run({ input: new ort.Tensor('float32', input, [1, 3, DH, DW]) });
  const scores = out.scores.data, boxes = out.boxes.data, n = scores.length / 2;

  // Collect candidate boxes (in original-image pixels)
  const cands = [];
  for (let i = 0; i < n; i++) {
    const conf = scores[i * 2 + 1];
    if (conf < 0.6) continue;
    const x1 = (boxes[i * 4] * DW - padX) / scale, y1 = (boxes[i * 4 + 1] * DH - padY) / scale;
    const x2 = (boxes[i * 4 + 2] * DW - padX) / scale, y2 = (boxes[i * 4 + 3] * DH - padY) / scale;
    if (x2 - x1 < 8 || y2 - y1 < 8) continue;
    cands.push({ x1, y1, x2, y2, conf, s: conf * (x2 - x1) * (y2 - y1) });
  }
  if (!cands.length) return null;

  // Best face = confidence × area (passport photo = one big face),
  // then average all overlapping detections for a stable box
  const top = cands.reduce((a, b) => (b.s > a.s ? b : a));
  const iou = (a, b) => {
    const ix = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
    const iy = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
    const inter = ix * iy;
    return inter / ((a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter);
  };
  const group = cands.filter(c => iou(c, top) > 0.5);
  const wsum = group.reduce((s, c) => s + c.conf, 0);
  const avg = k => group.reduce((s, c) => s + c[k] * c.conf, 0) / wsum;
  const best = { x1: avg('x1'), y1: avg('y1'), x2: avg('x2'), y2: avg('y2'), conf: top.conf };
  return {
    x: Math.max(0, best.x1), y: Math.max(0, best.y1),
    w: Math.min(W, best.x2) - Math.max(0, best.x1),
    h: Math.min(H, best.y2) - Math.max(0, best.y1),
    conf: best.conf,
  };
}

// === CROP WINDOW AROUND FACE ===
function computeCrop(face, W, H) {
  let side, cx, cy;
  if (face) {
    side = face.h * CONFIG.faceScale;
    cx = face.x + face.w / 2;
    cy = face.y + face.h / 2;
  } else {               // fallback: top-centred square
    side = Math.min(W, H) * 0.92;
    cx = W / 2;
    cy = side * CONFIG.faceCenterY + H * 0.02;
  }
  side = Math.min(side, W, H);
  let left = cx - side / 2;
  let top = cy - side * CONFIG.faceCenterY;
  left = Math.max(0, Math.min(left, W - side));
  top = Math.max(0, Math.min(top, H - side));
  return { left: Math.round(left), top: Math.round(top), size: Math.floor(side) };
}

// === PERSON SEGMENTATION (U²-Net-p, 320×320, ImageNet normalisation) ===
async function segmentPerson(cropBuf, size) {
  const S = 320;
  const { data } = await sharp(cropBuf).resize(S, S, { kernel: 'lanczos3' })
    .removeAlpha().raw().toBuffer({ resolveWithObject: true });
  const plane = S * S, input = new Float32Array(3 * plane);
  const mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225];
  for (let i = 0; i < plane; i++)
    for (let c = 0; c < 3; c++)
      input[c * plane + i] = (data[i * 3 + c] / 255 - mean[c]) / std[c];

  const feeds = { [segSession.inputNames[0]]: new ort.Tensor('float32', input, [1, 3, S, S]) };
  const out = await segSession.run(feeds);
  const pred = out[segSession.outputNames[0]].data;

  let mn = Infinity, mx = -Infinity;
  for (const v of pred) { if (v < mn) mn = v; if (v > mx) mx = v; }
  const mask = Buffer.alloc(plane);
  for (let i = 0; i < plane; i++) mask[i] = Math.round(((pred[i] - mn) / (mx - mn + 1e-8)) * 255);

  // Upscale mask to output resolution
  return sharp(mask, { raw: { width: S, height: S, channels: 1 } })
    .resize(size, size, { kernel: 'cubic' }).extractChannel(0).raw().toBuffer();
}

// === EDGE-PRESERVING SMOOTHING (bilateral filter, RGB) ===
function bilateral(src, W, H, radius, sigmaS, sigmaC) {
  const dst = Buffer.alloc(src.length);
  const spatial = [];
  for (let dy = -radius; dy <= radius; dy++)
    for (let dx = -radius; dx <= radius; dx++)
      spatial.push([dx, dy, Math.exp(-(dx * dx + dy * dy) / (2 * sigmaS * sigmaS))]);
  const colorLUT = new Float32Array(3 * 255 * 255 + 1);
  for (let i = 0; i < colorLUT.length; i++) colorLUT[i] = Math.exp(-i / (2 * sigmaC * sigmaC));

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const o = (y * W + x) * 3;
      const r0 = src[o], g0 = src[o + 1], b0 = src[o + 2];
      let sr = 0, sg = 0, sb = 0, sw = 0;
      for (const [dx, dy, ws] of spatial) {
        const xx = Math.min(W - 1, Math.max(0, x + dx));
        const yy = Math.min(H - 1, Math.max(0, y + dy));
        const p = (yy * W + xx) * 3;
        const dr = src[p] - r0, dg = src[p + 1] - g0, db = src[p + 2] - b0;
        const w = ws * colorLUT[dr * dr + dg * dg + db * db];
        sr += src[p] * w; sg += src[p + 1] * w; sb += src[p + 2] * w; sw += w;
      }
      dst[o] = sr / sw; dst[o + 1] = sg / sw; dst[o + 2] = sb / sw;
    }
  }
  return dst;
}

// === FACE PROTECTION MASK (soft ellipse covering face + hair + ears) ===
function faceMask(face, crop, size) {
  const m = new Float32Array(size * size);
  if (!face) { m.fill(1); return m; }   // no face found → protect everything
  const k = size / crop.size;
  const cx = (face.x + face.w / 2 - crop.left) * k;
  const cy = (face.y + face.h * 0.42 - crop.top) * k;   // shift up for hair
  const rx = face.w * 0.85 * k, ry = face.h * 0.95 * k;
  for (let y = 0; y < size; y++)
    for (let x = 0; x < size; x++) {
      const d = Math.hypot((x - cx) / rx, (y - cy) / ry);   // 1.0 = ellipse edge
      // 1 inside, smooth fall-off to 0 between d = 1.0 and 1.35
      const t = Math.min(1, Math.max(0, (d - 1.0) / 0.35));
      m[y * size + x] = 1 - t * t * (3 - 2 * t);
    }
  return m;
}

// === FACE CORE (soft oval inside the detected face box) – never made transparent ===
function faceCore(face, crop, size) {
  const m = new Float32Array(size * size);
  if (!face) return m;
  const k = size / crop.size;
  const cx = (face.x + face.w / 2 - crop.left) * k;
  const cy = (face.y + face.h * 0.55 - crop.top) * k;
  const rx = face.w * 0.40 * k, ry = face.h * 0.42 * k;
  for (let y = 0; y < size; y++)
    for (let x = 0; x < size; x++) {
      const d = Math.hypot((x - cx) / rx, (y - cy) / ry);
      m[y * size + x] = Math.min(1, Math.max(0, (1.15 - d) / 0.3));
    }
  return m;
}

// === BUILD FINAL PIXELS ===
async function buildImage(inputPath) {
  const img = await sharp(inputPath).rotate().removeAlpha().toBuffer();  // honour EXIF
  const meta = await sharp(img).metadata();
  const W = meta.width, H = meta.height;

  const face = await detectFace(img, W, H);
  const crop = computeCrop(face, W, H);
  const S = CONFIG.outputSize;

  const cropBuf = await sharp(img)
    .extract({ left: crop.left, top: crop.top, width: crop.size, height: crop.size })
    .toBuffer();

  // High-quality Lanczos downscale
  const sharp1 = await sharp(cropBuf).resize(S, S, { kernel: 'lanczos3' }).raw().toBuffer();
  // Face: optional very light denoise (sensor noise wastes bytes and turns
  // into blotches after compression), then optional gentle sharpening.
  const FD = [null, [1, 0.8, 9], [2, 1.0, 13]][CONFIG.faceDenoise];
  let faceSrc = FD ? bilateral(sharp1, S, S, FD[0], FD[1], FD[2]) : sharp1;
  const sharpened = CONFIG.faceSharpen
    ? await sharp(faceSrc, { raw: { width: S, height: S, channels: 3 } })
        .sharpen({ sigma: 0.5, m1: 0.3, m2: 0.8 }).raw().toBuffer()
    : faceSrc;

  // Smoothed version for non-face regions
  const SM = [null, [1, 1.0, 18], [2, 1.6, 26], [3, 2.2, 36]][CONFIG.bodySmoothing];
  const smooth = SM ? bilateral(sharp1, S, S, SM[0], SM[1], SM[2]) : sharp1;

  const fmask = faceMask(face, crop, S);
  const coreMask = faceCore(face, crop, S);
  const alpha = CONFIG.removeBackground ? await segmentPerson(cropBuf, S) : null;

  return { S, face, ref: sharp1, coreMask, fmask, alpha, faceLayer: sharpened, smooth };
}

// Body simplification levels used when busy clothing eats the byte budget.
// Level 0 = edge-preserving smoothing only; higher = extra blur on clothes.
const BODY_BLUR = [0, 1.2, 2.5, 4.0];

async function bodyLayer(parts, level) {
  if (!level) return parts.smooth;
  return sharp(parts.smooth, { raw: { width: parts.S, height: parts.S, channels: 3 } })
    .blur(BODY_BLUR[level]).raw().toBuffer();
}

// Blend: face layer inside the face/hair oval, simplified body outside,
// flat background where the segmentation says "not person".
function compose(parts, body) {
  const { S, fmask, alpha, coreMask, faceLayer } = parts;
  const out = Buffer.alloc(S * S * 3);
  const bg = [CONFIG.background.r, CONFIG.background.g, CONFIG.background.b];
  for (let i = 0; i < S * S; i++) {
    const f = fmask[i];
    let a = 1;
    if (alpha) {
      // Sharpen the matte a little (cleaner edges, fewer grey halos) …
      const v = alpha[i] / 255;
      a = Math.min(1, Math.max(0, (v - 0.15) / 0.7));
      // … but never cut into the face itself
      a = Math.max(a, coreMask[i]);
    }
    for (let c = 0; c < 3; c++) {
      const fg = faceLayer[i * 3 + c] * f + body[i * 3 + c] * (1 - f);
      out[i * 3 + c] = Math.round(fg * a + bg[c] * (1 - a));
    }
  }
  return out;
}

// Build → encode; if the face would end up below minFaceQuality,
// simplify the clothing more and try again.
async function renderToTarget(inputPath) {
  const parts = await buildImage(inputPath);
  let res, level = 0;
  for (; level < BODY_BLUR.length; level++) {
    const pixels = compose(parts, await bodyLayer(parts, level));
    res = await encodeToTarget(pixels, parts.S);
    res.pixels = pixels;
    if (res.q >= CONFIG.minFaceQuality) break;
  }
  return { ...res, level: Math.min(level, BODY_BLUR.length - 1), parts };
}

// === ENCODE: highest quality that fits the size budget ===
async function encodeToTarget(pixels, S) {
  const maxBytes = CONFIG.targetKB * 1024;
  const enc = (q) => sharp(pixels, { raw: { width: S, height: S, channels: 3 } })
    .webp({ quality: q, effort: 6, ...CONFIG.webpOptions })
    .toBuffer();

  let lo = CONFIG.minQuality, hi = CONFIG.maxQuality, best = null;
  const top = await enc(hi);
  if (top.length <= maxBytes) return { buf: top, q: hi };
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const b = await enc(mid);
    if (b.length <= maxBytes) { best = { buf: b, q: mid }; lo = mid + 1; }
    else hi = mid - 1;
  }
  if (!best) best = { buf: await enc(CONFIG.minQuality), q: CONFIG.minQuality };
  return best;
}

async function processOne(file) {
  const inPath = path.join(CONFIG.inputDir, file);
  const outPath = path.join(CONFIG.outputDir, path.parse(file).name + '.webp');
  try {
    const { buf, q, level, parts } = await renderToTarget(inPath);
    const face = parts.face;
    await fs.writeFile(outPath, buf);
    const kb = buf.length / 1024;
    console.log(`${file} → ${kb.toFixed(2)} KB  q=${q}${level ? '  body-blur ' + level : ''}  ${face ? 'face ' + (face.conf * 100).toFixed(0) + '%' : 'NO FACE (fallback crop)'}`);
    return { ok: true, kb, face: !!face };
  } catch (e) {
    console.error(`${file}: ${e.message}`);
    return { ok: false };
  }
}

module.exports = { CONFIG, loadModels, buildImage, renderToTarget };

// === MAIN ===
if (require.main === module) (async () => {
  console.log('FACE-AWARE WEBP CONVERTER');
  console.log('=========================');
  console.log(`Output: ${CONFIG.outputSize}×${CONFIG.outputSize} | target ≤ ${CONFIG.targetKB} KB | background ${CONFIG.removeBackground ? 'removed → white' : 'kept'}\n`);

  await loadModels();
  await fs.mkdir(CONFIG.outputDir, { recursive: true });
  const files = (await fs.readdir(CONFIG.inputDir))
    .filter(f => ['.jpg', '.jpeg', '.png', '.webp'].includes(path.extname(f).toLowerCase()));
  if (!files.length) return console.log('No images found.');

  const t0 = Date.now();
  const results = [];
  let next = 0;
  await Promise.all(Array.from({ length: CONFIG.concurrency }, async () => {
    while (next < files.length) results.push(await processOne(files[next++]));
  }));

  const ok = results.filter(r => r.ok);
  const total = ok.reduce((s, r) => s + r.kb, 0);
  console.log('\nDONE');
  console.log(`Processed : ${ok.length}/${files.length}`);
  console.log(`No face   : ${ok.filter(r => !r.face).length} (used fallback crop)`);
  console.log(`Average   : ${(total / (ok.length || 1)).toFixed(2)} KB`);
  console.log(`Total     : ${(total / 1024).toFixed(2)} MB`);
  console.log(`Time      : ${((Date.now() - t0) / 1000).toFixed(1)} s`);
})();
