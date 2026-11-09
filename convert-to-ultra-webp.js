const fs = require('fs').promises;
const path = require('path');
const os = require('os');
const sharp = require('sharp');
const { exec } = require('child_process');
const { promisify } = require('util');
const execAsync = promisify(exec);

// === CONFIGURATION ===
const inputDir = './23Allphotosnew';
// const outputDir = './23Allphotosnew_output';
const outputDir = './originaloutput';
const targetSize = 250;
const useCWebP = true; // Set false to skip cwebp

sharp.cache(false);

// === TEMP FILE IN OS TEMP DIR ===
async function getTempPngPath() {
  const tempDir = os.tmpdir();
  const filename = `img_${Date.now()}_${Math.random().toString(36).substr(2, 9)}.png`;
  return path.join(tempDir, filename);
}

// === CHECK cwebp ===
async function isCWebPAvailable() {
  try {
    await execAsync('cwebp -version');
    return true;
  } catch {
    return false;
  }
}

// === ESCAPE PATH (Windows safe) ===
function escapePath(p) {
  return `"${path.resolve(p).replace(/"/g, '\\"')}"`;
}

// === BACKGROUND REMOVAL (optional: keep for clean edges) ===
async function aggressiveBackgroundRemoval(imageBuffer) {
  try {
    const mask = await sharp(imageBuffer)
      .greyscale()
      .blur(2.5)
      .threshold(135)
      .negate()
      .toBuffer();

    const featheredMask = await sharp(mask)
      .blur(4)
      .toBuffer();

    return sharp(imageBuffer)
      .composite([{ input: featheredMask, blend: 'dest-in' }])
      .png()
      .toBuffer();
  } catch {
    console.warn("Background removal skipped");
    return imageBuffer;
  }
}

// === SMART CROP (preserve hair) ===
async function smartCropForHead(inputPath, metadata) {
  const { width, height } = metadata;
  let cropWidth, cropHeight, cropX, cropY;

  if (height >= width) {
    cropWidth = width * 0.92;
    cropHeight = cropWidth;
    cropY = Math.max(0, Math.floor(height * 0.02));
    cropX = Math.floor((width - cropWidth) / 2);
  } else if (width > height * 1.3) {
    cropHeight = height * 0.88;
    cropWidth = cropHeight;
    cropY = Math.floor(height * 0.08);
    cropX = Math.max(0, Math.floor((width - cropWidth) / 2));
  } else {
    const size = Math.min(width, height) * 0.91;
    cropWidth = cropHeight = size;
    cropY = Math.max(0, Math.floor(height * 0.03));
    cropX = Math.max(0, Math.floor((width - size) / 2));
  }

  cropX = Math.max(0, Math.min(cropX, width - cropWidth));
  cropY = Math.max(0, Math.min(cropY, height - cropHeight));

  return sharp(inputPath)
    .extract({
      left: Math.round(cropX),
      top: Math.round(cropY),
      width: Math.round(cropWidth),
      height: Math.round(cropHeight)
    })
    .toBuffer();
}

// === SHARP WEBP SETTINGS (opaque) ===
function getSharpWebPSettings() {
  return {
    quality: 85,
    effort: 6,
    smartSubsample: true,
    nearLossless: true,
    // No alphaQuality → no alpha
  };
}

// === cwebp ENCODING (opaque) ===
async function encodeWithCWebP(inputPngPath, outputWebpPath) {
  const input = escapePath(inputPngPath);
  const output = escapePath(outputWebpPath);

  const cmd = `cwebp -m 6 -q 75 -mt -af -sharp_yuv ${input} -o ${output}`;

  try {
    const { stderr } = await execAsync(cmd);
    if (stderr && !stderr.includes('Saved')) {
      console.warn(`cwebp warning: ${stderr.trim()}`);
    }
    return true;
  } catch (err) {
    console.warn(`cwebp failed: ${err.message}`);
    return false;
  }
}

// === PROCESS SINGLE IMAGE ===
async function processSingleImage(file, inputPath, outputPath) {
  let tempPngPath = null;

  try {
    tempPngPath = await getTempPngPath();

    const metadata = await sharp(inputPath).metadata();
    let processed = await smartCropForHead(inputPath, metadata);
    processed = await aggressiveBackgroundRemoval(processed);

    // === FINAL PIPELINE: Opaque, stripped, optimized ===
    processed = await sharp(processed)
      .resize(targetSize, targetSize, {
        fit: 'cover',
        position: sharp.gravity.north,
        background: { r: 255, g: 255, b: 255, alpha: 1 } // White background
      })
      .flatten({ background: { r: 255, g: 255, b: 255 } }) // Force opaque
      .removeAlpha()           // REMOVE ALPHA CHANNEL
      .png({ compressionLevel: 9, effort: 7, colors: 64 })
      .toBuffer();

    await fs.writeFile(tempPngPath, processed);

    let finalSizeKB = 0;
    let usedCWebP = false;

    if (useCWebP && await isCWebPAvailable()) {
      const success = await encodeWithCWebP(tempPngPath, outputPath);
      if (success) {
        const stat = await fs.stat(outputPath).catch(() => null);
        if (stat && stat.size > 0) usedCWebP = true;
      }
    }

    if (!usedCWebP) {
      await sharp(processed)
        .webp(getSharpWebPSettings())
        .toFile(outputPath);
    }

    finalSizeKB = (await fs.stat(outputPath)).size / 1024;
    const encoder = usedCWebP ? 'cwebp' : 'sharp';
    console.log(`${file} → ${finalSizeKB.toFixed(2)} KB [${encoder}]`);
    return { success: true, finalSize: finalSizeKB, usedCWebP };

  } catch (error) {
    console.error(`${file}: ${error.message}`);
    return { success: false, finalSize: 0 };
  } finally {
    if (tempPngPath) {
      try { await fs.unlink(tempPngPath); } catch {}
    }
  }
}

// === MAIN ===
(async () => {
  console.log('OPAQUE ULTRA WEBP CONVERTER');
  console.log('============================');
  console.log(`Target: ${targetSize}×${targetSize} | Opaque | No Metadata | ~1.8–3.5 KB`);
  console.log(`cwebp: ${useCWebP && await isCWebPAvailable() ? 'Available' : 'Using Sharp only'}`);
  console.log('');

  try {
    await fs.mkdir(outputDir, { recursive: true });
    const files = await fs.readdir(inputDir);
    const imageFiles = files.filter(f =>
      ['.jpg', '.jpeg', '.png', '.webp'].includes(path.extname(f).toLowerCase())
    );

    if (imageFiles.length === 0) {
      console.log('No images found in input directory.');
      return;
    }

    let totalSize = 0;
    let successCount = 0;
    let cwebpCount = 0;

    for (const file of imageFiles) {
      const inputPath = path.join(inputDir, file);
      const outputPath = path.join(outputDir, path.parse(file).name + '.webp');
      const result = await processSingleImage(file, inputPath, outputPath);
      if (result.success) {
        successCount++;
        totalSize += result.finalSize;
        if (result.usedCWebP) cwebpCount++;
      }
    }

    console.log('\nPROCESSING COMPLETE!');
    console.log(`Processed: ${successCount} images`);
    console.log(`cwebp used: ${cwebpCount}`);
    console.log(`Average size: ${(totalSize / successCount || 0).toFixed(2)} KB`);
    console.log(`Total size: ${totalSize.toFixed(1)} KB`);
    console.log('\nAll images are:');
    console.log('   • Opaque (no transparency)');
    console.log('   • Stripped of metadata');
    console.log('   • Optimized for web');

  } catch (error) {
    console.error('Script error:', error);
  }
})();