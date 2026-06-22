// Fetch a real public anatomical CT slice and convert it to a grayscale PGM the
// Rust engine can ingest as a ground-truth phantom.
//
// Source: Wikimedia Commons "Axial-Abdomen(L200)(W600).jpg" (an abdominal CT
// window). The image itself is NOT committed; it is fetched on demand and the
// derived low-resolution PGM is written to public/benchmark/ (gitignored).
// Verify the file's license on Commons before any redistribution.
//
// Usage: node tools/fetchRealSlice.mjs [n]

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import jpeg from "jpeg-js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const N = Number(process.argv[2] || 96);
const URL =
  "https://commons.wikimedia.org/wiki/Special:FilePath/Axial-Abdomen(L200)(W600).jpg";
const OUT = path.join(__dirname, "..", "public", "benchmark", "real_abdomen.pgm");

const resp = await fetch(URL);
if (!resp.ok) {
  console.error(`fetch failed: ${resp.status}`);
  process.exit(1);
}
const buf = Buffer.from(await resp.arrayBuffer());
const img = jpeg.decode(buf, { useTArray: true });
console.log(`decoded ${img.width}x${img.height}`);

// Center-crop to a square, then area-average downsample to N x N grayscale.
const side = Math.min(img.width, img.height);
const ox = ((img.width - side) / 2) | 0;
const oy = ((img.height - side) / 2) | 0;
const gray = new Uint8Array(N * N);
for (let y = 0; y < N; y++) {
  for (let x = 0; x < N; x++) {
    let acc = 0, cnt = 0;
    const x0 = ox + ((x * side) / N) | 0;
    const x1 = ox + (((x + 1) * side) / N) | 0;
    const y0 = oy + ((y * side) / N) | 0;
    const y1 = oy + (((y + 1) * side) / N) | 0;
    for (let yy = y0; yy < y1; yy++) {
      for (let xx = x0; xx < x1; xx++) {
        const i = (yy * img.width + xx) * 4;
        acc += (img.data[i] + img.data[i + 1] + img.data[i + 2]) / 3;
        cnt++;
      }
    }
    gray[y * N + x] = cnt ? Math.round(acc / cnt) : 0;
  }
}

// Write binary PGM (P5). Flip Y so it matches the engine's bottom-up grids.
const header = Buffer.from(`P5\n${N} ${N}\n255\n`, "ascii");
const body = Buffer.alloc(N * N);
for (let y = 0; y < N; y++) {
  for (let x = 0; x < N; x++) body[y * N + x] = gray[(N - 1 - y) * N + x];
}
fs.writeFileSync(OUT, Buffer.concat([header, body]));
console.log(`wrote ${OUT} (${N}x${N})`);
