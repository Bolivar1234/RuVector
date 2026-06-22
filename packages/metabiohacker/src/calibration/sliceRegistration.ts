// Slice registration V0: estimate how well a predicted body mask aligns to the
// target body mask, as a centroid-offset error in pixels. A full intensity- or
// landmark-based registration is future work; this is enough to drive the
// honesty gate (large misalignment => exclude from headline metrics).

export type Mask = { width: number; height: number; data: Uint8Array | number[] };

function centroid(mask: Mask): { cx: number; cy: number; area: number } {
  let sx = 0, sy = 0, area = 0;
  for (let y = 0; y < mask.height; y++) {
    for (let x = 0; x < mask.width; x++) {
      if (mask.data[y * mask.width + x]) {
        sx += x;
        sy += y;
        area++;
      }
    }
  }
  return area ? { cx: sx / area, cy: sy / area, area } : { cx: mask.width / 2, cy: mask.height / 2, area: 0 };
}

export function estimateRegistrationErrorPx(predicted: Mask, target: Mask): number {
  const p = centroid(predicted);
  const t = centroid(target);
  if (p.area === 0 || t.area === 0) return Number.POSITIVE_INFINITY;
  return Math.hypot(p.cx - t.cx, p.cy - t.cy);
}

// A coarse boundary-complexity proxy (0..1): perimeter/area ratio of the target
// mask, normalised. Higher = more intricate soft-tissue boundaries = harder.
export function boundaryComplexity(target: Mask): number {
  let perimeter = 0, area = 0;
  const at = (x: number, y: number) =>
    x >= 0 && y >= 0 && x < target.width && y < target.height ? target.data[y * target.width + x] : 0;
  for (let y = 0; y < target.height; y++) {
    for (let x = 0; x < target.width; x++) {
      if (!at(x, y)) continue;
      area++;
      if (!at(x - 1, y) || !at(x + 1, y) || !at(x, y - 1) || !at(x, y + 1)) perimeter++;
    }
  }
  if (area === 0) return 1;
  // Circle has perimeter ~ 2*sqrt(pi*area); ratio>1 => more complex than a disk.
  const ideal = 2 * Math.sqrt(Math.PI * area);
  return Math.max(0, Math.min(1, perimeter / ideal - 0.5));
}
