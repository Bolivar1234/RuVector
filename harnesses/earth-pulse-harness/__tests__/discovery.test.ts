// SPDX-License-Identifier: MIT
// DISCOVERY (offline, deterministic): re-derive the headline numbers of
// docs/research/discovery-resonator-decoupling.md from the committed per-window
// metrics — real GT.DBIC measurements, 1996–1997. No network, no fabrication.

import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { resonanceStats, pearson, permutationP } from '../src/climatology.js';
import type { LineMetrics } from '../src/climatology.js';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const data = JSON.parse(
  readFileSync(resolve(root, 'data/seismic/dbic-climatology-1996-1997.json'), 'utf8'),
) as {
  n: number;
  windows: { peakFreqHz: number; periodS: number; lineExcess: number; snr: number; secondary: number; primary: number }[];
};

const metrics: LineMetrics[] = data.windows.map((w) => ({
  peakFreqHz: w.peakFreqHz, periodS: w.periodS, lineExcess: w.lineExcess,
  snr: w.snr, secondary: w.secondary, primary: w.primary,
}));

describe('discovery — 26 s pulse is a frequency-stable, decoupled resonance', () => {
  it('has a meaningful number of real windows', () => {
    expect(data.n).toBeGreaterThanOrEqual(40);
    expect(metrics.length).toBe(data.n);
  });

  it('the line frequency is stable (CV well under 2 %)', () => {
    const s = resonanceStats(metrics);
    expect(s.meanFreqHz).toBeGreaterThan(0.0355);
    expect(s.meanFreqHz).toBeLessThan(0.0366);
    expect(s.freqCv).toBeLessThan(0.02); // < 2 %
  });

  it('the amplitude varies by more than 10x', () => {
    expect(resonanceStats(metrics).amplitudeRange).toBeGreaterThan(10);
  });

  it('frequency does not track amplitude (fixed resonance, |r| < 0.4)', () => {
    expect(Math.abs(resonanceStats(metrics).freqAmpCorr)).toBeLessThan(0.4);
  });

  it('the line is decoupled from the secondary microseism (no significant correlation)', () => {
    const L = metrics.map((m) => m.lineExcess);
    const S = metrics.map((m) => m.secondary);
    const r = pearson(L, S);
    const p = permutationP(L, S, 26, 1000);
    // The key claim: NOT a strong positive correlation; consistent with zero.
    expect(Math.abs(r)).toBeLessThan(0.3);
    expect(p).toBeGreaterThan(0.05);
  });
});
