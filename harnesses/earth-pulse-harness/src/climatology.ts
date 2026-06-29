// SPDX-License-Identifier: MIT
// Climatology / resonance analysis for the 26-second pulse. Pure, deterministic.
//
// These are the measurements behind the harness's first real-data discovery
// (docs/research/discovery-resonator-decoupling.md): across many windows we ask
//   - how STABLE is the line frequency? (resonator signature)
//   - how much does the AMPLITUDE vary, and does it shift the frequency?
//   - does the amplitude track the secondary (ocean-wave) microseism?
// All from real GT.DBIC observations — no fabrication.

import { welchPsd } from './spectrum.js';

export interface LineMetrics {
  /** Peak frequency of the line in the long-period band. */
  peakFreqHz: number;
  periodS: number;
  /** PSD at the line peak minus the local continuum — the 26 s source strength. */
  lineExcess: number;
  /** Line peak / continuum (a per-window SNR). */
  snr: number;
  /** Mean PSD in the secondary (double-frequency) microseism band — ocean-wave proxy. */
  secondary: number;
  /** Mean PSD in the primary microseism band. */
  primary: number;
}

export interface LineBands {
  linePeak?: [number, number];   // search band for the line peak
  continuum?: [[number, number], [number, number]]; // two sidebands for the background
  secondary?: [number, number];
  primary?: [number, number];
}

const DEFAULT_BANDS: Required<LineBands> = {
  linePeak: [0.0352, 0.0372],
  continuum: [[0.0305, 0.0340], [0.0390, 0.0425]],
  secondary: [0.10, 0.20],
  primary: [0.05, 0.08],
};

/** Measure the 26 s line + microseism bands for one window of samples. */
export function lineMetrics(
  samples: number[] | Float64Array,
  fs = 1.0,
  bands: LineBands = {},
): LineMetrics {
  const b = { ...DEFAULT_BANDS, ...bands };
  const s = welchPsd(samples, { fs, segment: 8192, overlap: 4096, average: 'median' });
  const avg = (lo: number, hi: number): number => {
    let sum = 0;
    let n = 0;
    for (let k = 0; k < s.freqs.length; k++) {
      const f = s.freqs[k];
      if (f >= lo && f <= hi) {
        sum += s.psd[k];
        n++;
      }
    }
    return n ? sum / n : 0;
  };
  let pk = 0;
  let pf = 0;
  for (let k = 0; k < s.freqs.length; k++) {
    const f = s.freqs[k];
    if (f >= b.linePeak[0] && f <= b.linePeak[1] && s.psd[k] > pk) {
      pk = s.psd[k];
      pf = f;
    }
  }
  const bg = (avg(b.continuum[0][0], b.continuum[0][1]) + avg(b.continuum[1][0], b.continuum[1][1])) / 2;
  return {
    peakFreqHz: pf,
    periodS: pf > 0 ? 1 / pf : 0,
    lineExcess: Math.max(0, pk - bg),
    snr: bg > 0 ? pk / bg : 0,
    secondary: avg(b.secondary[0], b.secondary[1]),
    primary: avg(b.primary[0], b.primary[1]),
  };
}

export interface ResonanceStats {
  windows: number;
  meanFreqHz: number;
  stdFreqHz: number;
  /** Coefficient of variation of the line frequency (σ/μ). Small ⇒ fixed resonator. */
  freqCv: number;
  amplitudeRange: number;   // max/min line excess
  /** Correlation of frequency with amplitude. ~0 ⇒ frequency independent of drive. */
  freqAmpCorr: number;
}

/** Resonance statistics over a set of per-window line metrics. */
export function resonanceStats(metrics: LineMetrics[]): ResonanceStats {
  const f = metrics.map((m) => m.peakFreqHz).filter((x) => x > 0);
  const ex = metrics.map((m) => m.lineExcess).filter((x) => x > 0);
  const meanF = mean(f);
  const stdF = std(f);
  return {
    windows: metrics.length,
    meanFreqHz: meanF,
    stdFreqHz: stdF,
    freqCv: meanF > 0 ? stdF / meanF : 0,
    amplitudeRange: Math.min(...ex) > 0 ? Math.max(...ex) / Math.min(...ex) : 0,
    freqAmpCorr: pearson(metrics.map((m) => m.peakFreqHz), metrics.map((m) => m.lineExcess)),
  };
}

export function mean(a: number[]): number {
  return a.reduce((s, v) => s + v, 0) / a.length;
}

export function std(a: number[]): number {
  const m = mean(a);
  return Math.sqrt(mean(a.map((v) => (v - m) ** 2)));
}

/** Pearson correlation coefficient. */
export function pearson(a: number[], b: number[]): number {
  const n = a.length;
  const ma = mean(a);
  const mb = mean(b);
  let nu = 0;
  let da = 0;
  let db = 0;
  for (let i = 0; i < n; i++) {
    nu += (a[i] - ma) * (b[i] - mb);
    da += (a[i] - ma) ** 2;
    db += (b[i] - mb) ** 2;
  }
  return nu / Math.sqrt(da * db);
}

/**
 * Two-sided permutation p-value for |corr(a,b)| using a seeded shuffle (no
 * Math.random — deterministic and reproducible). A large p means the observed
 * correlation is consistent with no relationship.
 */
export function permutationP(a: number[], b: number[], seed = 26, iters = 2000): number {
  const r0 = Math.abs(pearson(a, b));
  let s = seed >>> 0 || 1;
  const rnd = (): number => {
    s = (Math.imul(s, 1103515245) + 12345) & 0x7fffffff;
    return s / 0x7fffffff;
  };
  let ge = 0;
  for (let it = 0; it < iters; it++) {
    const sh = [...b];
    for (let j = sh.length - 1; j > 0; j--) {
      const k = Math.floor(rnd() * (j + 1));
      [sh[j], sh[k]] = [sh[k], sh[j]];
    }
    if (Math.abs(pearson(a, sh)) >= r0) ge++;
  }
  return ge / iters;
}

export interface SeasonalPhase {
  peakMonth: number; // 1..12, month of the annual-harmonic maximum
  relAmplitude: number; // annual-harmonic amplitude / mean (how seasonal it is)
}

/**
 * Fit y = a + b·cos(ωm) + c·sin(ωm), ω=2π/12, to a 1..12 monthly series and
 * return the month of maximum and the relative seasonal amplitude.
 */
export function seasonalPhase(monthly: Record<number, number>): SeasonalPhase {
  const w = (2 * Math.PI) / 12;
  const A = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  const B = [0, 0, 0];
  for (const [mStr, y] of Object.entries(monthly)) {
    const m = Number(mStr);
    const xs = [1, Math.cos(w * m), Math.sin(w * m)];
    for (let i = 0; i < 3; i++) {
      B[i] += xs[i] * y;
      for (let j = 0; j < 3; j++) A[i][j] += xs[i] * xs[j];
    }
  }
  const det = m3(A);
  const sol = [0, 1, 2].map((c) => m3(A.map((row, i) => row.map((v, j) => (j === c ? B[i] : v)))) / det);
  const peak = (((Math.atan2(sol[2], sol[1]) / w) % 12) + 12) % 12;
  const amp = Math.hypot(sol[1], sol[2]);
  return { peakMonth: peak + 1, relAmplitude: sol[0] !== 0 ? amp / sol[0] : 0 };
}

function m3(M: number[][]): number {
  return (
    M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) -
    M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
    M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0])
  );
}
