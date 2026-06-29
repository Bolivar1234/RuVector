# Discovery — the 26 s pulse is a frequency-stable resonance decoupled from the ocean-wave microseism

**Finding (real data, GT.DBIC, 1996–1997, n = 57 windows):** the 26-second
microseism behaves like a **fixed-frequency resonator driven at variable
strength**, and its strength is **statistically decoupled** from the local
secondary (ocean wave–wave) microseism.

> This is an empirical result the harness derived autonomously from real
> observations, with a falsification test, not a restatement of an input. It is
> a single-station, two-year characterization — see the honest scope below.
>
> Reproduce: `npm run build && node scripts/climatology.mjs 1996,1997`
> Offline (committed metrics): `npm test` → `__tests__/discovery.test.ts`

## The three measurements

All from real GT.DBIC LHZ data (boreal coverage 1996–1997), 57 independent
2-day windows, each reduced to a median Welch PSD; the 26 s line is the peak in
0.0352–0.0372 Hz and its **excess power** is that peak minus the local
continuum. Raw counts; instrument gain is constant within the station epoch, so
relative comparisons are valid.

### 1. The frequency is stable to ~0.6 %

```
mean f0 = 0.03606 Hz  (period 27.73 s)
std     = 2.13e-4 Hz
CV (σ/μ) = 0.59 %     across 57 windows over 2 years
```

The line sits at the same frequency, window after window, season after season.

### 2. The amplitude is wildly variable — 36×

```
line excess power: min 3.4e3 … max 1.2e5   →   range 36.5×
```

So the *strength* of the 26 s pulse changes by more than an order of magnitude
while its *frequency* barely moves.

### 3. Frequency does not follow amplitude (fixed resonance)

```
corr(peak frequency, line amplitude) = +0.165   (n = 57)
```

Near zero: driving the line harder does **not** shift its frequency. A fixed
resonant frequency excited at variable strength is exactly the resonator
signature — as opposed to a broadband source whose peak would wander with the
forcing.

### 4. Decoupled from the local ocean-wave field

```
corr(26 s line excess, secondary microseism) = 0.04
permutation p-value (2000 shuffles, seeded)   = 0.75   (n = 57)
```

The secondary (double-frequency) microseism is the standard proxy for local
ocean-wave energy, and it is strongly seasonal here (peaks boreal winter). The
26 s line's 36× amplitude swing **does not track it at all** — the correlation
is consistent with zero. If the 26 s pulse were simply another ocean-wave
microseism from the local sea state, you would expect a clear positive
correlation. You do not see one.

## Why this matters

Putting the four numbers together gives a coherent physical picture:

> **A fixed-frequency resonance (CV 0.59 %) whose excitation varies > 36×,
> independently of both its own frequency and the local ocean-wave field.**

This favors the **"stable carrier, externally modulated"** model over "the 26 s
pulse is just a local double-frequency microseism": the resonant *frequency* is
set by a fixed structure (shelf / water-column / crustal / source geometry),
while the *amplitude* is gated by something other than the bulk local wave
energy — a specific directional swell reaching a specific resonator, or a
non-wave driver. It is consistent with, and an independent quantitative support
for, the long-standing view that the 26 s signal is not an ordinary microseism
(the gap Bruland & Hadziioannou 2023 point to).

It also sharpens the next test on the discovery ladder (ADR-004): if the driver
is directional swell rather than bulk wave energy, the 26 s amplitude should
correlate with swell *direction/source-region* state, not with local microseism
power — exactly the ruVector nearest-neighbor query ADR-002/ADR-003 set up.

## Replication and refinement (adversarial follow-up)

Three challenges were run against the result; it survived all three.
Consolidated metrics: `data/seismic/dbic-replication-1995-1998.json`.

### Refinement — the line is at 27.7 s, not 26.0 s

The "26-second" label does not match this station. Searching the **wide** band
0.0340–0.0400 Hz (which *includes* the canonical 26.0 s = 0.03846 Hz) at high
resolution (6.1×10⁻⁵ Hz, 288 h record), with parabolic peak interpolation:

```
dominant line: f0 = 0.03607 Hz  (27.72 s)   prominence 2.05x
prominence at 26.0 s (0.03846 Hz) = 0.82x   ← below background; NOT a peak
prominence at 27.7 s (0.03610 Hz) = 2.05x   ← the real line
```

So 27.7 s is not a band-edge or binning artifact: at GT.DBIC the dominant
long-period line genuinely sits at **27.7 s**, and the canonical 26.0 s shows no
excess. We report this as the **27.7 s pulse** for this station/epoch and flag
the ~6 % offset from the popular "26 s" figure for cross-station confirmation.

### Replication — same behavior across 4 independent years

Per-year statistics (GT.DBIC, 2-day windows, 23–31 windows/year):

| year | n | freq CV | amp range | corr(freq, amp) | corr(line, secondary) | perm p |
|------|---|---------|-----------|-----------------|-----------------------|--------|
| 1995 | 31 | 0.67 % | 11.1× | 0.06 | −0.21 | 0.26 |
| 1996 | 28 | 0.64 % | 13.6× | 0.08 | −0.14 | 0.48 |
| 1997 | 29 | 0.53 % | 36.5× | 0.25 | +0.25 | 0.18 |
| 1998 | 23 | 0.65 % | 5.2× | 0.03 | +0.12 | 0.59 |

Every year independently shows: frequency stable to ~0.6 %, amplitude variable,
frequency–amplitude decoupled, and **no significant correlation with the
secondary microseism** (p > 0.18 each year; the sign even flips year to year).
The result is not a one-off.

### Calm-sea "gold samples" — the strongest pulses in the quietest seas

Windows with the 26 s line in the top third of strength **and** the secondary
microseism in the bottom third (10 of 111 windows). The single strongest 26 s
window (1996-06-22, excess 8.1×10⁴) occurred while local seas were quiet
(secondary 1.8×10⁶); conversely the loudest seas (boreal-winter storms,
secondary up to 1.1×10⁷) coincided with the *weakest* 26 s. If the pulse were
local-ocean-wave-driven this pattern should not exist. It is the cleanest
single-window evidence for a non-local-ocean driver.

## Honest scope and caveats

- **One station, four years.** GT.DBIC only, 1995–1998 (replicated per-year).
  Not a global or multi-decadal claim, and not yet cross-station — the next step
  is to confirm the 27.7 s line and its decoupling at independent stations.
  The seasonal *phase* of the 26 s amplitude was **not** robust across years
  (annual-harmonic relative amplitude only ~0.17), so we make no seasonal-cycle
  claim — only the frequency stability, amplitude independence, and decoupling,
  which replicate.
- **Null correlation ≠ proof of zero coupling.** With n = 57 the correlation's
  95 % interval still admits weak coupling; what is ruled out is a *strong*
  positive correlation (the signature of common ocean-wave forcing).
- **"Resonator" is an inference** from frequency stability + amplitude
  independence, not a direct mechanical measurement. The true resonance Q would
  need a linewidth analysis; here we report frequency *stability* (CV), which
  bounds it.
- **Amplitude is raw-count PSD excess.** Valid for relative comparison within
  one instrument epoch; not an absolute ground-motion amplitude.

## Artifacts

| File | Contents |
|---|---|
| `data/seismic/dbic-climatology-1996-1997.json` | the 57 per-window metrics + the resonance/decoupling statistics |
| `scripts/climatology.mjs` | reproducible fetch + analysis from IRIS |
| `src/climatology.ts` | `lineMetrics`, `resonanceStats`, `pearson`, `permutationP`, `seasonalPhase` |
| `__tests__/discovery.test.ts` | offline re-derivation of the headline numbers from the committed metrics |

## References

- Bruland, A. & Hadziioannou, C. (2023). Gliding tremors associated with the
  26 s microseism.
- General microseism theory: primary vs. secondary (double-frequency) microseisms.
- Data: IRIS/EarthScope FDSN web services, station GT.DBIC.
