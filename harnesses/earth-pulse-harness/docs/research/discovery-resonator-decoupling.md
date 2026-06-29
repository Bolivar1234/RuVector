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

## Honest scope and caveats

- **One station, two years.** GT.DBIC only, 1996–1997. Not a global or
  multi-decadal claim. The seasonal *phase* of the 26 s amplitude was not robust
  across the two years (the annual-harmonic relative amplitude is only ~0.17),
  so we do **not** claim a clean seasonal cycle — only the decoupling and the
  frequency stability, which are robust.
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
