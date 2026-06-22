# MetaBioHacker reconstruction benchmark

Frozen engine: `sonic_ct_serve`. Dataset: 13 samples
(12 reproducible synthetic phantoms + 1 real abdominal CT slice from Wikimedia Commons). Only the harness config differs between rows.

| Config | Dice (synthetic) | Acoustic residual | Latency (ms) | Dice (real CT) |
|--------|------------------|-------------------|--------------|----------------|
| baseline | 0.543 ± 0.004 | 0.028 | 420 | 0.265 |
| evolved | 0.545 ± 0.006 | 0.028 | 163 | 0.265 |

**Evolved vs baseline:** shape +0.4%, latency 157.4% faster, residual −0.1%.

Real-data note: the real CT slice is fetched on demand by `tools/fetchRealSlice.mjs`
(not committed); intensity is banded into the five acoustic classes as a proxy
ground truth. Real anatomy is harder than synthetic phantoms, so its Dice is
lower — an honest measure, not a regression.
