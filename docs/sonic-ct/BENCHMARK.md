# MetaBioHacker reconstruction benchmark

Frozen engine: `sonic_ct_serve`. Dataset: 42 samples
(40 reproducible synthetic phantoms + 1 real abdominal CT slice from Wikimedia Commons). Only the harness config differs between rows.

Statistics over 40 synthetic samples (mean ± 95% CI) and
2 real CT slice(s).

| Config | Dice (synthetic, 95% CI) | Acoustic residual | Latency (ms) | Dice (real CT) |
|--------|--------------------------|-------------------|--------------|----------------|
| baseline | 0.543 ± 0.002 | 0.028 | 431 | 0.300 |
| evolved | 0.543 ± 0.002 | 0.028 | 173 | 0.300 |

**Evolved vs baseline:** shape +0.1%, latency 149.3% faster, residual +0.4%.

Real-data note: the real CT slice is fetched on demand by `tools/fetchRealSlice.mjs`
(not committed); intensity is banded into the five acoustic classes as a proxy
ground truth. Real anatomy is harder than synthetic phantoms, so its Dice is
lower — an honest measure, not a regression.
