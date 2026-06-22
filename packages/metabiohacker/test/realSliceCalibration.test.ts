import { test } from "node:test";
import assert from "node:assert/strict";
import { diceByRegionFromLabels, regionDiceFromClassDice, meanRegionDice } from "../src/calibration/diceByRegion.ts";
import { scoreDomainGap, classifyRealSliceResult } from "../src/calibration/domainGapScoring.ts";
import { estimateRegistrationErrorPx, boundaryComplexity } from "../src/calibration/sliceRegistration.ts";

test("region Dice from labels: perfect match scores 1, mismatch < 1", () => {
  // classes: 0=water/fluid,1=fat,2=muscle,3=organ,4=bone
  const target = [0, 1, 2, 3, 4, 4];
  const perfect = diceByRegionFromLabels([...target], target);
  for (const r of perfect) assert.equal(r.dice, 1, `${r.region} perfect`);

  const wrong = diceByRegionFromLabels([0, 0, 0, 0, 0, 0], target);
  const bone = wrong.find((r) => r.region === "bone")!;
  assert.equal(bone.dice, 0, "bone absent in prediction => 0");
});

test("region Dice from the engine class-Dice array maps muscle+organ to softTissue", () => {
  // [fluid, fat, muscle, organ, bone] = abdomen example
  const scores = regionDiceFromClassDice([0.71, 0.52, 0.0, 0.22, 0.0]);
  const soft = scores.find((r) => r.region === "softTissue")!;
  assert.equal(soft.dice, 0.0); // min(muscle, organ) — conservative
  assert.ok(meanRegionDice(scores) < 0.45, "abdomen real slice is research-only territory");
});

test("domain gap is clamped to [0,1]", () => {
  assert.equal(scoreDomainGap({ registrationErrorPx: 1, targetBoundaryComplexity: 1, classImbalance: 1, missingAcousticEquivalent: 1 }), 1);
  assert.equal(scoreDomainGap({ registrationErrorPx: 0, targetBoundaryComplexity: 0, classImbalance: 0, missingAcousticEquivalent: 0 }), 0);
});

test("honesty gate prevents overclaiming", () => {
  // High registration error => exclude
  assert.equal(classifyRealSliceResult({ meanDice: 0.9, domainGapScore: 0.1, registrationErrorPx: 20 }), "exclude");
  // High domain gap => exclude
  assert.equal(classifyRealSliceResult({ meanDice: 0.9, domainGapScore: 0.7, registrationErrorPx: 2 }), "exclude");
  // Low Dice => research only
  assert.equal(classifyRealSliceResult({ meanDice: 0.3, domainGapScore: 0.2, registrationErrorPx: 2 }), "researchOnly");
  // Moderate gap => research only
  assert.equal(classifyRealSliceResult({ meanDice: 0.7, domainGapScore: 0.4, registrationErrorPx: 2 }), "researchOnly");
  // Clean => headline
  assert.equal(classifyRealSliceResult({ meanDice: 0.7, domainGapScore: 0.2, registrationErrorPx: 2 }), "headline");
});

test("registration error is centroid offset; misaligned masks score higher", () => {
  const w = 8, h = 8;
  const left = { width: w, height: h, data: new Uint8Array(w * h) };
  const right = { width: w, height: h, data: new Uint8Array(w * h) };
  for (let y = 2; y < 6; y++) {
    left.data[y * w + 1] = 1;
    left.data[y * w + 2] = 1;
    right.data[y * w + 5] = 1;
    right.data[y * w + 6] = 1;
  }
  assert.ok(estimateRegistrationErrorPx(left, left) < 0.001, "self-registration ~0");
  assert.ok(estimateRegistrationErrorPx(left, right) > 3, "shifted mask has offset");
  assert.ok(boundaryComplexity(left) >= 0 && boundaryComplexity(left) <= 1);
});
