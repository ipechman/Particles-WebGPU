import assert from "node:assert/strict";
import test from "node:test";
import { Engine } from "../web/js/engine.js";
import { pcg, sampleCounter, fixedPoint, fullDepth, batchPoints, qualityFixtures,
  projectionBounds, collectImage, densityDistance } from "./helpers/sampling.mjs";

// BigInt arithmetic provides an independent overflow/unsigned-shift oracle.
function referencePcg(input) {
  const mask = 0xffffffffn;
  const state = (BigInt(input >>> 0) * 747796405n + 2891336453n) & mask;
  const word = (((state >> ((state >> 28n) + 4n)) ^ state) * 277803737n) & mask;
  return Number((word >> 22n) ^ word);
}

test("sampling oracle matches u32 PCG including unsigned overflow", () => {
  assert.deepEqual([0, 1, 2, 0xffffffff, 0x80000000].map(pcg),
    [129708002, 2831084092, 2055130248, 3861530882, 566699590]);
  for (let i = 0; i < 1000; i++) {
    const value = (Math.imul(i, 2654435761) + 0xfffff000) >>> 0;
    assert.equal(pcg(value), referencePcg(value));
  }
});

test("counter seeding removes the exact 4M batch 10/11 XOR permutation", () => {
  const count = 2 ** 22;
  const key10 = pcg(10 ^ 0x9e3779b9), key11 = pcg(11 ^ 0x9e3779b9);
  const permutation = (key10 ^ key11) >>> 0;
  // A low-bit-only XOR maps [0, 2^22) bijectively to itself. Thus the old
  // batches contain exactly the same hash inputs, independently of the IFS.
  assert.equal(permutation, 2005371);
  assert.ok(permutation > 0 && permutation < count);
  for (const i of [0, 1, count - 1, ...Array.from({ length: 1000 }, (_, j) => pcg(j) % count)]) {
    const duplicateIndex = i ^ permutation;
    assert.ok(duplicateIndex >= 0 && duplicateIndex < count);
    assert.equal(pcg(i ^ key10), pcg(duplicateIndex ^ key11));
    assert.notEqual(sampleCounter(i, 10, count), sampleCounter(duplicateIndex, 11, count));
  }
  // New input intervals are disjoint, including all scheduled 4M batches;
  // no 4-million-element allocation or probabilistic sampling is needed.
  for (let batch = 0; batch < 16; batch++) {
    assert.equal(sampleCounter(0, batch, count), batch * count);
    assert.equal(sampleCounter(count - 1, batch, count), (batch + 1) * count - 1);
    if (batch) assert.ok(sampleCounter(0, batch, count) > sampleCounter(count - 1, batch - 1, count));
  }
});

test("application fixed points agree with an independent linear solver", () => {
  const engine = new Engine({ width: 512, height: 512 });
  for (const { matrices } of qualityFixtures) {
    engine.transformData = new Float32Array(32 * 16);
    for (const [i, m] of matrices.entries()) engine.transformData.set(m, i * 16);
    const actual = engine._fixedPoints(matrices.length);
    for (const [i, m] of matrices.entries()) {
      const expected = fixedPoint(m);
      for (let axis = 0; axis < 3; axis++) assert.ok(Math.abs(actual[i * 4 + axis] - expected[axis]) < 1e-6);
    }
  }
});

test("continued batches are deterministic, finite, and advance each particle", () => {
  for (const { matrices } of qualityFixtures) {
    const particles = 1024;
    const first = batchPoints(matrices, particles, 0, fullDepth(matrices.length, particles));
    const next = batchPoints(matrices, particles, 1, 4, first);
    assert.deepEqual(next, batchPoints(matrices, particles, 1, 4, first));
    assert.notDeepEqual(first, next);
    assert.ok(next.every(Number.isFinite));
    let changed = 0;
    for (let i = 0; i < particles; i++) {
      if (next[i * 3] !== first[i * 3] || next[i * 3 + 1] !== first[i * 3 + 1] || next[i * 3 + 2] !== first[i * 3 + 2]) changed++;
    }
    assert.ok(changed > particles * 0.999);
  }
});

// This is a CPU sampling regression, not a GPU speed or visual-equivalence
// claim. Compare the actual instanced transform copies at equal point counts.
// Pixel occupancy catches holes/detail loss; coarse density catches a biased
// invariant distribution which occupancy alone can miss.
for (const { name, matrices } of qualityFixtures) {
  test(`four-hop detail and density remain near the full-depth reference: ${name}`, () => {
    const particles = 8192, batches = 8, resolution = 512;
    const bounds = projectionBounds(matrices);
    const reference = collectImage(matrices, particles, batches, 0, resolution, bounds);
    const continued = collectImage(matrices, particles, batches, 4, resolution, bounds);
    assert.equal(reference.clipped, 0);
    assert.equal(continued.clipped, 0);
    assert.ok(continued.covered / reference.covered > 0.98, `coverage ratio ${continued.covered / reference.covered}`);
    assert.ok(densityDistance(continued.density, reference.density) < 0.04);
    assert.ok(continued.affineOperations < reference.affineOperations * 0.45);
    // Sensitivity control: reusing one unchanged batch must fail the detail
    // requirement in these deliberately undersampled fixtures.
    const single = collectImage(matrices, particles, 1, 0, resolution, bounds);
    assert.ok(single.covered / reference.covered < 0.98);
  });
}
