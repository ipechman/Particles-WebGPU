import assert from "node:assert/strict";
import test from "node:test";
import { attractorBounds, projectedBox, buildViewPlan, packViewDraws, multiply64, viewWeights } from "../web/js/view-sampling.js";
import { mat4 } from "../web/js/math.js";
import { Engine } from "../web/js/engine.js";
import { qualityFixtures, fixedPoint, batchPoints, fullDepth } from "./helpers/sampling.mjs";
import { focusedPoints, projectPoints, sampleFit, centeredView } from "./helpers/view-quality.mjs";

for (const { name, matrices } of qualityFixtures) test(`certified bounds contain independent deep samples: ${name}`, () => {
  const bounds = attractorBounds(matrices, matrices.map(fixedPoint));
  assert.ok(bounds);
  const points = batchPoints(matrices, 4096, 99, 80);
  for (let i = 0; i < points.length; i++) assert.ok(points[i] >= bounds.lo[i % 3] && points[i] <= bounds.hi[i % 3]);
});

test("noncontractive, projective, and nonfinite maps report an explicit unsupported fallback", () => {
  for (const m of [mat4.identity(), new Float32Array(16).fill(NaN),
    new Float32Array([0.5, 0, 0, 1, 0, 0.5, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 1])]) {
    assert.equal(attractorBounds([m], [[0, 0, 0]]), null);
  }
  const plan = buildViewPlan({ bounds: null });
  assert.equal(plan.active, false); assert.equal(plan.reason, "uncertified bounds");
});

test("homogeneous culling preserves near-plane intersections and rejects offscreen/behind boxes", () => {
  const projection = mat4.perspective(Math.PI / 3, 1, 0.1, 100);
  assert.ok(projectedBox(projection, { lo: [-1, -1, -1], hi: [1, 1, 1] }, 512, 512));
  assert.equal(projectedBox(projection, { lo: [-1, -1, 2], hi: [1, 1, 3] }, 512, 512), null);
  assert.equal(projectedBox(projection, { lo: [20, 20, -2], hi: [21, 21, -1] }, 512, 512), null);
});

for (const { name, matrices } of qualityFixtures) test(`Focus activates in ordinary centered views: ${name}`, () => {
  const n = 16384;
  const points = batchPoints(matrices, n, 0, fullDepth(matrices.length, n));
  const fit = sampleFit(points), seeds = matrices.map(fixedPoint), bounds = attractorBounds(matrices, seeds);
  for (const distance of [4.5, 2, 1, 0.5, 0.2]) {
    const plan = buildViewPlan({ matrices, bounds, viewFit: multiply64(centeredView(distance), fit), width: 640, height: 480, particles: n * matrices.length });
    assert.equal(plan.active, true, `Focus silently disabled at distance ${distance}`);
    assert.ok(plan.leaves.length <= 1024 && plan.visited <= 8192);
  }
});

for (const { name, matrices } of qualityFixtures) test(`same drawing budget improves normal zoom coverage: ${name}`, () => {
  const n = 16384, points = batchPoints(matrices, n, 0, fullDepth(matrices.length, n));
  // Normal centered wheel zoom; no target movement or special corner aim.
  const distance = name.startsWith("procedural") ? 2 : 1;
  const viewFit = multiply64(centeredView(distance), sampleFit(points));
  const seeds = matrices.map(fixedPoint);
  const plan = buildViewPlan({ matrices, bounds: attractorBounds(matrices, seeds), viewFit,
    width: 640, height: 480, particles: n * matrices.length });
  const weights = viewWeights(plan.leaves, { matrices, seeds, viewFit, width: 640, height: 480 });
  assert.ok(weights.every(w => Number.isFinite(w) && w > 0), "Pilot data must never remove a branch");
  const global = projectPoints(points, matrices, viewFit, 640, 480);
  const focused = projectPoints(focusedPoints(matrices, plan, n, 0, weights), [mat4.identity()], viewFit, 640, 480);
  assert.equal(focused.submitted, global.submitted);
  assert.ok(focused.covered > global.covered * 1.05, `${name}: ${focused.covered} vs ${global.covered}`);
  console.log(JSON.stringify({ name, distance, particles: n, vertices: focused.submitted,
    globalPixels: global.covered, focusedPixels: focused.covered, ratio: focused.covered / global.covered }));
});

test("draw partitions cover exactly N*M with valid base indices, including 100M", () => {
  const leaves = [1, 2, 3, 4, 5].map(mass => ({ matrix: mat4.identity(), mass }));
  for (const n of [64, 509, 100000000]) for (const count of [2, 3, 8, 32]) {
    const { bytes, draws, vertices } = packViewDraws(leaves, n, count, mat4.identity(), leaves.map(l => l.mass));
    assert.equal(vertices, n * count);
    assert.equal(draws.reduce((sum, d) => sum + d.count, 0), vertices);
    assert.ok(draws.length <= leaves.length + count - 1);
    assert.ok(bytes.byteLength <= (1024 + 32) * 80);
    let expected = 0;
    for (const d of draws) {
      assert.equal(d.copy * n + d.first, expected);
      assert.ok(d.first >= 0 && d.first + d.count <= n && d.count > 0);
      expected += d.count;
    }
  }
});

test("fit readback is bounded and discards results from old geometry", async () => {
  const e = new Engine({}); let complete; let mapped = 0, unmapped = 0, copied = 0;
  const result = mat4.identity();
  e.fitReadback = { mapAsync: () => { mapped++; return new Promise(resolve => { complete = resolve; }); },
    getMappedRange: () => result.buffer, unmap: () => { unmapped++; } };
  const oldMode = globalThis.GPUMapMode; globalThis.GPUMapMode = { READ: 1 };
  try {
    const enc = { copyBufferToBuffer: () => { copied++; } };
    const revision = e._encodeFitReadback(enc);
    assert.equal(e._encodeFitReadback(enc), null); e._readFit(revision); e._fitRevision++;
    complete(); await new Promise(resolve => setImmediate(resolve));
    assert.equal(e._fitCPU, undefined); assert.equal(e._fitPending, false);
    e._readFit(e._encodeFitReadback(enc)); complete(); await new Promise(resolve => setImmediate(resolve));
    assert.deepEqual(e._fitCPU, result); assert.equal(e._encodeFitReadback(enc), null);
    assert.deepEqual([mapped, unmapped, copied], [2, 2, 2]);
  } finally { globalThis.GPUMapMode = oldMode; }
});
