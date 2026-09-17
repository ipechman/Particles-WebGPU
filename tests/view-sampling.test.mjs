import assert from "node:assert/strict";
import test from "node:test";
import { attractorBounds, projectedBox, buildViewPlan, packViewLeaves, multiply64, VIEW_LEAF_BYTES } from "../web/js/view-sampling.js";
import { mat4 } from "../web/js/math.js";
import { qualityFixtures, fixedPoint, batchPoints, fullDepth } from "./helpers/sampling.mjs";
import { focusedPoints, projectPoints } from "./helpers/view-quality.mjs";

for (const { name, matrices } of qualityFixtures) test(`certified bounds contain independent deep samples: ${name}`, () => {
  const bounds = attractorBounds(matrices, matrices.map(fixedPoint));
  assert.ok(bounds);
  const points = batchPoints(matrices, 4096, 99, 80);
  for (let i = 0; i < points.length; i++) {
    assert.ok(points[i] >= bounds.lo[i % 3] && points[i] <= bounds.hi[i % 3]);
  }
});

test("noncontractive, projective, and nonfinite maps safely keep global sampling", () => {
  for (const m of [mat4.identity(), new Float32Array(16).fill(NaN),
    new Float32Array([0.5, 0, 0, 1, 0, 0.5, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 1])]) {
    assert.equal(attractorBounds([m], [[0, 0, 0]]), null);
  }
  assert.equal(buildViewPlan({ bounds: null }).active, false);
});

test("homogeneous culling preserves near-plane intersections and rejects offscreen/behind boxes", () => {
  const projection = mat4.perspective(Math.PI / 3, 1, 0.1, 100);
  assert.ok(projectedBox(projection, { lo: [-1, -1, -1], hi: [1, 1, 1] }, 512, 512));
  assert.equal(projectedBox(projection, { lo: [-1, -1, 2], hi: [1, 1, 3] }, 512, 512), null);
  assert.equal(projectedBox(projection, { lo: [20, 20, -2], hi: [21, 21, -1] }, 512, 512), null);
});

test("frontier preserves total mass at overview and obeys node, leaf, and depth budgets", () => {
  const { matrices } = qualityFixtures[0];
  const bounds = attractorBounds(matrices, matrices.map(fixedPoint));
  const viewFit = multiply64(mat4.perspective(Math.PI / 3, 1, 0.01, 100),
    mat4.lookAt([0, 0, 3], [0, 0, 0], [0, 1, 0]));
  const plan = buildViewPlan({ matrices, bounds, viewFit, width: 512, height: 512,
    particles: 509, maxLeaves: 8, maxNodes: 40, maxDepth: 2 });
  assert.equal(plan.active, false);
  assert.ok(Math.abs(plan.visibleMass - 1) < 1e-12);
  assert.ok(plan.leaves.length <= 8 && plan.visited <= 40);
  assert.ok(plan.leaves.every(l => l.depth <= 2));
  const packed = packViewLeaves(plan.leaves, 509);
  let previous = 0;
  for (let i = 0; i < plan.leaves.length; i++) {
    const end = new Uint32Array(packed, i * VIEW_LEAF_BYTES + 64, 1)[0];
    assert.ok(end > previous); previous = end;
  }
  assert.equal(previous, Math.ceil(509 / 64));
});

for (const name of ["triangle-2d", "pyramid-3d", "vicsek-3d"]) test(`equal-budget close-up covers more pixels: ${name}`, () => {
  const { matrices } = qualityFixtures.find(f => f.name === name);
  const target = fixedPoint(matrices[0]);
  const offset = name === "pyramid-3d" ? [0.04, 0.04, 0.09] : [0.015, 0.01, 0.09];
  const eye = target.map((v, i) => v + offset[i]);
  const view = multiply64(mat4.perspective(Math.PI / 3, 1, 0.0001, 100), mat4.lookAt(eye, target, [0, 1, 0]));
  const particles = 16384;
  const plan = buildViewPlan({ matrices, bounds: attractorBounds(matrices, matrices.map(fixedPoint)),
    viewFit: view, width: 512, height: 512, particles });
  assert.equal(plan.active, true);
  assert.ok(plan.visited <= 8192 && plan.leaves.length <= 1024);
  const global = projectPoints(batchPoints(matrices, particles, 0, fullDepth(matrices.length, particles)), matrices, view, 512, 512);
  const focused = projectPoints(focusedPoints(matrices, plan, particles), [mat4.identity()], view, 512, 512);
  assert.ok(focused.covered > global.covered * 1.5, `${focused.covered} vs ${global.covered}`);
  assert.ok(focused.onScreen / particles > global.onScreen / global.submitted * 3);
  assert.equal(focused.submitted, particles);
  console.log(JSON.stringify({ name, global: global.covered, focused: focused.covered,
    coverageRatio: focused.covered / global.covered, onScreenRatio: focused.onScreen / particles,
    visibleMass: plan.visibleMass, leaves: plan.leaves.length }));
});

test("overlapping 3D overview keeps the denser global draw", () => {
  const { matrices } = qualityFixtures.find(f => f.name === "pyramid-3d");
  const target = fixedPoint(matrices[0]);
  const eye = target.map((v, i) => v + [-0.04, -0.04, 0.06][i]);
  const viewFit = multiply64(mat4.perspective(Math.PI / 3, 1, 0.0001, 100), mat4.lookAt(eye, target, [0, 1, 0]));
  const plan = buildViewPlan({ matrices, bounds: attractorBounds(matrices, matrices.map(fixedPoint)),
    viewFit, width: 512, height: 512, particles: 16384 });
  assert.equal(plan.active, false);
});
