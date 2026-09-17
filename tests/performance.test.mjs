import assert from "node:assert/strict";
import test from "node:test";
import { dispatchShape, lightingSchedule, FrameRate, PARTICLE_STRIDE } from "../web/js/performance.js";
import { Engine } from "../web/js/engine.js";

test("dispatch covers every particle with bounded dimensions and less than one row of waste", () => {
  for (const n of [1, 63, 64, 65, 262144, 4194304, 8388608, 16777216, 100000000]) {
    const { x, y, width } = dispatchShape(n, 64, 65535);
    const required = Math.ceil(n / 64);
    assert.ok(x <= 65535 && y <= 65535);
    assert.ok(x * y >= required && x * y - required < y);
    assert.equal(width, x * 64);
    // The shader's flattened coordinates cover a contiguous, unique range.
    assert.equal((y - 1) * width + (x * 64 - 1), x * y * 64 - 1);
  }
  assert.deepEqual(dispatchShape(4194304, 64, 65535), { x: 32768, y: 2, width: 2097152 });
  assert.throws(() => dispatchShape(1000, 1, 4), RangeError);
});

test("lighting budgets keep sample indices unique, distributed, in bounds, and below budget", () => {
  for (const n of [65, 262144, 4194305, 100000000]) {
    for (const budget of [1, 13, 524288, 2097152, Infinity]) {
      const { count, stride } = lightingSchedule(n, budget);
      assert.ok(count <= budget && count <= n);
      assert.ok((count - 1) * stride < n);
      assert.ok(count * stride >= n);
      if (budget >= n) assert.deepEqual({ count, stride }, { count: n, stride: 1 });
    }
  }
});

test("FPS uses wall time even below 20fps and ignores invalid intervals", () => {
  const metric = new FrameRate();
  assert.equal(metric.update(0), null);
  assert.equal(metric.update(NaN), null);
  assert.equal(metric.update(0.25), null);
  assert.equal(metric.update(0.25), 4);
  assert.equal(metric.update(1), 1);
});

test("explicit refinement works at 100M where Auto and both methods otherwise run one pass", () => {
  const e = new Engine({});
  assert.equal(e.samplingMode, "global", "Startup morphing must not claim Focus is running");
  assert.equal(e._accumBatchTarget(), 8);
  e.particlesPerBatch = 100000000;
  for (const mode of ["reuse", "independent"]) {
    e.accumulationMode = mode;
    e.refinementPasses = "auto";
    assert.equal(e._accumBatchTarget(), 1);
    for (const passes of [1, 2, 4, 8]) {
      e.refinementPasses = passes;
      assert.equal(e._accumBatchTarget(), passes);
      assert.equal(e.particlesPerBatch, 100000000, "More passes must not grow the particle buffer");
    }
  }
});

function postHarness() {
  const e = new Engine({});
  e.displayMode = "classic";
  e._front = 0; e._fbW = 640; e._fbH = 480;
  e.pipe = Object.fromEntries(["kuwStructure", "kuwBlurH", "kuwAniso", "kuwFilter", "prefilter", "blur", "present"].map(k => [k, k]));
  e.bgKuwStructure = e.bgKuwFilter = e.bgPreFromScene = e.bgPresentScene = [{}, {}];
  e.ctx = { getCurrentTexture: () => ({ createView: () => ({}) }) };
  e.calls = [];
  e._fullscreen = (_, pipe) => e.calls.push(pipe);
  return e;
}

test("unchanged fronts reuse bloom, while intensity-only edits still composite", () => {
  const e = postHarness();
  e._encodePost({});
  assert.equal(e.calls.length, 8);
  e.calls = []; e._encodePost({});
  assert.deepEqual(e.calls, ["present"]);
  e.bloomIntensity = 0; e._encodePost({});
  e.bloomIntensity = 3; e._encodePost({});
  assert.equal(e.postStats.bloom, 1);
  e.sceneRevision++; e._front = 1; e._encodePost({});
  assert.equal(e.postStats.bloom, 2);
  e.bloomSpread++; e._encodePost({});
  assert.equal(e.postStats.bloom, 3);
});

test("Kuwahara, resize, and disabled-effect edits invalidate only dependent cached results", () => {
  const e = postHarness();
  e._encodePost({});
  e.kuwaharaKernelSize = 8; e._encodePost({});
  assert.equal(e.postStats.bloom, 1);
  e.kuwaharaEnabled = true; e._encodePost({});
  assert.deepEqual(e.postStats, { kuwahara: 1, bloom: 2 });
  e._encodePost({});
  assert.deepEqual(e.postStats, { kuwahara: 1, bloom: 2 });
  e.kuwaharaSharpness++; e._encodePost({});
  assert.deepEqual(e.postStats, { kuwahara: 2, bloom: 3 });
  e._fbW++; e._encodePost({});
  assert.deepEqual(e.postStats, { kuwahara: 3, bloom: 4 });
  e.kuwaharaEnabled = false; e._encodePost({});
  assert.deepEqual(e.postStats, { kuwahara: 3, bloom: 5 });
});

test("frame state resets on geometry changes and retains unique counters across camera redraws", () => {
  const e = new Engine({});
  e.transformData = new Float32Array(512); e.maxComputeDim = 65535;
  e._front = 0; e._fbW = 640; e._fbH = 480;
  e.sceneTexs = [{}, {}]; e.depthTexs = [{}, {}];
  e.device = { queue: { writeBuffer() {}, submit() {} }, createCommandEncoder: () => ({ copyTextureToTexture() {}, finish() {} }) };
  e.profiler = { beginFrame() {}, finishFrame() {}, afterSubmit() {} };
  e.pipe = { combine: {} };
  e._ensureSizes = e.resize = e._writePostUniforms = e._encodePost = e._encodeFit = e._encodeVoxelize = e._encodeRender = e._dispatch = e._encodeIterate = () => {};
  e._writeUniforms = (_, __, mode) => { e.uniformMode = mode; };
  let view = 0;
  e._buildRenderUniform = () => { const u = new ArrayBuffer(224); new Uint32Array(u)[0] = view; return u; };
  const blender = { getTransformCount: () => 3, packMatrices() {} };
  e.frame(blender, {});
  assert.equal(e.frameMode, "compute");
  assert.equal(e.sceneRevision, 1);
  e.frame(blender, {});
  assert.equal(e.frameMode, "accumulate");
  assert.equal(e._batchSeed, 1);
  assert.equal(e.sceneRevision, 1);
  view++; e.frame(blender, {});
  assert.equal(e.frameMode, "redraw");
  e.frame(blender, {});
  assert.equal(e._batchSeed, 2); // does not repeat the retained seed 1
  for (let i = 0; i < 12; i++) e.frame(blender, {});
  assert.equal(e.frameMode, "idle");
  const version = e.sceneRevision;
  e.frame(blender, {});
  assert.equal(e.sceneRevision, version);
  e.lightingParticleBudget = 524288; e.frame(blender, {});
  assert.equal(e.frameMode, "compute");
  assert.equal(e._batchSeed, 0);
  e.accumulationMode = "independent"; e.frame(blender, {});
  assert.equal(e.frameMode, "compute");
  e.transformData[0] = 2; e.frame(blender, {});
  assert.equal(e.frameMode, "compute");
});

test("changing the pass budget at 100M restarts image accumulation and runs the selected method", () => {
  const e = new Engine({});
  e.particlesPerBatch = 100000000; e.maxComputeDim = 65535;
  e.transformData = new Float32Array(512); e.uChaosCPU = new ArrayBuffer(544);
  e._front = 0; e._fbW = 640; e._fbH = 480;
  e.sceneTexs = [{}, {}]; e.depthTexs = [{}, {}];
  e.device = { queue: { writeBuffer() {}, submit() {} }, createCommandEncoder: () => ({ copyTextureToTexture() {}, finish() {} }) };
  e.profiler = { beginFrame() {}, finishFrame() {}, afterSubmit() {} };
  e.pipe = { combine: {} };
  e._ensureSizes = e.resize = e._writePostUniforms = e._encodePost = e._encodeFit = e._encodeVoxelize = e._encodeRender = e._dispatch = () => {};
  const iterations = [];
  e._encodeIterate = () => iterations.push(Array.from(new Uint32Array(e.uChaosCPU, 0, 8)));
  e._buildRenderUniform = () => new ArrayBuffer(224);
  const blender = { getTransformCount: () => 3, packMatrices() {} };
  // Execute the production frame scheduler and uniform writer without GPU
  // allocation. This verifies 100M control semantics, not 100M GPU throughput.
  e.frame(blender, {}); e.frame(blender, {});
  assert.equal(e.frameMode, "idle"); assert.equal(e._accumCount, 1);
  const revision = e._fitRevision;
  e.refinementPasses = 4;
  e.frame(blender, {});
  assert.equal(e.frameMode, "redraw");
  assert.equal(e._fitRevision, revision, "A budget edit must preserve the fit and lighting");
  for (let i = 0; i < 4; i++) e.frame(blender, {});
  assert.equal(e._accumCount, 4); assert.equal(e.frameMode, "idle");
  assert.deepEqual(iterations.slice(1).map(u => [u[1], u[3], u[5]]), Array.from({ length: 3 }, () => [100000000, 4, 1]));
  e.refinementPasses = 1; e.frame(blender, {});
  assert.equal(e._accumCount, 1); assert.equal(e.frameMode, "redraw");
  iterations.length = 0;
  e.accumulationMode = "independent"; e.refinementPasses = 4;
  for (let i = 0; i < 5; i++) e.frame(blender, {});
  assert.equal(e._accumCount, 4); assert.equal(iterations.length, 4);
  assert.ok(iterations.every(u => u[3] > 4 && u[5] === 0));
});

test("chaos and lighting uniforms match packed shader layouts, and only static reuse advances", () => {
  assert.equal(PARTICLE_STRIDE, 12);
  const e = new Engine({}); e.maxComputeDim = 65535;
  e.transformData = new Float32Array(512); e.uChaosCPU = new ArrayBuffer(544);
  e.uChaos = "chaos"; e.uGrid = "grid"; e.uRender = "render";
  const writes = new Map();
  e.device = { queue: { writeBuffer: (b, _, data) => writes.set(b, data.slice ? data.slice(0) : data) } };
  const s = e._schedule(3), rb = new ArrayBuffer(224);
  e.lightingParticleBudget = 524288;
  e._writeUniforms(s, rb, "compute");
  assert.deepEqual([...new Uint32Array(writes.get("chaos"), 12, 3)], [23, 0, 0]);
  assert.deepEqual([...new Uint32Array(writes.get("grid"), 24, 2)], [524288, 16]);
  writes.clear(); e._batchSeed = 1; e._writeUniforms(s, rb, "accumulate");
  assert.equal(writes.size, 1);
  assert.deepEqual([...new Uint32Array(writes.get("chaos"), 12, 3)], [4, 1, 1]);
  e.accumulationMode = "independent"; e._writeUniforms(s, rb, "accumulate");
  assert.deepEqual([...new Uint32Array(writes.get("chaos"), 12, 3)], [23, 1, 0]);
  writes.clear(); e._writeUniforms(s, rb, "idle"); assert.equal(writes.size, 0);
  e._writeUniforms(s, rb, "redraw"); assert.deepEqual([...writes.keys()], ["render"]);
});
