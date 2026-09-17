import assert from "node:assert/strict";
import test from "node:test";
import { GPUProfiler } from "../web/js/gpu-profiler.js";

globalThis.GPUBufferUsage = { QUERY_RESOLVE: 1, COPY_SRC: 2, COPY_DST: 4, MAP_READ: 8 };
globalThis.GPUMapMode = { READ: 1 };

const settle = () => new Promise((resolve) => setImmediate(resolve));

function fakeDevice(supported = true) {
  let deviceLost;
  const device = {
    features: new Set(supported ? ["timestamp-query"] : []),
    querySets: [], buffers: [],
    lost: new Promise((resolve) => { deviceLost = resolve; }),
    createQuerySet(descriptor) {
      const set = { ...descriptor, destroyed: false, destroy() { this.destroyed = true; } };
      this.querySets.push(set);
      return set;
    },
    createBuffer(descriptor) {
      const buffer = {
        ...descriptor, destroyed: false, unmapped: 0, maps: 0,
        data: new ArrayBuffer(descriptor.size),
        mapAsync(...args) {
          this.maps++;
          this.mapArgs = args;
          return new Promise((resolve, reject) => { this.resolve = resolve; this.reject = reject; });
        },
        getMappedRange(offset, size) { return this.data.slice(offset, offset + size); },
        unmap() { this.unmapped++; },
        destroy() { this.destroyed = true; this.reject?.(new Error("destroyed")); },
      };
      this.buffers.push(buffer);
      return buffer;
    },
  };
  device.lose = () => deviceLost({ reason: "unknown" });
  device.readbacks = () => device.buffers.filter((b) => b.usage & GPUBufferUsage.MAP_READ);
  return device;
}

const fakeEncoder = () => ({
  resolved: [], copied: [],
  resolveQuerySet(...args) { this.resolved.push(args); },
  copyBufferToBuffer(...args) { this.copied.push(args); },
});

function record(profiler, labels = ["iterate"]) {
  assert.equal(profiler.beginFrame(), true);
  for (const label of labels) profiler.stamp(label);
  const encoder = fakeEncoder();
  profiler.finishFrame(encoder);
  profiler.afterSubmit();
  return encoder;
}

test("unsupported or disabled profiling allocates no resources and is a no-op", () => {
  for (const [supported, enabled] of [[false, true], [true, false]]) {
    const device = fakeDevice(supported);
    const profiler = new GPUProfiler(device, { enabled });
    assert.equal(profiler.supported, supported);
    assert.equal(profiler.enabled, false);
    assert.equal(profiler.beginFrame(), false);
    assert.equal(profiler.stamp("points"), undefined);
    const encoder = fakeEncoder();
    profiler.finishFrame(encoder);
    profiler.afterSubmit();
    profiler.dispose();
    assert.equal(device.buffers.length, 0);
    assert.equal(device.querySets.length, 0);
    assert.equal(encoder.resolved.length, 0);
    assert.equal(profiler.latest, null);
  }
});

test("resolves only used queries, maps after submit, and aggregates duplicate labels", async () => {
  const device = fakeDevice();
  const profiler = new GPUProfiler(device, { ringSize: 1, sampleEvery: 1, maxPasses: 4 });
  assert.equal(profiler.beginFrame(), true);
  assert.deepEqual(profiler.stamp("iterate"), {
    querySet: device.querySets[0], beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1,
  });
  profiler.stamp("blur");
  profiler.stamp("blur");
  const encoder = fakeEncoder();
  profiler.finishFrame(encoder);
  const buffer = device.readbacks()[0];
  assert.equal(buffer.maps, 0);
  assert.deepEqual(encoder.resolved[0], [device.querySets[0], 0, 6, device.buffers[0], 0]);
  assert.deepEqual(encoder.copied[0], [device.buffers[0], 0, buffer, 0, 48]);
  profiler.afterSubmit();
  profiler.afterSubmit();
  assert.equal(buffer.maps, 1);
  assert.deepEqual(buffer.mapArgs, [GPUMapMode.READ, 0, 48]);
  new BigUint64Array(buffer.data).set([100n, 1000100n, 200n, 2000200n, 300n, 3000300n]);
  buffer.resolve();
  await settle();
  assert.deepEqual(profiler.latest, {
    frame: 0, totalMs: 6, passes: { iterate: 1, blur: 5 }, droppedPasses: 0,
  });
  assert.equal(buffer.unmapped, 1);
  assert.ok(Object.isFrozen(profiler.latest.passes));
  profiler.dispose();
});

test("a busy ring skips samples without allocating or reusing pending buffers", async () => {
  const device = fakeDevice();
  const profiler = new GPUProfiler(device, { sampleEvery: 1, ringSize: 2 });
  record(profiler);
  record(profiler);
  assert.equal(profiler.beginFrame(), false);
  assert.equal(profiler.stamp("unrecorded"), undefined);
  profiler.finishFrame(fakeEncoder());
  assert.equal(profiler.skippedSamples, 1);
  assert.equal(device.buffers.length, 4);
  assert.equal(device.querySets.length, 2);
  const [first, second] = device.readbacks();
  new BigUint64Array(second.data).set([0n, 3000000n]);
  second.resolve();
  await settle();
  assert.equal(profiler.latest.frame, 1);
  new BigUint64Array(first.data).set([0n, 1000000n]);
  first.resolve();
  await settle();
  assert.equal(profiler.latest.frame, 1, "an older asynchronous result must not overwrite a newer result");
  assert.equal(profiler.latest.totalMs, 3);
  assert.equal(profiler.beginFrame(), true);
  profiler.finishFrame(fakeEncoder());
  profiler.dispose();
});

test("sampling interval and query budget are bounded, including frames with no passes", () => {
  const device = fakeDevice();
  const profiler = new GPUProfiler(device, { sampleEvery: 2, ringSize: 1, maxPasses: 1 });
  assert.equal(profiler.beginFrame(), true);
  const empty = fakeEncoder();
  profiler.finishFrame(empty);
  profiler.afterSubmit();
  assert.equal(empty.resolved.length, 0);
  assert.equal(device.readbacks()[0].maps, 0);
  assert.equal(profiler.beginFrame(), false);
  profiler.finishFrame(fakeEncoder());
  assert.equal(profiler.beginFrame(), true);
  assert.ok(profiler.stamp("points"));
  assert.equal(profiler.stamp("extra"), undefined);
  const encoder = fakeEncoder();
  profiler.finishFrame(encoder);
  assert.equal(encoder.resolved[0][2], 2);
  profiler.dispose();
});

test("rejected and synchronously failed mappings release their slot safely", async () => {
  const device = fakeDevice();
  const profiler = new GPUProfiler(device, { sampleEvery: 1, ringSize: 1 });
  record(profiler);
  const buffer = device.readbacks()[0];
  buffer.reject(new Error("map failed"));
  await settle();
  assert.equal(profiler.failedSamples, 1);
  assert.equal(profiler.latest, null);
  const mapAsync = buffer.mapAsync;
  buffer.mapAsync = () => { throw new Error("synchronous failure"); };
  record(profiler);
  await settle();
  assert.equal(profiler.failedSamples, 2);
  buffer.mapAsync = mapAsync;
  record(profiler);
  buffer.resolve();
  await settle();
  assert.equal(profiler.latest.frame, 2);
  profiler.dispose();
});

test("counter reset is ignored and device loss cancels readbacks without publishing", async () => {
  const device = fakeDevice();
  const profiler = new GPUProfiler(device, { sampleEvery: 1, ringSize: 1 });
  record(profiler, ["reset", "valid"]);
  const buffer = device.readbacks()[0];
  new BigUint64Array(buffer.data).set([99n, 1n, 500n, 2000500n]);
  buffer.resolve();
  await settle();
  assert.deepEqual(profiler.latest.passes, { valid: 2 });
  const latest = profiler.latest;
  record(profiler);
  device.lose();
  await settle();
  assert.equal(profiler.enabled, false);
  assert.equal(profiler.beginFrame(), false);
  assert.equal(profiler.latest, latest);
  assert.ok(device.buffers.every((b) => b.destroyed));
  assert.ok(device.querySets.every((q) => q.destroyed));
  assert.equal(profiler.failedSamples, 0);
  profiler.dispose();
});

test("invalid ring sizes, intervals and query budgets fail before allocation", () => {
  for (const options of [{ ringSize: 0 }, { ringSize: 9 }, { sampleEvery: 0 }, { maxPasses: 257 }]) {
    const device = fakeDevice();
    assert.throws(() => new GPUProfiler(device, options), RangeError);
    assert.equal(device.buffers.length, 0);
  }
});
