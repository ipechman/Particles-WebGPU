import assert from "node:assert/strict";
import test from "node:test";
import { Engine } from "../web/js/engine.js";
import { THEMES, MAX_PALETTE_STOPS, applyTheme, findTheme } from "../web/js/themes.js";

const references = ["gilded-lagoon", "amber-fern", "glacial-ember"];
const makeEngine = () => new Engine({ width: 640, height: 480 });
const camera = { viewProj: () => new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
  clipPlanes: () => ({ near: 0.01, far: 100 }), fov: Math.PI / 3 };
const uniform = (engine) => engine._buildRenderUniform({ count: 3 }, camera);

test("reference ramps fit the GPU layout and cover the full lighting range", () => {
  assert.equal(new Set(THEMES.map((t) => t.id)).size, THEMES.length);
  for (const id of references) {
    const t = findTheme(id);
    assert.ok(t.paletteStops.length >= 3 && t.paletteStops.length <= MAX_PALETTE_STOPS);
    assert.equal(t.paletteStops[0].position, 0);
    assert.equal(t.paletteStops.at(-1).position, 1);
    assert.deepEqual(t.bg, [0, 0, 0]);
    for (const [i, stop] of t.paletteStops.entries()) {
      assert.equal(stop.color.length, 3);
      assert.ok(stop.color.every((c) => Number.isFinite(c) && c >= 0 && c <= 1));
      if (i) assert.ok(stop.position > t.paletteStops[i - 1].position);
    }
  }
});

test("custom edits are isolated and reselecting a preset restores every color", () => {
  const engine = makeEngine();
  for (const id of references) {
    const original = structuredClone(findTheme(id));
    applyTheme(engine, id);
    engine.particleColor[0] = 0.123;
    engine.occlusionColor[1] = 0.456;
    engine.backgroundColor[2] = 0.789;
    engine.paletteStops[2].color[0] = 0.987;
    engine.paletteStops[2].position = 0.333;
    assert.deepEqual(findTheme(id), original);
    applyTheme(engine, id);
    assert.deepEqual(engine.particleColor, original.particle);
    assert.deepEqual(engine.occlusionColor, original.shadow);
    assert.deepEqual(engine.backgroundColor, original.bg);
    assert.deepEqual(engine.paletteStops, original.paletteStops);
  }
});

test("render uniform packs AO stops at WGSL offsets and honors edited endpoints", () => {
  const engine = makeEngine();
  applyTheme(engine, "gilded-lagoon");
  engine.occlusionColor = [0.125, 0.25, 0.5];
  engine.particleColor = [0.75, 0.875, 1];
  const rb = uniform(engine);
  const data = new DataView(rb);
  assert.equal(rb.byteLength, 224);
  assert.equal(data.getUint32(96, true), engine.voxelGridDim);
  assert.equal(data.getUint32(100, true), 3);
  assert.equal(data.getFloat32(104, true), 2 * engine.voxelBounds * engine.scalePadding);
  assert.equal(data.getUint32(116, true), engine.paletteStops.length);
  for (const [i, stop] of engine.paletteStops.entries()) {
    const expected = i === 0 ? engine.occlusionColor
      : i === engine.paletteStops.length - 1 ? engine.particleColor : stop.color;
    for (let channel = 0; channel < 3; channel++) {
      assert.equal(data.getFloat32(128 + i * 16 + channel * 4, true), Math.fround(expected[channel]));
    }
    assert.equal(data.getFloat32(128 + i * 16 + 12, true), Math.fround(stop.position));
  }
});

test("editing an accent or stop position invalidates a paused accumulated frame", () => {
  const engine = makeEngine();
  applyTheme(engine, "amber-fern");
  assert.equal(engine._renderStale(uniform(engine)), true);
  assert.equal(engine._renderStale(uniform(engine)), false);
  engine.paletteStops[2].color[0] = 0.5;
  assert.equal(engine._renderStale(uniform(engine)), true);
  assert.equal(engine._renderStale(uniform(engine)), false);
  engine.paletteStops[2].position += 0.01;
  assert.equal(engine._renderStale(uniform(engine)), true);
  engine.particleColor = [1, 0, 1];
  assert.equal(engine._renderStale(uniform(engine)), true);
});

test("switching back to any legacy theme clears the ramp and redraws", () => {
  const engine = makeEngine();
  for (const legacy of THEMES.filter((t) => !t.paletteStops)) {
    applyTheme(engine, "glacial-ember");
    engine._renderStale(uniform(engine));
    applyTheme(engine, legacy.id);
    const rb = uniform(engine);
    assert.deepEqual(engine.paletteStops, []);
    assert.deepEqual(engine.particleColor, legacy.particle);
    assert.deepEqual(engine.occlusionColor, legacy.shadow);
    assert.equal(new DataView(rb).getUint32(116, true), 0);
    assert.ok(new Uint8Array(rb, 128).every((b) => b === 0));
    assert.equal(engine._renderStale(rb), true);
    assert.equal(engine._renderStale(uniform(engine)), false);
  }
});
