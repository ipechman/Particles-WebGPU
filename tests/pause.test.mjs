import test from "node:test";
import assert from "node:assert/strict";
import { Blender, MORPH_FUNCTIONS } from "../web/js/blender.js";

for (const [mode] of MORPH_FUNCTIONS) test(`pause freezes the current geometry and transition immediately: ${mode}`, () => {
  const b = new Blender(); b.setMorphMode(mode);
  for (let i = 0; i < 8; i++) b.update(0.05);
  b.animate = false;
  const before = b.packMatrices().slice(), time = [b.t, b.ramp, b.morphProgress];
  for (let i = 0; i < 100; i++) b.update(0.05);
  assert.deepEqual(b.packMatrices(), before);
  assert.deepEqual([b.t, b.ramp, b.morphProgress], time);
  b.animate = true; b.update(0.05);
  assert.notDeepEqual(b.packMatrices(), before);
});

test("Randomize while paused selects a new static shape", () => {
  const b = new Blender(); b.animate = false;
  const before = b.packMatrices().slice(); b.randomizeTarget();
  const after = b.packMatrices().slice();
  assert.notDeepEqual(after, before);
  b.update(1); assert.deepEqual(b.packMatrices(), after);
});
