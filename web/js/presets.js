// presets.js
// The "generating functions": affine-transform instruction sets for each
// fractal preset, plus the procedural random generator.
// Ported from AttractorPresets.cs, TransformSet.cs and ProceduralWizard.cs.

import { quat } from "./math.js";

// A TransformInstructions object. Rotation is stored as a quaternion
// internally (presets specify euler degrees, converted on construction);
// everything else matches TransformSet.TransformInstructions.
export function makeInstr({
  scale = [1, 1, 1],
  shearX = [0, 0, 0],
  shearY = [0, 0, 0],
  shearZ = [0, 0, 0],
  rot = quat.identity(),
  translate = [0, 0, 0],
} = {}) {
  return { scale, shearX, shearY, shearZ, rot, translate };
}

export function identityInstr() {
  return makeInstr({ scale: [1, 1, 1] });
}

// Operator+ from TransformSet: scales multiply, shears/translate add,
// rotations compose (q1 * q2).
export function addInstr(a, b) {
  return {
    scale: [a.scale[0] * b.scale[0], a.scale[1] * b.scale[1], a.scale[2] * b.scale[2]],
    shearX: [a.shearX[0] + b.shearX[0], a.shearX[1] + b.shearX[1], a.shearX[2] + b.shearX[2]],
    shearY: [a.shearY[0] + b.shearY[0], a.shearY[1] + b.shearY[1], a.shearY[2] + b.shearY[2]],
    shearZ: [a.shearZ[0] + b.shearZ[0], a.shearZ[1] + b.shearZ[1], a.shearZ[2] + b.shearZ[2]],
    rot: quat.mul(a.rot, b.rot),
    translate: [a.translate[0] + b.translate[0], a.translate[1] + b.translate[1], a.translate[2] + b.translate[2]],
  };
}

const fromTranslations = (scale, translations) =>
  translations.map((t) => makeInstr({ scale, translate: t }));

// ---------------------------------------------------------------------------
// The six deterministic presets.
// ---------------------------------------------------------------------------
export function SierpinskiTriangle2D() {
  return fromTranslations([0.5, 0.5, 0.5], [
    [-0.5, -0.5, 0.0],
    [0.0, 0.36, 0.0],
    [0.5, -0.5, 0.0],
  ]);
}

export function Vicsek2D() {
  return fromTranslations([0.33, 0.33, 0.33], [
    [-0.5, -0.5, 0.0],
    [-0.5, 0.5, 0.0],
    [0.5, 0.5, 0.0],
    [0.5, -0.5, 0.0],
    [0.0, 0.0, 0.0],
  ]);
}

export function SierpinskiCarpet2D() {
  return fromTranslations([0.33, 0.33, 0.33], [
    [-0.5, -0.5, 0.0],
    [-0.5, 0.5, 0.0],
    [0.5, 0.5, 0.0],
    [0.5, -0.5, 0.0],
    [-0.5, 0.0, 0.0],
    [0.5, 0.0, 0.0],
    [0.0, 0.5, 0.0],
    [0.0, -0.5, 0.0],
  ]);
}

export function SierpinskiTriangle3D() {
  return fromTranslations([0.5, 0.5, 0.5], [
    [-0.5, -0.5, 0.5],
    [-0.5, -0.5, -0.5],
    [0.5, -0.5, 0.5],
    [0.5, -0.5, -0.5],
    [0.0, 0.36, 0.0],
  ]);
}

export function Vicsek3D() {
  return fromTranslations([0.33, 0.33, 0.33], [
    [-0.5, -0.5, -0.5],
    [-0.5, -0.5, 0.5],
    [0.5, -0.5, -0.5],
    [0.5, -0.5, 0.5],
    [-0.5, 0.5, -0.5],
    [-0.5, 0.5, 0.5],
    [0.5, 0.5, -0.5],
    [0.5, 0.5, 0.5],
    [0.0, 0.0, 0.0],
  ]);
}

export function SierpinskiCarpet3D() {
  return fromTranslations([0.33, 0.33, 0.33], [
    [-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5], [0.5, -0.5, -0.5], [0.5, -0.5, 0.5],
    [-0.5, 0.5, -0.5], [-0.5, 0.5, 0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5],
    [-0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, -0.5, 0.0], [0.5, -0.5, 0.0],
    [0.0, 0.5, -0.5], [0.0, 0.5, 0.5], [0.0, -0.5, -0.5], [0.0, -0.5, 0.5],
    [-0.5, 0.0, -0.5], [0.5, 0.0, 0.5], [0.5, 0.0, -0.5], [-0.5, 0.0, 0.5],
  ]);
}

// ---------------------------------------------------------------------------
// Procedural generator (ProceduralWizard.cs). Default ranges taken from the
// original SampleScene.
// ---------------------------------------------------------------------------
export const PROCEDURAL_RANGES = {
  scaleMin: [0.85, 0.85, 0.85], scaleMax: [0.75, 0.75, 0.75],
  shearXMin: [0, -0.1, -0.1], shearXMax: [0, 0.1, 0.1],
  shearYMin: [-0.1, 0, -0.1], shearYMax: [0.1, 0, 0.1],
  shearZMin: [-0.1, -0.1, 0], shearZMax: [0.1, 0.1, 0],
  rotateMin: [-40, -40, -40], rotateMax: [40, 40, 40],
  translateMin: [0, 0, 0], translateMax: [1, 1, 1],
};

const rand = (min, max) => min + (max - min) * Math.random();
const randVec = (min, max) => [rand(min[0], max[0]), rand(min[1], max[1]), rand(min[2], max[2])];

export function generateRandomInstr(r = PROCEDURAL_RANGES) {
  return makeInstr({
    scale: randVec(r.scaleMin, r.scaleMax),
    shearX: randVec(r.shearXMin, r.shearXMax),
    shearY: randVec(r.shearYMin, r.shearYMax),
    shearZ: randVec(r.shearZMin, r.shearZMax),
    rot: quat.fromEuler(randVec(r.rotateMin, r.rotateMax)),
    translate: randVec(r.translateMin, r.translateMax),
  });
}

export function Procedural(count = 3) {
  const out = [];
  for (let i = 0; i < count; i++) out.push(generateRandomInstr());
  return out;
}

export const PRESETS = {
  SierpinskiTriangle2D,
  Vicsek2D,
  SierpinskiCarpet2D,
  SierpinskiTriangle3D,
  Vicsek3D,
  SierpinskiCarpet3D,
  Procedural,
};

// Display labels for the UI dropdown.
export const PRESET_LABELS = [
  ["SierpinskiTriangle2D", "Sierpinski Triangle (2D)"],
  ["Vicsek2D", "Vicsek (2D)"],
  ["SierpinskiCarpet2D", "Sierpinski Carpet (2D)"],
  ["SierpinskiTriangle3D", "Sierpinski Triangle (3D)"],
  ["Vicsek3D", "Vicsek (3D)"],
  ["SierpinskiCarpet3D", "Sierpinski Carpet (3D)"],
  ["Procedural", "Procedural (random)"],
];

export function buildPreset(name, proceduralCount = 3) {
  if (name === "Procedural") return Procedural(proceduralCount);
  return PRESETS[name]();
}
