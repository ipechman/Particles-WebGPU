// blender.js
// The morphing / smoothing engine. Ports SetBlender.cs (the lerp-smoothing
// "MoveToward" path and the animation-curve "BlendSets" path) together with
// AffineTransformations.AffineFromInstructions.

import { mat4, quat, v3 } from "./math.js";
import { AnimationCurve } from "./animcurve.js";
import { buildPreset, identityInstr, addInstr } from "./presets.js";

// Build the final affine matrix for one instruction set entry.
//   affine = Scale * Rotation * Shear * Translate   (Shear = ShearZ*ShearY*ShearX)
export function affineFromInstr(t) {
  const scale = mat4.scale(t.scale);
  const rotation = mat4.fromQuat(t.rot);
  const shearX = mat4.shearX(t.shearX);
  const shearY = mat4.shearY(t.shearY);
  const shearZ = mat4.shearZ(t.shearZ);
  const shear = mat4.multiply(mat4.multiply(shearZ, shearY), shearX);
  const translate = mat4.translate(t.translate);
  return mat4.multiply(mat4.multiply(scale, rotation), mat4.multiply(shear, translate));
}

// InterpolateInstructions: lerp scalars, slerp rotation.
function interpolate(a, b, t) {
  return {
    scale: v3.lerpUnclamped(a.scale, b.scale, t),
    shearX: v3.lerpUnclamped(a.shearX, b.shearX, t),
    shearY: v3.lerpUnclamped(a.shearY, b.shearY, t),
    shearZ: v3.lerpUnclamped(a.shearZ, b.shearZ, t),
    translate: v3.lerpUnclamped(a.translate, b.translate, t),
    rot: quat.slerp(a.rot, b.rot, t),
  };
}

// ---- helpers for the alternative morph functions -------------------------
const smooth01 = (p) => { const x = Math.min(1, Math.max(0, p)); return x * x * (3 - 2 * x); };

const moveTowards1 = (a, b, maxDelta) => {
  const d = b - a;
  if (Math.abs(d) <= maxDelta) return b;
  return a + Math.sign(d) * maxDelta;
};
const moveTowardsVec = (a, b, md) => [
  moveTowards1(a[0], b[0], md), moveTowards1(a[1], b[1], md), moveTowards1(a[2], b[2], md),
];

// Angle-limited slerp (Quaternion.RotateTowards).
function rotateTowards(from, to, maxRad) {
  let d = from[0] * to[0] + from[1] * to[1] + from[2] * to[2] + from[3] * to[3];
  const angle = 2 * Math.acos(Math.min(1, Math.abs(d)));
  if (angle < 1e-5) return to.slice();
  return quat.slerp(from, to, Math.min(1, maxRad / angle));
}

const zeroVelInstr = () => ({
  scale: [0, 0, 0], shearX: [0, 0, 0], shearY: [0, 0, 0], shearZ: [0, 0, 0], translate: [0, 0, 0],
});

// The selectable morph functions (value -> label). The first is the default.
export const MORPH_FUNCTIONS = [
  ["lerpSmoothing", "Lerp Smoothing"],
  ["linear", "Linear"],
  ["smoothstep", "Smoothstep"],
  ["spring", "Spring"],
];

const MAX_TRANSFORMS = 32; // matches the original attractor buffer capacity

export class Blender {
  constructor() {
    this.preset = "Procedural";
    this.proceduralCount = 3;

    this.set1 = buildPreset(this.preset, this.proceduralCount); // current source
    this.set2 = buildPreset(this.preset, this.proceduralCount); // target

    this.moveTowardSet = this.set1.map((x) => ({ ...x }));
    this.blendedSet = this.moveTowardSet;

    this.curve = new AnimationCurve();

    // Tunables (defaults from SampleScene).
    this.animate = true;
    this.speed = 1.6;
    this.useRamp = true;
    this.rampSpeed = 1.0;
    this.epsilon = 3.0;

    this.t = 0;
    this.ramp = 0;

    // Morph function selection + state for the alternative functions.
    this.morphMode = "lerpSmoothing"; // default = the original lerp smoothing
    this.snapshotSet = this.moveTowardSet.map((x) => ({ ...x })); // smoothstep source
    this.velSet = this.moveTowardSet.map(() => zeroVelInstr());    // spring velocities
    this.morphProgress = 1; // smoothstep progress 0..1 (1 = settled)
  }

  // Reset the per-function morph state to start a fresh transition from the
  // currently displayed set.
  _resetMorphState() {
    this.snapshotSet = this.moveTowardSet.map((x) => ({ ...x }));
    this.velSet = this.moveTowardSet.map(() => zeroVelInstr());
    this.morphProgress = 0;
  }

  setMorphMode(mode) {
    this.morphMode = mode;
    this._resetMorphState();
  }

  getTransformCount() {
    return Math.min(this.blendedSet.length, MAX_TRANSFORMS);
  }

  // Regenerate the target set (equivalent to pressing 'f' / set2.ApplyPreset()).
  randomizeTarget() {
    this.set2 = buildPreset(this.preset, this.proceduralCount);
    this.ramp = 0;
    // Start a fresh eased transition from the current form (spring keeps its
    // momentum, so its velocity is intentionally left untouched).
    this.snapshotSet = this.moveTowardSet.map((x) => ({ ...x }));
    this.morphProgress = 0;
  }

  // Switch preset: reset both source and target so we snap to the new shape.
  setPreset(name, proceduralCount = this.proceduralCount) {
    this.preset = name;
    this.proceduralCount = proceduralCount;
    this.set1 = buildPreset(name, proceduralCount);
    this.set2 = buildPreset(name, proceduralCount);
    this.moveTowardSet = this.set1.map((x) => ({ ...x }));
    this.blendedSet = this.moveTowardSet;
    this.t = 0;
    this.ramp = 0;
    this._resetMorphState();
  }

  // MoveTowardInstructions: frame-rate independent exponential decay
  // ("lerp smoothing"), optionally modulated by the ramp animation curve.
  _moveTowardInstr(cur, target, dt) {
    let decay = this.speed * dt; // frameRateIndependent = true
    if (this.useRamp) decay *= this.curve.evaluate(this.ramp);
    decay = Math.min(decay, 1.0);

    return {
      scale: v3.lerpUnclamped(cur.scale, target.scale, decay),
      shearX: v3.lerpUnclamped(cur.shearX, target.shearX, decay),
      shearY: v3.lerpUnclamped(cur.shearY, target.shearY, decay),
      shearZ: v3.lerpUnclamped(cur.shearZ, target.shearZ, decay),
      translate: v3.lerpUnclamped(cur.translate, target.translate, decay),
      rot: quat.slerp(cur.rot, target.rot, Math.min(decay, 1.0)),
    };
  }

  // Linear: constant-rate move toward the target (Vector3.MoveTowards style).
  _linearInstr(cur, target, dt) {
    const md = this.speed * dt;
    return {
      scale: moveTowardsVec(cur.scale, target.scale, md),
      shearX: moveTowardsVec(cur.shearX, target.shearX, md),
      shearY: moveTowardsVec(cur.shearY, target.shearY, md),
      shearZ: moveTowardsVec(cur.shearZ, target.shearZ, md),
      translate: moveTowardsVec(cur.translate, target.translate, md),
      rot: rotateTowards(cur.rot, target.rot, md),
    };
  }

  // Smoothstep: ease in/out from the snapshot to the target over one cycle.
  _easedInstr(snap, target) {
    return interpolate(snap, target, smooth01(this.morphProgress));
  }

  // Spring: damped harmonic oscillator per channel (gives a little overshoot).
  _springInstr(cur, target, vel, dt) {
    const omega = this.speed * 2.0;
    const zeta = 0.35;
    const ch = (x, xt, v) => {
      const out = [0, 0, 0];
      for (let k = 0; k < 3; k++) {
        const a = omega * omega * (xt[k] - x[k]) - 2 * zeta * omega * v[k];
        v[k] += a * dt;
        out[k] = x[k] + v[k] * dt;
      }
      return out;
    };
    return {
      scale: ch(cur.scale, target.scale, vel.scale),
      shearX: ch(cur.shearX, target.shearX, vel.shearX),
      shearY: ch(cur.shearY, target.shearY, vel.shearY),
      shearZ: ch(cur.shearZ, target.shearZ, vel.shearZ),
      translate: ch(cur.translate, target.translate, vel.translate),
      rot: quat.slerp(cur.rot, target.rot, Math.min(this.speed * dt, 1.0)),
    };
  }

  _morphInstr(i, dt) {
    const cur = this.moveTowardSet[i];
    const target = this.set2[i];
    switch (this.morphMode) {
      case "linear": return this._linearInstr(cur, target, dt);
      case "smoothstep": return this._easedInstr(this.snapshotSet[i], target);
      case "spring": return this._springInstr(cur, target, this.velSet[i], dt);
      default: return this._moveTowardInstr(cur, target, dt); // lerpSmoothing
    }
  }

  // Advance the simulation by dt seconds and return the freshly blended set.
  update(dt) {
    // Ramp timer for lerp smoothing + smoothstep transition progress.
    this.ramp += dt * this.rampSpeed;
    this.morphProgress = Math.min(1, this.morphProgress + dt * this.speed);

    if (this.animate) {
      this.t += dt * this.speed;
      if (this.t >= this.epsilon) {
        this.t = 0;
        this.ramp = 0;
        this.randomizeTarget();
      }
    }

    // Keep the per-function state sized to the source set.
    if (this.moveTowardSet.length !== this.set1.length) {
      this.moveTowardSet = this.set1.map((x) => ({ ...x }));
      this.snapshotSet = this.moveTowardSet.map((x) => ({ ...x }));
      this.velSet = this.moveTowardSet.map(() => zeroVelInstr());
    }

    const out = [];
    const n = Math.min(this.moveTowardSet.length, this.set2.length);
    for (let i = 0; i < n; i++) {
      this.moveTowardSet[i] = this._morphInstr(i, dt);
      out.push(this.moveTowardSet[i]);
    }
    this.blendedSet = out;
    return out;
  }

  // Pack the current blended set into a column-major Float32Array of
  // MAX_TRANSFORMS mat4x4 (the GPU attractor buffer).
  packMatrices(dst = new Float32Array(MAX_TRANSFORMS * 16)) {
    const set = this.blendedSet;
    for (let i = 0; i < set.length && i < MAX_TRANSFORMS; i++) {
      dst.set(affineFromInstr(set[i]), i * 16);
    }
    return dst;
  }
}

export { MAX_TRANSFORMS };
