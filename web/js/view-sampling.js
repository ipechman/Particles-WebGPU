// A bounded, prefix-free IFS frontier. Cull entire addresses before generating
// their particles. All bounds enclose the attractor, never just a sample cloud.
export const MAX_VIEW_LEAVES = 1024;
export const VIEW_LEAF_BYTES = 80;

export function multiply64(a, b) {
  const out = new Float64Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    for (let k = 0; k < 4; k++) out[c * 4 + r] += a[k * 4 + r] * b[c * 4 + k];
  }
  return out;
}

const identity = () => new Float64Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
const point = (m, p, r) => m[r] * p[0] + m[4 + r] * p[1] + m[8 + r] * p[2] + m[12 + r];

function boxImage(m, box) {
  const lo = [], hi = [];
  for (let r = 0; r < 3; r++) {
    lo[r] = hi[r] = m[12 + r];
    for (let c = 0; c < 3; c++) {
      const a = m[c * 4 + r] * box.lo[c], b = m[c * 4 + r] * box.hi[c];
      lo[r] += Math.min(a, b); hi[r] += Math.max(a, b);
    }
  }
  return { lo, hi };
}

export function attractorBounds(matrices, seeds) {
  if (!matrices.length || matrices.some(m => !Array.from(m).every(Number.isFinite))) return null;
  const center = [0, 1, 2].map(r => seeds.reduce((sum, p) => sum + p[r], 0) / seeds.length);
  if (!center.every(Number.isFinite)) return null;
  let radius = 0;
  for (const m of matrices) {
    // sqrt(||A^T A||_infinity) >= spectral norm(A), including rotated maps.
    let normSquared = 0;
    for (let r = 0; r < 3; r++) {
      let row = 0;
      for (let c = 0; c < 3; c++) {
        let dot = 0;
        for (let k = 0; k < 3; k++) dot += m[r * 4 + k] * m[c * 4 + k];
        row += Math.abs(dot);
      }
      normSquared = Math.max(normSquared, row);
    }
    const q = Math.sqrt(normSquared) * (1 + 1e-12);
    if (q >= 0.999 || m[3] || m[7] || m[11] || m[15] !== 1) return null;
    const displacement = Math.hypot(...center.map((v, r) => point(m, center, r) - v));
    radius = Math.max(radius, displacement / (1 - q));
  }
  // Inflate for CPU composition and f32 shader rounding. Keep a floor for 2D
  // shapes and degenerate fixed-point attractors.
  const epsilon = Math.max(1, radius, ...center.map(Math.abs)) * 2e-6;
  radius += epsilon;
  let box = { lo: center.map(v => v - radius), hi: center.map(v => v + radius) };
  // Both B and hull(F_i(B)) enclose the attractor, so their intersection does.
  // This tightens the initial sphere's box, especially for planar presets.
  for (let iteration = 0; iteration < 32; iteration++) {
    const images = matrices.map(m => boxImage(m, box));
    box = {
      lo: box.lo.map((v, r) => Math.max(v, Math.min(...images.map(b => b.lo[r])) - epsilon)),
      hi: box.hi.map((v, r) => Math.min(v, Math.max(...images.map(b => b.hi[r])) + epsilon)),
    };
  }
  return box;
}

// Clip in homogeneous coordinates. In particular, a box crossing the near
// plane must not be rejected by dividing corners with negative w.
export function projectedBox(matrix, box, width, height) {
  const corners = [];
  for (let bits = 0; bits < 8; bits++) {
    const p = [0, 1, 2].map(r => bits & (1 << r) ? box.hi[r] : box.lo[r]);
    corners.push([0, 1, 2, 3].map(r => point(matrix, p, r)));
  }
  if (corners.some(p => !p.every(Number.isFinite))) return { area: width * height };
  const planes = [p => p[0] + p[3], p => p[3] - p[0], p => p[1] + p[3],
    p => p[3] - p[1], p => p[2], p => p[3] - p[2]];
  const margin = 2e-5 * Math.max(1, ...corners.flat().map(Math.abs));
  if (planes.some(plane => corners.every(p => plane(p) < -margin))) return null;
  if (corners.some(p => p[3] <= margin || p[2] <= margin)) return { area: width * height };
  const xs = corners.map(p => p[0] / p[3]), ys = corners.map(p => p[1] / p[3]);
  const dx = Math.min(1, Math.max(...xs)) - Math.max(-1, Math.min(...xs));
  const dy = Math.min(1, Math.max(...ys)) - Math.max(-1, Math.min(...ys));
  return { area: Math.max(1, (dx * width / 2 + 2) * (dy * height / 2 + 2)) };
}

class MaxHeap {
  constructor() { this.items = []; }
  push(node) {
    const a = this.items;
    let i = a.length; a.push(node);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (a[p].area >= node.area) break;
      a[i] = a[p]; i = p;
    }
    a[i] = node;
  }
  pop() {
    const a = this.items, top = a[0], last = a.pop();
    if (a.length) {
      let i = 0;
      while (i * 2 + 1 < a.length) {
        let c = i * 2 + 1;
        if (c + 1 < a.length && a[c + 1].area > a[c].area) c++;
        if (last.area >= a[c].area) break;
        a[i] = a[c]; i = c;
      }
      a[i] = last;
    }
    return top;
  }
}

export function buildViewPlan({ matrices, bounds, viewFit, width, height, particles,
  maxLeaves = MAX_VIEW_LEAVES, maxNodes = 8192, maxDepth = 24, tileSize = 24 }) {
  if (!bounds) return { active: false, reason: "uncertified bounds", leaves: [], visited: 0 };
  const leafLimit = Math.min(maxLeaves, Math.ceil(particles / 64));
  const heap = new MaxHeap(), done = [];
  const projection = projectedBox(viewFit, bounds, width, height);
  if (projection) heap.push({ matrix: identity(), depth: 0, mass: 1, area: projection.area });
  let visited = 1;
  while (heap.items.length && visited + matrices.length <= maxNodes) {
    const node = heap.pop();
    if (node.area <= tileSize * tileSize || node.depth >= maxDepth ||
        heap.items.length + done.length + matrices.length > leafLimit) {
      done.push(node);
      continue;
    }
    for (const m of matrices) {
      // F_prefix(F_child(y)): appending a child never keeps its parent too.
      const matrix = multiply64(node.matrix, m);
      const visible = projectedBox(multiply64(viewFit, matrix), bounds, width, height);
      visited++;
      if (visible) heap.push({ matrix, depth: node.depth + 1,
        mass: node.mass / matrices.length, area: visible.area });
    }
  }
  const leaves = done.concat(heap.items);
  const visibleMass = leaves.reduce((sum, leaf) => sum + leaf.mass, 0);
  // The global renderer submits N * transformCount points. Switch only when
  // culling can recover that multiplicity and give a useful sampling gain.
  const active = visibleMass * matrices.length < 0.7;
  return { active, reason: active ? "focused" : "overview", leaves, visited, visibleMass };
}

export function packViewLeaves(leaves, particles) {
  const bytes = new ArrayBuffer(Math.max(1, leaves.length) * VIEW_LEAF_BYTES);
  const groups = Math.ceil(particles / 64);
  if (!leaves.length) return bytes;
  // One workgroup minimum per leaf preserves thin branches; the remainder is
  // distributed by projected footprint, not the original branch probability.
  const weights = leaves.map(leaf => Math.sqrt(leaf.area));
  const total = weights.reduce((sum, v) => sum + v, 0);
  const spare = groups - leaves.length;
  let cumulative = 0, end = 0;
  leaves.forEach((leaf, i) => {
    new Float32Array(bytes, i * VIEW_LEAF_BYTES, 16).set(leaf.matrix);
    cumulative += weights[i];
    end = i === leaves.length - 1 ? groups : i + 1 + Math.floor(spare * cumulative / total);
    new Uint32Array(bytes, i * VIEW_LEAF_BYTES + 64, 4).set([end, 0, 0, 0]);
  });
  return bytes;
}
