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
  // The focused renderer now keeps Global's full N * transformCount drawing
  // budget. Selecting Focus must not silently turn it off at ordinary zooms.
  return { active: true, reason: "focused", leaves, visited, visibleMass };
}

function pcg(value) {
  const state = (Math.imul(value, 747796405) + 2891336453) >>> 0;
  const word = Math.imul((state >>> ((state >>> 28) + 4)) ^ state, 277803737) >>> 0;
  return ((word >>> 22) ^ word) >>> 0;
}

// Conservative boxes alone cannot describe overlapping procedural shapes.
// A small, deterministic pilot estimates where their samples actually land.
// This changes allocation only: a branch is never culled by pilot visibility.
export function viewWeights(leaves, { matrices, seeds, viewFit, width, height }) {
  if (!leaves.length) return [];
  const pool = [];
  for (let i = 0; i < 1024; i++) {
    let h = pcg(i + 7919), p = Array.from(seeds[h % matrices.length]);
    for (let k = 0; k < 32; k++) {
      h = pcg(h);
      const m = matrices[h % matrices.length];
      p = [0, 1, 2].map(r => point(m, p, r));
    }
    pool.push(p);
  }
  const nx = Math.min(128, Math.ceil(width / 8));
  const ny = Math.min(128, Math.ceil(height / 8));
  const density = new Float64Array(nx * ny);
  const nearest = new Float64Array(nx * ny).fill(Infinity);
  const probes = leaves.map((leaf, leafIndex) => {
    const matrix = multiply64(viewFit, leaf.matrix), visible = [];
    for (let j = 0; j < 32; j++) {
      const p = pool[(leafIndex * 37 + j * 17) % pool.length];
      const x = point(matrix, p, 0), y = point(matrix, p, 1);
      const z = point(matrix, p, 2), w = point(matrix, p, 3);
      if (w <= 0 || x < -w || x >= w || y <= -w || y > w || z < 0 || z >= w) continue;
      const tile = Math.floor((0.5 - y / w * 0.5) * ny) * nx + Math.floor((x / w * 0.5 + 0.5) * nx);
      density[tile] += leaf.mass / 32;
      nearest[tile] = Math.min(nearest[tile], w);
      visible.push({ tile, distance: w });
    }
    return visible;
  });
  const scores = probes.map((samples, i) => samples.reduce((sum, p) => {
    const depthDifference = (p.distance - nearest[p.tile]) / Math.max(nearest[p.tile] * 0.025, 1e-6);
    const visibility = 0.2 + 0.8 / (1 + depthDifference);
    return sum + leaves[i].mass / 32 * visibility / Math.sqrt(density[p.tile]);
  }, 0));
  const total = scores.reduce((sum, v) => sum + v, 0);
  const mass = leaves.reduce((sum, l) => sum + l.mass, 0);
  return leaves.map((l, i) => total > 0 ? 0.15 * l.mass / mass + 0.85 * scores[i] / total : l.mass / mass);
}

// Split the full drawing budget at particle-buffer boundaries. Each draw
// references valid base-particle indices, even at N=100M and 32 transforms.
// At most leafCount + transformCount - 1 draws are needed.
export function packViewDraws(leaves, particles, transformCount, fit, weights) {
  const total = particles * transformCount;
  const draws = [];
  if (!leaves.length) return { bytes: new ArrayBuffer(VIEW_LEAF_BYTES), draws, vertices: 0 };
  const weightSum = weights.reduce((sum, v) => sum + v, 0);
  let cumulative = 0, start = 0;
  leaves.forEach((leaf, i) => {
    cumulative += weights[i];
    const end = i === leaves.length - 1 ? total
      : i + 1 + Math.floor((total - leaves.length) * cumulative / weightSum);
    const matrix = multiply64(fit, leaf.matrix);
    while (start < end) {
      const copy = Math.floor(start / particles);
      const next = Math.min(end, (copy + 1) * particles);
      draws.push({ first: start % particles, count: next - start, copy, matrix });
      start = next;
    }
  });
  const bytes = new ArrayBuffer(draws.length * VIEW_LEAF_BYTES);
  draws.forEach((draw, i) => {
    new Float32Array(bytes, i * VIEW_LEAF_BYTES, 16).set(draw.matrix);
    new Uint32Array(bytes, i * VIEW_LEAF_BYTES + 64, 4).set([draw.copy, 0, 0, 0]);
  });
  return { bytes, draws, vertices: total };
}
