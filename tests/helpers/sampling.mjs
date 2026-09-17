// CPU reference for sampling-quality experiments. This intentionally does not
// import the GPU implementation. Float32 point stores model WGSL storage;
// JS expression evaluation is not a promise of bit-identical GPU arithmetic.
import { affineFromInstr } from "../../web/js/blender.js";
import { makeInstr, SierpinskiTriangle2D, SierpinskiCarpet2D, SierpinskiTriangle3D,
  Vicsek3D } from "../../web/js/presets.js";
import { quat } from "../../web/js/math.js";
import { pathToFileURL } from "node:url";

export function pcg(value) {
  const state = (Math.imul(value, 747796405) + 2891336453) >>> 0;
  const word = Math.imul((state >>> ((state >>> 28) + 4)) ^ state, 277803737) >>> 0;
  return ((word >>> 22) ^ word) >>> 0;
}

export function sampleCounter(index, batch, count) {
  return (index + Math.imul(batch, count)) >>> 0;
}

// Gaussian elimination independently checks the closed-form fixed-point
// calculation used by the application.
export function fixedPoint(matrix) {
  const rows = [0, 1, 2].map((r) => [0, 1, 2].map((c) =>
    (r === c ? 1 : 0) - matrix[c * 4 + r]).concat(matrix[12 + r]));
  for (let c = 0; c < 3; c++) {
    let pivot = c;
    for (let r = c + 1; r < 3; r++) if (Math.abs(rows[r][c]) > Math.abs(rows[pivot][c])) pivot = r;
    [rows[c], rows[pivot]] = [rows[pivot], rows[c]];
    const divisor = rows[c][c];
    if (Math.abs(divisor) < 1e-12) throw new Error("Singular test transform");
    for (let j = c; j <= 3; j++) rows[c][j] /= divisor;
    for (let r = 0; r < 3; r++) if (r !== c) {
      const factor = rows[r][c];
      for (let j = c; j <= 3; j++) rows[r][j] -= factor * rows[c][j];
    }
  }
  return rows.map((row) => Math.fround(row[3]));
}

export function fullDepth(count, particles) {
  return count < 2 ? 4 : Math.min(40, Math.max(12, Math.ceil(Math.log(particles) / Math.log(count)) + 8));
}

export function batchPoints(matrices, particles, batch, hops, previous = null) {
  const seeds = matrices.map(fixedPoint);
  const positions = new Float32Array(particles * 3);
  for (let i = 0; i < particles; i++) {
    let h = pcg(sampleCounter(i, batch, particles));
    const seed = seeds[h % matrices.length];
    let x = previous ? previous[i * 3] : seed[0];
    let y = previous ? previous[i * 3 + 1] : seed[1];
    let z = previous ? previous[i * 3 + 2] : seed[2];
    for (let k = 0; k < hops; k++) {
      h = pcg(h);
      const m = matrices[h % matrices.length];
      const nx = Math.fround(m[0] * x + m[4] * y + m[8] * z + m[12]);
      const ny = Math.fround(m[1] * x + m[5] * y + m[9] * z + m[13]);
      z = Math.fround(m[2] * x + m[6] * y + m[10] * z + m[14]);
      x = nx; y = ny;
    }
    positions[i * 3] = x; positions[i * 3 + 1] = y; positions[i * 3 + 2] = z;
  }
  return positions;
}

// Freeze representative procedural shapes without changing Math.random or
// depending on the browser's random generator.
function frozenProcedural(count, seed) {
  let counter = seed;
  const random = () => pcg(counter++) / 2 ** 32;
  const vec = (lo, hi) => [0, 1, 2].map(() => lo + (hi - lo) * random());
  const f = count > 3 ? Math.cbrt(3 / count) : 1;
  return Array.from({ length: count }, () => makeInstr({
    scale: vec(0.75, 0.85).map((x) => x * f),
    shearX: [0, ...vec(-0.1, 0.1).slice(0, 2)],
    shearY: [random() * 0.2 - 0.1, 0, random() * 0.2 - 0.1],
    shearZ: [...vec(-0.1, 0.1).slice(0, 2), 0],
    rot: quat.fromEuler(vec(-40, 40)), translate: vec(0, 1),
  }));
}

export const qualityFixtures = [
  ["triangle-2d", SierpinskiTriangle2D()],
  ["carpet-2d", SierpinskiCarpet2D()],
  ["pyramid-3d", SierpinskiTriangle3D()],
  ["vicsek-3d", Vicsek3D()],
  ["procedural-3", frozenProcedural(3, 31047)],
  ["procedural-8", frozenProcedural(8, 89213)],
].map(([name, instructions]) => ({ name, matrices: instructions.map(affineFromInstr) }));

function project(x, y, z) {
  // Orthographic view: fixed 0.55-radian yaw, 0.3-radian pitch.
  const rx = 0.8525245220595057 * x + 0.5226872289306592 * z;
  return [rx, 0.15446463791283113 * x + 0.955336489125606 * y - 0.25193822294282454 * z];
}

function projectedCopies(points, matrices, visit) {
  for (let i = 0; i < points.length; i += 3) {
    const x = points[i], y = points[i + 1], z = points[i + 2];
    for (const m of matrices) {
      const a = m[0] * x + m[4] * y + m[8] * z + m[12];
      const b = m[1] * x + m[5] * y + m[9] * z + m[13];
      const c = m[2] * x + m[6] * y + m[10] * z + m[14];
      const [px, py] = project(a, b, c);
      visit(px, py);
    }
  }
}

export function projectionBounds(matrices) {
  // Deep independent reference points plus fixed points establish one shared
  // projection for all candidates. A small margin avoids clipping outliers.
  const reference = batchPoints(matrices, 8192, 937, 80);
  const extrema = matrices.flatMap(fixedPoint);
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  const visit = (x, y) => {
    minX = Math.min(minX, x); maxX = Math.max(maxX, x);
    minY = Math.min(minY, y); maxY = Math.max(maxY, y);
  };
  projectedCopies(reference, matrices, visit);
  projectedCopies(extrema, matrices, visit);
  const extent = Math.max(maxX - minX, maxY - minY) * 1.06;
  return { minX: (maxX + minX - extent) / 2, minY: (maxY + minY - extent) / 2, extent };
}

export function collectImage(matrices, particles, batches, advanceHops, resolution = 512, bounds = projectionBounds(matrices)) {
  const image = new Uint32Array(resolution * resolution);
  const density = new Float64Array(32 * 32);
  const depth = fullDepth(matrices.length, particles);
  let points = null, covered = 0, clipped = 0, total = 0;
  for (let batch = 0; batch < batches; batch++) {
    const continued = advanceHops > 0 && batch > 0;
    points = batchPoints(matrices, particles, batch, continued ? advanceHops : depth, continued ? points : null);
    projectedCopies(points, matrices, (x, y) => {
      total++;
      const u = (x - bounds.minX) / bounds.extent, v = (y - bounds.minY) / bounds.extent;
      if (u < 0 || u >= 1 || v < 0 || v >= 1) { clipped++; return; }
      const index = Math.floor(v * resolution) * resolution + Math.floor(u * resolution);
      if (!image[index]) covered++;
      image[index]++;
      density[Math.floor(v * 32) * 32 + Math.floor(u * 32)]++;
    });
  }
  for (let i = 0; i < density.length; i++) density[i] /= total;
  return { image, density, covered, clipped, total,
    affineOperations: particles * (advanceHops > 0 ? depth + (batches - 1) * advanceHops : depth * batches),
  };
}

export function densityDistance(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += Math.abs(a[i] - b[i]);
  return sum / 2;
}

// Reproduce the larger quality experiment with:
//   node tests/helpers/sampling.mjs --benchmark
// Equal affine budgets intentionally do not equal GPU-time budgets: extra
// batches still require more rasterization, depth tests, and frame scheduling.
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href && process.argv.includes("--benchmark")) {
  const particles = 32768, batches = 8, resolution = 1024;
  console.log(JSON.stringify({ particles, batches, resolution,
    note: "CPU projected sampling quality; affine counts exclude rendering and are not GPU timings" }));
  for (const { name, matrices } of qualityFixtures) {
    const bounds = projectionBounds(matrices);
    const reference = collectImage(matrices, particles, batches, 0, resolution, bounds);
    const depth = fullDepth(matrices.length, particles);
    for (const hops of [1, 2, 4, 8]) {
      const candidate = collectImage(matrices, particles, batches, hops, resolution, bounds);
      const equalBatches = 1 + Math.floor(depth * (batches - 1) / hops);
      const equal = collectImage(matrices, particles, equalBatches, hops, resolution, bounds);
      console.log(JSON.stringify({ name, depth, hops, referenceCoverage: reference.covered,
        coverageRatio: candidate.covered / reference.covered,
        densityTV: densityDistance(candidate.density, reference.density),
        affineRatio: candidate.affineOperations / reference.affineOperations,
        equalBatches, equalAffineCoverageRatio: equal.covered / reference.covered,
        clipped: candidate.clipped }));
    }
  }
}
