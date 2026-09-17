import { batchPoints, fullDepth } from "./sampling.mjs";
import { packViewDraws } from "../../web/js/view-sampling.js";
import { mat4 } from "../../web/js/math.js";

export function transformPoint(m, p) {
  return [0, 1, 2].map(r => m[r] * p[0] + m[4 + r] * p[1] + m[8 + r] * p[2] + m[12 + r]);
}

export function focusedPoints(matrices, plan, particles, batch = 0, weights = plan.leaves.map(l => l.mass)) {
  const base = batchPoints(matrices, particles, batch, fullDepth(matrices.length, particles));
  const points = new Float32Array(particles * matrices.length * 3);
  const { draws } = packViewDraws(plan.leaves, particles, matrices.length, mat4.identity(), weights);
  let at = 0;
  for (const draw of draws) for (let i = draw.first; i < draw.first + draw.count; i++) {
    const child = matrices[(i + draw.copy) % matrices.length];
    points.set(transformPoint(draw.matrix, transformPoint(child, base.subarray(i * 3, i * 3 + 3))), at++ * 3);
  }
  return points;
}

export function sampleFit(points) {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity], mean = [0, 0, 0];
  const n = points.length / 3;
  for (let i = 0; i < points.length; i++) {
    const r = i % 3;
    lo[r] = Math.min(lo[r], points[i]); hi[r] = Math.max(hi[r], points[i]); mean[r] += points[i] / n;
  }
  const extent = Math.max(Math.hypot(...lo.map((v, r) => v - mean[r])), Math.hypot(...hi.map((v, r) => v - mean[r])));
  const fit = mat4.identity(), scale = 1.4925 / extent;
  fit[0] = fit[5] = fit[10] = scale;
  for (let r = 0; r < 3; r++) fit[12 + r] = -mean[r] * scale;
  return fit;
}

export function centeredView(distance, aspect = 4 / 3) {
  const eye = [Math.cos(0.45) * Math.cos(0.6), Math.sin(0.45), Math.cos(0.45) * Math.sin(0.6)].map(v => v * distance);
  return mat4.multiply(mat4.perspective(Math.PI / 3, aspect, distance * 0.004, distance * 60 + 50),
    mat4.lookAt(eye, [0, 0, 0], [0, 1, 0]));
}

// Independent pixel/depth projection, including the old renderer's M copies.
export function projectPoints(points, matrices, view, width, height) {
  const depth = new Float32Array(width * height).fill(1);
  let onScreen = 0, submitted = 0;
  for (let i = 0; i < points.length; i += 3) {
    for (const m of matrices) {
      submitted++;
      const p = transformPoint(m, points.subarray(i, i + 3));
      const [x, y, z] = transformPoint(view, p);
      const w = view[3] * p[0] + view[7] * p[1] + view[11] * p[2] + view[15];
      if (w <= 0 || x < -w || x >= w || y < -w || y >= w || z < 0 || z >= w) continue;
      onScreen++;
      const px = Math.floor((x / w * 0.5 + 0.5) * width);
      const py = Math.floor((0.5 - y / w * 0.5) * height);
      const at = py * width + px;
      depth[at] = Math.min(depth[at], z / w);
    }
  }
  return { depth, onScreen, submitted, covered: depth.reduce((sum, z) => sum + Number(z < 1), 0) };
}
