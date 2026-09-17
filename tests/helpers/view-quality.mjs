import { batchPoints, fullDepth } from "./sampling.mjs";
import { packViewLeaves, VIEW_LEAF_BYTES } from "../../web/js/view-sampling.js";

export function transformPoint(m, p) {
  return [0, 1, 2].map(r => m[r] * p[0] + m[4 + r] * p[1] + m[8 + r] * p[2] + m[12 + r]);
}

export function focusedPoints(matrices, plan, particles, batch = 0) {
  const points = batchPoints(matrices, particles, batch, fullDepth(matrices.length, particles));
  const packed = packViewLeaves(plan.leaves, particles);
  let leaf = 0;
  for (let i = 0; i < particles; i++) {
    while (Math.floor(i / 64) >= new Uint32Array(packed, leaf * VIEW_LEAF_BYTES + 64, 1)[0]) leaf++;
    const m = new Float32Array(packed, leaf * VIEW_LEAF_BYTES, 16);
    points.set(transformPoint(m, points.subarray(i * 3, i * 3 + 3)), i * 3);
  }
  return points;
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
