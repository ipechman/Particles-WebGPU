import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

// CPU reference for the shader's shared-memory indexing. Native WebGPU smoke
// tests remain necessary to check WGSL compilation and GPU pass integration.
const shader = readFileSync(new URL("../web/shaders/voxelize.wgsl", import.meta.url), "utf8");
const tileSize = [8, 8, 4];
const haloSize = tileSize.map((n) => n + 2);
const haloCells = haloSize.reduce((a, b) => a * b);
const lanes = tileSize.reduce((a, b) => a * b);

function directAO(grid, dim) {
  const output = new Float64Array(grid.length);
  for (let z = 0; z < dim; z++) {
    for (let y = 0; y < dim; y++) {
      for (let x = 0; x < dim; x++) {
        let occupied = 0;
        for (let nz = Math.max(0, z - 1); nz <= Math.min(dim - 1, z + 1); nz++) {
          for (let ny = Math.max(0, y - 1); ny <= Math.min(dim - 1, y + 1); ny++) {
            for (let nx = Math.max(0, x - 1); nx <= Math.min(dim - 1, x + 1); nx++) {
              if (nx !== x || ny !== y || nz !== z) occupied += grid[nx + ny * dim + nz * dim * dim];
            }
          }
        }
        output[x + y * dim + z * dim * dim] = 1 - occupied / 27;
      }
    }
  }
  return output;
}

function tiledAO(grid, dim) {
  const output = new Float64Array(grid.length).fill(NaN);
  const writes = new Uint8Array(grid.length);
  for (let wz = 0; wz < Math.ceil(dim / tileSize[2]); wz++) {
    for (let wy = 0; wy < Math.ceil(dim / tileSize[1]); wy++) {
      for (let wx = 0; wx < Math.ceil(dim / tileSize[0]); wx++) {
        const tile = new Uint8Array(haloCells);
        const loaded = new Uint8Array(haloCells);
        // Simulate cooperative strided loads including all lanes in edge groups.
        for (let lane = 0; lane < lanes; lane++) {
          for (let i = lane; i < haloCells; i += lanes) {
            const x = wx * tileSize[0] - 1 + i % haloSize[0];
            const y = wy * tileSize[1] - 1 + Math.floor(i / haloSize[0]) % haloSize[1];
            const z = wz * tileSize[2] - 1 + Math.floor(i / (haloSize[0] * haloSize[1]));
            if (x >= 0 && y >= 0 && z >= 0 && x < dim && y < dim && z < dim) {
              tile[i] = grid[x + y * dim + z * dim * dim];
            }
            loaded[i]++;
          }
        }
        assert.ok(loaded.every((n) => n === 1), "every halo cell has exactly one loader");
        for (let lz = 0; lz < tileSize[2]; lz++) {
          for (let ly = 0; ly < tileSize[1]; ly++) {
            for (let lx = 0; lx < tileSize[0]; lx++) {
              const x = wx * tileSize[0] + lx;
              const y = wy * tileSize[1] + ly;
              const z = wz * tileSize[2] + lz;
              if (x >= dim || y >= dim || z >= dim) continue;
              let occupied = 0;
              for (let dx = -1; dx <= 1; dx++) {
                for (let dy = -1; dy <= 1; dy++) {
                  for (let dz = -1; dz <= 1; dz++) {
                    if (dx === 0 && dy === 0 && dz === 0) continue;
                    const i = lx + 1 - dx + (ly + 1 - dy) * haloSize[0]
                      + (lz + 1 - dz) * haloSize[0] * haloSize[1];
                    assert.ok(i >= 0 && i < haloCells, "neighbor stays inside shared tile");
                    occupied += tile[i];
                  }
                }
              }
              const i = x + y * dim + z * dim * dim;
              output[i] = 1 - occupied / 27;
              writes[i]++;
            }
          }
        }
      }
    }
  }
  assert.ok(writes.every((n) => n === 1), "each output voxel is written exactly once");
  return output;
}

test("AO CPU indexing model stays synchronized with shader tile dimensions", () => {
  assert.match(shader, /@workgroup_size\(8, 8, 4\)\s+fn occlusion/);
  assert.match(shader, /occupancyTile: array<u32, 600>/);
  assert.match(shader, /i = i \+ 256u/);
  assert.match(shader, /n\.x \+ n\.y \* 10 \+ n\.z \* 100/);
  assert.match(shader, /f32\(neighborCount\) \/ 27\.0/);
});

test("tiled AO matches independent reference at grid, tile, and partial-group boundaries", () => {
  let seed = 0x12345678;
  const randomBit = () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed >>> 31;
  };
  for (const dim of [1, 2, 3, 4, 7, 8, 9, 13, 16, 17]) {
    for (const pattern of ["empty", "full", "random", "corner", "tile-edge"]) {
      const grid = Uint8Array.from({ length: dim ** 3 }, (_, i) => {
        if (pattern === "full") return 1;
        if (pattern === "random") return randomBit();
        if (pattern === "corner") return Number(i === 0);
        if (pattern === "tile-edge") return Number(i % dim === Math.min(dim - 1, 7));
        return 0;
      });
      assert.deepEqual(tiledAO(grid, dim), directAO(grid, dim), `dim=${dim}, ${pattern}`);
    }
  }
});

test("AO excludes the center voxel and preserves the original 27-cell normalization", () => {
  assert.deepEqual([...tiledAO(Uint8Array.of(1), 1)], [1]);
  const result = tiledAO(new Uint8Array(27).fill(1), 3);
  assert.equal(result[13], 1 - 26 / 27);
  assert.equal(result[0], 1 - 7 / 27);
});
