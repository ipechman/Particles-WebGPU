// voxelize.wgsl
// Voxelizes the point cloud into a 3D grid and computes the brute-force
// ambient-occlusion approximation used for lighting. Ports ClearVoxelBuffer,
// ClearOcclusion, VoxelizePositions and CalculateOcclusion from
// UpdateParticles.compute.

struct Grid {
  gridSize: u32,
  transformCount: u32,
  particleCount: u32,
  voxelCount: u32,
  gridBounds: f32,
  voxWidth: u32, // threads spanned by the x dispatch dimension (2D flattening)
  sampleCount: u32,
  particleStride: u32,
};

struct Point { x: f32, y: f32, z: f32 };

@group(0) @binding(0) var<storage, read> positions: array<Point>;
// combined[i] = finalTransform * transforms[i] (see combine.wgsl)
@group(0) @binding(1) var<storage, read> combined: array<mat4x4<f32>>;
@group(0) @binding(2) var<storage, read> finalTransform: array<mat4x4<f32>>;
@group(0) @binding(3) var<storage, read_write> voxelGrid: array<atomic<u32>>;
// AO is written to a real 3D texture so the render pass can sample it with
// one hardware-filtered trilinear tap instead of 8 buffer loads.
@group(0) @binding(4) var occlusionTex: texture_storage_3d<rgba16float, write>;
@group(0) @binding(5) var<uniform> u: Grid;

fn to1D(p: vec3<u32>) -> u32 {
  return p.x + p.y * u.gridSize + p.z * u.gridSize * u.gridSize;
}

fn cellOf(world: vec3<f32>) -> vec3<f32> {
  var p = world + vec3<f32>(u.gridBounds * 0.5);
  p = p / u.gridBounds;
  p = p * f32(u.gridSize);
  return p;
}

fn mark(world: vec3<f32>) {
  let c = cellOf(world);
  let g = f32(u.gridSize);
  if (all(c >= vec3<f32>(0.0)) && all(c < vec3<f32>(g))) {
    // Many particles can hit the same cell, so even identical writes must be
    // atomic. Later passes observe the completed binary occupancy grid.
    atomicStore(&voxelGrid[to1D(vec3<u32>(c))], 1u);
  }
}

@compute @workgroup_size(64)
fn clearGrids(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= u.voxelCount) { return; }
  atomicStore(&voxelGrid[i], 0u);
  // occlusionTex needs no clear: the occlusion pass writes every voxel.
}

@compute @workgroup_size(64)
fn voxelize(@builtin(global_invocation_id) gid: vec3<u32>) {
  // 2D dispatch flattened to a linear index (see iterate.wgsl).
  let sampleIdx = gid.y * u.voxWidth + gid.x;
  if (sampleIdx >= u.sampleCount) { return; }
  // The engine chooses ceil(particleCount / stride) samples. Integer stride
  // keeps samples distributed over the cloud without float rounding or a
  // sampleIdx * particleCount multiplication that could overflow u32.
  let idx = sampleIdx * u.particleStride;
  if (idx >= u.particleCount) { return; }

  let point = positions[idx];
  let pos = vec3<f32>(point.x, point.y, point.z);
  let ft = finalTransform[0];

  // The point itself.
  mark((ft * vec4<f32>(pos, 1.0)).xyz);

  // ...and its image under each top-level transform (matches the instanced
  // render, which draws transformCount copies).
  for (var i = 0u; i < u.transformCount; i = i + 1u) {
    let p = combined[i] * vec4<f32>(pos, 1.0);
    mark(p.xyz);
  }
}

// 8x8x4 output cells share a 10x10x6 occupancy tile including a one-cell halo.
// This replaces up to 26 global loads per cell with 600 loads per workgroup.
var<workgroup> occupancyTile: array<u32, 600>;

@compute @workgroup_size(8, 8, 4)
fn occlusion(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(local_invocation_index) linearLid: u32,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  let g = i32(u.gridSize);
  let origin = vec3<i32>(wid * vec3<u32>(8u, 8u, 4u)) - vec3<i32>(1);

  // Every invocation participates, including those beyond a partial edge
  // tile. Out-of-grid halo cells are zero, preserving the boundary behavior.
  for (var i = linearLid; i < 600u; i = i + 256u) {
    let offset = vec3<i32>(i32(i % 10u), i32((i / 10u) % 10u), i32(i / 100u));
    let p = origin + offset;
    var occupied = 0u;
    if (all(p >= vec3<i32>(0)) && all(p < vec3<i32>(g))) {
      occupied = atomicLoad(&voxelGrid[to1D(vec3<u32>(p))]);
    }
    occupancyTile[i] = occupied;
  }
  workgroupBarrier();

  if (any(gid >= vec3<u32>(u.gridSize))) { return; }
  let center = vec3<i32>(lid) + vec3<i32>(1);

  var neighborCount = 0u;
  for (var x = -1; x <= 1; x = x + 1) {
    for (var y = -1; y <= 1; y = y + 1) {
      for (var z = -1; z <= 1; z = z + 1) {
        if (x == 0 && y == 0 && z == 0) { continue; }
        let n = center - vec3<i32>(x, y, z);
        let tileIndex = u32(n.x + n.y * 10 + n.z * 100);
        neighborCount = neighborCount + occupancyTile[tileIndex];
      }
    }
  }

  let occ = f32(neighborCount) / 27.0;
  textureStore(occlusionTex, gid, vec4<f32>(1.0 - occ, 0.0, 0.0, 0.0));
}
