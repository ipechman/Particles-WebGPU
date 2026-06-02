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
  _p1: f32,
  _p2: f32,
};

@group(0) @binding(0) var<storage, read> positions: array<vec3<f32>>;
@group(0) @binding(1) var<storage, read> transforms: array<mat4x4<f32>>;
@group(0) @binding(2) var<storage, read> finalTransform: array<mat4x4<f32>>;
@group(0) @binding(3) var<storage, read_write> voxelGrid: array<u32>;
@group(0) @binding(4) var<storage, read_write> occlusionGrid: array<f32>;
@group(0) @binding(5) var<uniform> u: Grid;

fn to1D(p: vec3<u32>) -> u32 {
  return p.x + p.y * u.gridSize + p.z * u.gridSize * u.gridSize;
}

fn to3D(idx: u32) -> vec3<u32> {
  let res = u.gridSize;
  let x = idx % res;
  let y = (idx / res) % res;
  let z = idx / (res * res);
  return vec3<u32>(x, y, z);
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
    voxelGrid[to1D(vec3<u32>(c))] = 1u;
  }
}

@compute @workgroup_size(64)
fn clearGrids(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= u.voxelCount) { return; }
  voxelGrid[i] = 0u;
  occlusionGrid[i] = 0.0;
}

@compute @workgroup_size(64)
fn voxelize(@builtin(global_invocation_id) gid: vec3<u32>) {
  // 2D dispatch flattened to a linear index (see iterate.wgsl).
  let idx = gid.y * u.voxWidth + gid.x;
  if (idx >= u.particleCount) { return; }

  let pos = positions[idx];
  let ft = finalTransform[0];

  // The point itself.
  mark((ft * vec4<f32>(pos, 1.0)).xyz);

  // ...and its image under each top-level transform (matches the instanced
  // render, which draws transformCount copies).
  for (var i = 0u; i < u.transformCount; i = i + 1u) {
    let p = ft * (transforms[i] * vec4<f32>(pos, 1.0));
    mark(p.xyz);
  }
}

@compute @workgroup_size(64)
fn occlusion(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= u.voxelCount) { return; }

  let pos = vec3<i32>(to3D(i));
  let g = i32(u.gridSize);

  var neighborCount = 0;
  for (var x = -1; x <= 1; x = x + 1) {
    for (var y = -1; y <= 1; y = y + 1) {
      for (var z = -1; z <= 1; z = z + 1) {
        if (x == 0 && y == 0 && z == 0) { continue; }
        let n = pos - vec3<i32>(x, y, z);
        if (all(n >= vec3<i32>(0)) && all(n < vec3<i32>(g))) {
          neighborCount = neighborCount + i32(voxelGrid[to1D(vec3<u32>(n))]);
        }
      }
    }
  }

  let occ = f32(neighborCount) / 27.0;
  occlusionGrid[i] = 1.0 - occ;
}
