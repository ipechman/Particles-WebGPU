// redistribute.wgsl
// Surface-bias redistribution: re-seats particles buried inside the shape
// onto its visible outside. Runs after the occlusion pass and reads the
// openness field it produced (occlusionGrid = 1 - neighborOccupancy/27, so
// ~0 = buried, 1 = fully exposed). With probability `bias` a particle takes
// STEPS chaos-game hops (a randomly chosen attractor transform applied to
// itself, which always lands back on the attractor) and keeps the most open
// position it visited; otherwise it stays put. The slider is therefore the
// fraction of the particle budget reallocated toward open voxels, and the
// cloud never leaves the shape.
//
// The RNG is a hash of the particle index, so the pass is deterministic:
// identical transforms produce identical output, which keeps the engine's
// static-frame compute cache valid and avoids temporal shimmer.

struct Redist {
  gridSize: u32,
  transformCount: u32,
  particleCount: u32,
  width: u32,        // threads spanned by the x dispatch dimension (2D flattening)
  gridBounds: f32,
  bias: f32,         // 0 = no-op (the dispatch is skipped), 1 = full surface bias
  _p0: f32,
  _p1: f32,
};

@group(0) @binding(0) var<storage, read_write> positions: array<vec3<f32>>;
@group(0) @binding(1) var<storage, read> transforms: array<mat4x4<f32>>;
@group(0) @binding(2) var<storage, read> finalTransform: array<mat4x4<f32>>;
@group(0) @binding(3) var<storage, read> occlusionGrid: array<f32>;
@group(0) @binding(4) var<uniform> u: Redist;

const STEPS = 16u;  // chaos-game hops sampled per re-seated particle

fn pcg(v: u32) -> u32 {
  let state = v * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn to1D(p: vec3<u32>) -> u32 {
  return p.x + p.y * u.gridSize + p.z * u.gridSize * u.gridSize;
}

// Openness of the voxel the point lands in after the auto-fit transform (the
// same mapping voxelize.wgsl marks). Points outside the grid count as fully
// open so they are left where they are.
fn openness(pos: vec3<f32>) -> f32 {
  let world = (finalTransform[0] * vec4<f32>(pos, 1.0)).xyz;
  let half = u.gridBounds * 0.5;
  if (any(abs(world) > vec3<f32>(half))) { return 1.0; }
  var c = (world + vec3<f32>(half)) / u.gridBounds * f32(u.gridSize);
  c = clamp(c, vec3<f32>(0.0), vec3<f32>(f32(u.gridSize) - 1.0));
  return occlusionGrid[to1D(vec3<u32>(c))];
}

@compute @workgroup_size(64)
fn redistribute(@builtin(global_invocation_id) gid: vec3<u32>) {
  // 2D dispatch flattened to a linear index (see iterate.wgsl).
  let idx = gid.y * u.width + gid.x;
  if (idx >= u.particleCount) { return; }

  var seed = pcg(idx ^ 0x9E3779B9u);
  if (f32(seed) * 2.3283064e-10 >= u.bias) { return; }

  var cur = positions[idx];
  var best = cur;
  var bestScore = openness(cur);

  for (var k = 0u; k < STEPS; k = k + 1u) {
    seed = pcg(seed);
    cur = (transforms[seed % u.transformCount] * vec4<f32>(cur, 1.0)).xyz;
    let score = openness(cur);
    if (score > bestScore) {
      bestScore = score;
      best = cur;
    }
  }

  positions[idx] = best;
}
