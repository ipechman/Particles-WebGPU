// iterate.wgsl
// Builds the IFS attractor point cloud with a per-particle chaos game.
//
// Each particle is seeded on one of the transforms' fixed points (computed
// CPU-side; the fixed point of a contractive affine map lies exactly on the
// attractor) and then takes `iters` hops, each applying a hash-chosen
// transform. Every hop maps an attractor point to another attractor point
// exactly, so every rendered particle lies ON the attractor regardless of hop
// count — `iters` only controls how finely the address space is sampled and
// is chosen so distinct hop sequences comfortably outnumber the particles.
//
// All choices derive from a PCG hash of (particle index, batch seed), so the
// cloud is deterministic: identical transforms produce an identical cloud (no
// temporal shimmer, and the engine's static-frame compute cache stays valid).
// Distinct batch seeds give independent samples of the same distribution,
// which progressive accumulation uses on static frames.
//
// Compared with the previous breadth-first tree enumeration (one generation
// per dispatch, every tree node rendered) this runs in a single dispatch,
// upgrades the 1/count of the budget that was coarse interior scaffolding,
// and removes the convergence blur at high transform counts, where the
// deepest tree generation was only ~log_count(N) levels deep.

struct Chaos {
  count: u32,
  particleCount: u32,
  width: u32,     // threads spanned by the x dispatch dimension (2D flattening)
  iters: u32,
  batchSeed: u32,
  _p0: u32,
  _p1: u32,
  _p2: u32,
  seeds: array<vec4<f32>, 32>, // fixed point of each transform (xyz)
};

@group(0) @binding(0) var<storage, read_write> positions: array<vec3<f32>>;
@group(0) @binding(1) var<storage, read> transforms: array<mat4x4<f32>>;
@group(0) @binding(2) var<uniform> u: Chaos;

fn pcg(v: u32) -> u32 {
  let state = v * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

@compute @workgroup_size(64)
fn iterate(@builtin(global_invocation_id) gid: vec3<u32>) {
  // 2D dispatch is flattened to a linear index so particle counts beyond
  // 65535*64 (the per-dimension workgroup cap) are reachable.
  let idx = gid.y * u.width + gid.x;
  if (idx >= u.particleCount) { return; }

  var h = pcg(idx ^ pcg(u.batchSeed ^ 0x9E3779B9u));
  var pos = u.seeds[h % u.count].xyz;

  for (var k = 0u; k < u.iters; k = k + 1u) {
    h = pcg(h);
    pos = (transforms[h % u.count] * vec4<f32>(pos, 1.0)).xyz;
  }

  positions[idx] = pos;
}
