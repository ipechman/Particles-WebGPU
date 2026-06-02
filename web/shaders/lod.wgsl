// lod.wgsl
// Generates a low-detail copy of the attractor used for bounds prediction
// (FirstLODIteration / LODIteration in ParallelReduce.compute). Each
// generation expands every point into `count` children by applying all
// transforms. Buffers ping-pong between dispatches.

struct Lod {
  count: u32,
  inCount: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> lodIn: array<vec3<f32>>;
@group(0) @binding(1) var<storage, read_write> lodOut: array<vec3<f32>>;
@group(0) @binding(2) var<storage, read> transforms: array<mat4x4<f32>>;
@group(0) @binding(3) var<uniform> u: Lod;

@compute @workgroup_size(64)
fn lodFirst(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= u.count) { return; }
  lodOut[gid.x] = (transforms[gid.x] * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;
}

@compute @workgroup_size(64)
fn lodIterate(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= u.inCount) { return; }
  let seed = lodIn[gid.x];
  let base = gid.x * u.count;
  for (var i = 0u; i < u.count; i = i + 1u) {
    lodOut[base + i] = (transforms[i] * vec4<f32>(seed, 1.0)).xyz;
  }
}
