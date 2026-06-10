// combine.wgsl
// Pre-multiplies the auto-fit (final) transform into each top-level transform
// right after the fit pass, so the render vertex shader and the voxelizer
// apply ONE matrix per point instead of two. With ~25M instanced vertices per
// frame this halves the per-vertex matrix work.

@group(0) @binding(0) var<storage, read> transforms: array<mat4x4<f32>>;
@group(0) @binding(1) var<storage, read> finalTransform: array<mat4x4<f32>>;
@group(0) @binding(2) var<storage, read_write> combined: array<mat4x4<f32>>;
@group(0) @binding(3) var<uniform> u: vec4<u32>; // x = transformCount

@compute @workgroup_size(32)
fn combine(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= u.x) { return; }
  combined[gid.x] = finalTransform[0] * transforms[gid.x];
}
