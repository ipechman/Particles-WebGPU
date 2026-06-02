// iterate.wgsl
// Builds the IFS attractor point cloud (the deterministic "iterated system"
// from UpdateParticles.compute / IteratedFunctionSystem.IterateSystem).
//
// One dispatch per generation. Point n derives from its parent
// floor((n-1)/count) by applying transform (n % count). Because each
// generation is a separate dispatch, parents always live in an already
// completed generation (WebGPU inserts a barrier between dispatches), so the
// reads are race free. The single root (index 0) is T0 applied to the origin.

struct Iter {
  count: u32,
  genOffset: u32,
  genLimit: u32,
  dispatchWidth: u32, // threads spanned by the x dispatch dimension (for 2D flattening)
};

@group(0) @binding(0) var<storage, read_write> positions: array<vec3<f32>>;
@group(0) @binding(1) var<storage, read> transforms: array<mat4x4<f32>>;
@group(0) @binding(2) var<uniform> u: Iter;

fn root() -> vec3<f32> {
  // T0 applied to the origin -> the 4th column's translation.
  return (transforms[0] * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;
}

@compute @workgroup_size(64)
fn reset(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * u.dispatchWidth + gid.x;
  if (i >= u.count) { return; }
  positions[i] = vec3<f32>(0.0, 0.0, 0.0);
}

@compute @workgroup_size(64)
fn iterate(@builtin(global_invocation_id) gid: vec3<u32>) {
  // 2D dispatch is flattened to a linear index so particle counts beyond
  // 65535*64 (the per-dimension workgroup cap) are reachable.
  let linearIndex = gid.y * u.dispatchWidth + gid.x;
  let threadID = linearIndex + u.genOffset;
  if (threadID >= u.genLimit) { return; }

  if (threadID == 0u) {
    positions[0] = root();
    return;
  }

  let parent = (threadID - 1u) / u.count;
  var seedPos: vec3<f32>;
  if (parent == 0u) {
    seedPos = root();
  } else {
    seedPos = positions[parent];
  }

  let m = transforms[threadID % u.count];
  positions[threadID] = (m * vec4<f32>(seedPos, 1.0)).xyz;
}
