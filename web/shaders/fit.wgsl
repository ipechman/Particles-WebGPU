// fit.wgsl
// Builds the auto-fit "final transform" that rescales and recenters the
// fractal to fill the target bounds box. Direct port of
// ReductionToTransformation in ParallelReduce.compute.

struct Fit {
  targetBounds: f32,
  scalePadding: f32,
  particleCount: f32,
  _pad: f32,
};

struct Bounds {
  mn: vec3<f32>,
  mx: vec3<f32>,
  sm: vec3<f32>,
};

@group(0) @binding(0) var<storage, read> result: array<Bounds>;
@group(0) @binding(1) var<storage, read_write> finalTransform: array<mat4x4<f32>>;
@group(0) @binding(2) var<uniform> u: Fit;

@compute @workgroup_size(1)
fn fit() {
  let minPos = result[0].mn;
  let maxPos = result[0].mx;
  let midPos = result[0].sm / u.particleCount;

  let boundsExtent = max(max(distance(minPos, midPos), distance(maxPos, midPos)), 1.0e-6);

  var rescale = u.targetBounds / boundsExtent;
  rescale = rescale * u.scalePadding;

  let m = midPos * rescale;

  finalTransform[0] = mat4x4<f32>(
    vec4<f32>(rescale, 0.0, 0.0, 0.0),
    vec4<f32>(0.0, rescale, 0.0, 0.0),
    vec4<f32>(0.0, 0.0, rescale, 0.0),
    vec4<f32>(-m.x, -m.y, -m.z, 1.0),
  );
}
