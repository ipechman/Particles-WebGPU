// present.wgsl
// Final composite to the swap chain: base image + bloom * intensity.

struct PresentParams {
  p0: vec4<f32>, // .x = bloom intensity
  p1: vec4<f32>,
};

@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var sceneTex: texture_2d<f32>;
@group(0) @binding(2) var bloomTex: texture_2d<f32>;
@group(0) @binding(3) var<uniform> u: PresentParams;

struct VOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vsFull(@builtin(vertex_index) vid: u32) -> VOut {
  var verts = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(3.0, -1.0),
    vec2<f32>(-1.0, 3.0),
  );
  let p = verts[vid];
  var o: VOut;
  o.pos = vec4<f32>(p, 0.0, 1.0);
  o.uv = vec2<f32>(p.x * 0.5 + 0.5, 1.0 - (p.y * 0.5 + 0.5));
  return o;
}

@fragment
fn fsPresent(i: VOut) -> @location(0) vec4<f32> {
  let base = textureSampleLevel(sceneTex, samp, i.uv, 0.0).rgb;
  let bloom = textureSampleLevel(bloomTex, samp, i.uv, 0.0).rgb;
  let col = clamp(base + bloom * u.p0.x, vec3<f32>(0.0), vec3<f32>(1.0));
  return vec4<f32>(col, 1.0);
}
