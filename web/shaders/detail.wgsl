// Small-scale depth shading at the displayed resolution. Background samples
// never create edges or halos; the pass only darkens existing geometry.
struct Detail { camera: vec4<f32>, _pad: vec4<f32> };
@group(0) @binding(0) var scene: texture_2d<f32>;
@group(0) @binding(1) var depth: texture_depth_2d;
@group(0) @binding(2) var<uniform> u: Detail;

@vertex
fn vsFull(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
  let xy = vec2<f32>(f32((vid << 1u) & 2u), f32(vid & 2u));
  return vec4<f32>(xy * 2.0 - 1.0, 0.0, 1.0);
}

fn viewDistance(z: f32) -> f32 {
  let near = u.camera.x;
  let far = u.camera.y;
  return near * far / max(far - z * (far - near), 1e-7);
}

@fragment
fn fsDetail(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
  let pixel = vec2<i32>(position.xy);
  let base = textureLoad(scene, pixel, 0);
  let z = textureLoad(depth, pixel, 0);
  if (z >= 1.0 || u.camera.w <= 0.0) { return base; }
  let center = viewDistance(z);
  let pixelSize = max(center * u.camera.z, 1e-8);
  let dimensions = vec2<i32>(textureDimensions(depth));
  let offsets = array<vec2<i32>, 8>(
    vec2<i32>(1, 0), vec2<i32>(-1, 0), vec2<i32>(0, 1), vec2<i32>(0, -1),
    vec2<i32>(1, 1), vec2<i32>(-1, 1), vec2<i32>(1, -1), vec2<i32>(-1, -1));
  var occlusion = 0.0;
  var samples = 0.0;
  for (var ring = 0; ring < 2; ring++) {
    let radius = select(2, 5, ring == 1);
    for (var i = 0; i < 8; i++) {
      let at = pixel + offsets[i] * radius;
      if (any(at < vec2<i32>(0)) || any(at >= dimensions)) { continue; }
      let other = textureLoad(depth, at, 0);
      if (other >= 1.0) { continue; }
      let scale = pixelSize * length(vec2<f32>(offsets[i] * radius));
      let delta = (center - viewDistance(other)) / scale;
      // Ignore unrelated distant surfaces. The paired directions avoid a
      // preferred light direction, and coplanar points receive no darkening.
      occlusion += smoothstep(0.05, 1.0, delta) * (1.0 - smoothstep(4.0, 12.0, abs(delta)));
      samples += 1.0;
    }
  }
  let shade = 1.0 - 0.7 * u.camera.w * occlusion / max(samples, 1.0);
  return vec4<f32>(base.rgb * shade, base.a);
}
