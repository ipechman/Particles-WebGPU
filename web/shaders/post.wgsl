// post.wgsl
// Post-processing passes that share one bind group layout:
//   binding 0: sampler   binding 1: input texture   binding 2: PostParams
//
//   fsKuwahara  - the original (basic) Kuwahara filter (Acerola). For each
//                 pixel it splits a square window into four overlapping
//                 quadrants, and outputs the mean colour of the quadrant with
//                 the lowest luminance variance -> edge-preserving painterly look.
//   fsPrefilter - bloom bright-pass (soft threshold).
//   fsBlur      - separable Gaussian blur (direction in PostParams.p1).

struct PostParams {
  p0: vec4<f32>, // .xy = texel size, .z = radius / threshold
  p1: vec4<f32>, // .xy = blur direction, .z = spread
};

@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> u: PostParams;

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

fn lum(c: vec3<f32>) -> f32 {
  return dot(c, vec3<f32>(0.299, 0.587, 0.114));
}

// One Kuwahara quadrant: accumulate mean colour and luminance variance over an
// (radius+1)x(radius+1) block, then keep it if it is the smoothest so far.
fn sampleQuadrant(uv: vec2<f32>, texel: vec2<f32>, radius: i32,
                  x0: i32, x1: i32, y0: i32, y1: i32,
                  bestVar: ptr<function, f32>, bestColor: ptr<function, vec3<f32>>) {
  var sum = vec3<f32>(0.0);
  var sum2 = 0.0;
  var n = 0.0;
  for (var y = y0; y <= y1; y = y + 1) {
    for (var x = x0; x <= x1; x = x + 1) {
      let c = textureSampleLevel(tex, samp, uv + vec2<f32>(f32(x), f32(y)) * texel, 0.0).rgb;
      sum = sum + c;
      let l = lum(c);
      sum2 = sum2 + l * l;
      n = n + 1.0;
    }
  }
  let mean = sum / n;
  let meanLum = lum(mean);
  let variance = sum2 / n - meanLum * meanLum;
  if (variance < *bestVar) {
    *bestVar = variance;
    *bestColor = mean;
  }
}

@fragment
fn fsKuwahara(i: VOut) -> @location(0) vec4<f32> {
  let texel = u.p0.xy;
  let radius = i32(u.p0.z);

  var bestVar = 1.0e20;
  var bestColor = textureSampleLevel(tex, samp, i.uv, 0.0).rgb;

  // Four overlapping quadrants around the pixel.
  sampleQuadrant(i.uv, texel, radius, -radius, 0, -radius, 0, &bestVar, &bestColor);
  sampleQuadrant(i.uv, texel, radius, 0, radius, -radius, 0, &bestVar, &bestColor);
  sampleQuadrant(i.uv, texel, radius, -radius, 0, 0, radius, &bestVar, &bestColor);
  sampleQuadrant(i.uv, texel, radius, 0, radius, 0, radius, &bestVar, &bestColor);

  return vec4<f32>(bestColor, 1.0);
}

@fragment
fn fsPrefilter(i: VOut) -> @location(0) vec4<f32> {
  let threshold = u.p0.z;
  let c = textureSampleLevel(tex, samp, i.uv, 0.0).rgb;
  let brightness = max(max(c.r, c.g), c.b);
  // Pixels above the threshold pass through at (close to) full colour so the
  // bloom has real energy to spread.
  let contrib = smoothstep(threshold, threshold + 0.25, brightness);
  return vec4<f32>(c * contrib, 1.0);
}

@fragment
fn fsBlur(i: VOut) -> @location(0) vec4<f32> {
  let step = u.p1.xy * u.p0.xy * u.p1.z; // direction * texel * spread

  // 9-tap Gaussian.
  let w0 = 0.227027;
  let w1 = 0.1945946;
  let w2 = 0.1216216;
  let w3 = 0.054054;
  let w4 = 0.016216;

  var col = textureSampleLevel(tex, samp, i.uv, 0.0).rgb * w0;
  col = col + textureSampleLevel(tex, samp, i.uv + step * 1.0, 0.0).rgb * w1;
  col = col + textureSampleLevel(tex, samp, i.uv - step * 1.0, 0.0).rgb * w1;
  col = col + textureSampleLevel(tex, samp, i.uv + step * 2.0, 0.0).rgb * w2;
  col = col + textureSampleLevel(tex, samp, i.uv - step * 2.0, 0.0).rgb * w2;
  col = col + textureSampleLevel(tex, samp, i.uv + step * 3.0, 0.0).rgb * w3;
  col = col + textureSampleLevel(tex, samp, i.uv - step * 3.0, 0.0).rgb * w3;
  col = col + textureSampleLevel(tex, samp, i.uv + step * 4.0, 0.0).rgb * w4;
  col = col + textureSampleLevel(tex, samp, i.uv - step * 4.0, 0.0).rgb * w4;
  return vec4<f32>(col, 1.0);
}
