// kuwahara.wgsl
// Anisotropic Kuwahara filter, ported from Acerola's AcerolaFX_KuwaharaFilter.fx.
// Four full-screen passes, all sharing one bind group layout:
//   binding 0: primary input texture   binding 1: secondary input (TFM)
//   binding 2: KuwParams
//
//   fsStructureTensor - Sobel gradients -> structure tensor (Sxx, Syy, Sxy)
//   fsTensorBlurH     - horizontal gaussian blur of the tensor
//   fsAnisotropy      - vertical gaussian blur + eigen-analysis -> flow map (t, phi, A)
//   fsKuwahara        - oriented, sectored Kuwahara using the flow map

const PI: f32 = 3.14159265358979;

struct KuwParams {
  kernelSize: f32,   // window size (radius = kernelSize / 2)
  q: f32,            // sharpness
  alpha: f32,        // kernel eccentricity
  zeroCrossing: f32, // sector overlap
  blurRadius: f32,   // tensor blur radius
  _p0: f32,
  _p1: f32,
  _p2: f32,
};

@group(0) @binding(0) var texA: texture_2d<f32>;
@group(0) @binding(1) var texB: texture_2d<f32>;
@group(0) @binding(2) var<uniform> u: KuwParams;

struct VOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vsFull(@builtin(vertex_index) vid: u32) -> VOut {
  var verts = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0),
  );
  let p = verts[vid];
  var o: VOut;
  o.pos = vec4<f32>(p, 0.0, 1.0);
  o.uv = vec2<f32>(p.x * 0.5 + 0.5, 1.0 - (p.y * 0.5 + 0.5));
  return o;
}

fn loadA(p: vec2<i32>, dim: vec2<i32>) -> vec3<f32> {
  return textureLoad(texA, clamp(p, vec2<i32>(0), dim - vec2<i32>(1)), 0).rgb;
}
fn load4A(p: vec2<i32>, dim: vec2<i32>) -> vec4<f32> {
  return textureLoad(texA, clamp(p, vec2<i32>(0), dim - vec2<i32>(1)), 0);
}
fn gaussian(sigma: f32, pos: f32) -> f32 {
  return (1.0 / sqrt(2.0 * PI * sigma * sigma)) * exp(-(pos * pos) / (2.0 * sigma * sigma));
}

@fragment
fn fsStructureTensor(i: VOut) -> @location(0) vec4<f32> {
  let dim = vec2<i32>(textureDimensions(texA));
  let p = vec2<i32>(i.pos.xy);

  let Sx = (
     1.0 * loadA(p + vec2<i32>(-1, -1), dim) + 2.0 * loadA(p + vec2<i32>(-1, 0), dim) + 1.0 * loadA(p + vec2<i32>(-1, 1), dim) +
    -1.0 * loadA(p + vec2<i32>( 1, -1), dim) - 2.0 * loadA(p + vec2<i32>( 1, 0), dim) - 1.0 * loadA(p + vec2<i32>( 1, 1), dim)
  ) / 4.0;

  let Sy = (
     1.0 * loadA(p + vec2<i32>(-1, -1), dim) + 2.0 * loadA(p + vec2<i32>(0, -1), dim) + 1.0 * loadA(p + vec2<i32>(1, -1), dim) +
    -1.0 * loadA(p + vec2<i32>(-1,  1), dim) - 2.0 * loadA(p + vec2<i32>(0,  1), dim) - 1.0 * loadA(p + vec2<i32>(1,  1), dim)
  ) / 4.0;

  return vec4<f32>(dot(Sx, Sx), dot(Sy, Sy), dot(Sx, Sy), 1.0);
}

@fragment
fn fsTensorBlurH(i: VOut) -> @location(0) vec4<f32> {
  let dim = vec2<i32>(textureDimensions(texA));
  let p = vec2<i32>(i.pos.xy);
  let r = i32(u.blurRadius);
  var col = vec4<f32>(0.0);
  var sum = 0.0;
  for (var x = -r; x <= r; x = x + 1) {
    let g = gaussian(2.0, f32(x));
    col = col + load4A(p + vec2<i32>(x, 0), dim) * g;
    sum = sum + g;
  }
  return col / sum;
}

@fragment
fn fsAnisotropy(i: VOut) -> @location(0) vec4<f32> {
  let dim = vec2<i32>(textureDimensions(texA));
  let p = vec2<i32>(i.pos.xy);
  let r = i32(u.blurRadius);
  var col = vec4<f32>(0.0);
  var sum = 0.0;
  for (var y = -r; y <= r; y = y + 1) {
    let g = gaussian(2.0, f32(y));
    col = col + load4A(p + vec2<i32>(0, y), dim) * g;
    sum = sum + g;
  }
  let g = col.rgb / sum;

  let disc = sqrt(g.y * g.y - 2.0 * g.x * g.y + g.x * g.x + 4.0 * g.z * g.z);
  let lambda1 = 0.5 * (g.y + g.x + disc);
  let lambda2 = 0.5 * (g.y + g.x - disc);

  let v = vec2<f32>(lambda1 - g.x, -g.z);
  var t = vec2<f32>(0.0, 1.0);
  if (length(v) > 0.0) { t = normalize(v); }
  let phi = -atan2(t.y, t.x);

  var A = 0.0;
  if (lambda1 + lambda2 > 0.0) { A = (lambda1 - lambda2) / (lambda1 + lambda2); }

  return vec4<f32>(t, phi, A);
}

@fragment
fn fsKuwahara(i: VOut) -> @location(0) vec4<f32> {
  let dim = vec2<i32>(textureDimensions(texA));
  let p = vec2<i32>(i.pos.xy);

  let alpha = u.alpha;
  let t = textureLoad(texB, clamp(p, vec2<i32>(0), dim - vec2<i32>(1)), 0);

  let radius = max(1.0, u.kernelSize * 0.5);
  let a = radius * clamp((alpha + t.w) / alpha, 0.1, 2.0);
  let b = radius * clamp(alpha / (alpha + t.w), 0.1, 2.0);

  let cos_phi = cos(t.z);
  let sin_phi = sin(t.z);

  // R and S built so that (WGSL M)*v matches the original mul(M, v).
  let R = mat2x2<f32>(vec2<f32>(cos_phi, sin_phi), vec2<f32>(-sin_phi, cos_phi));
  let S = mat2x2<f32>(vec2<f32>(0.5 / a, 0.0), vec2<f32>(0.0, 0.5 / b));
  let SR = S * R;

  let max_x = min(48, i32(sqrt(a * a * cos_phi * cos_phi + b * b * sin_phi * sin_phi)));
  let max_y = min(48, i32(sqrt(a * a * sin_phi * sin_phi + b * b * cos_phi * cos_phi)));

  let zeta = 2.0 / radius;
  let zeroCross = u.zeroCrossing;
  let sinZeroCross = sin(zeroCross);
  let eta = (zeta + cos(zeroCross)) / (sinZeroCross * sinZeroCross);

  var m: array<vec4<f32>, 8>;
  var s: array<vec3<f32>, 8>;
  for (var k = 0; k < 8; k = k + 1) { m[k] = vec4<f32>(0.0); s[k] = vec3<f32>(0.0); }

  for (var y = -max_y; y <= max_y; y = y + 1) {
    for (var x = -max_x; x <= max_x; x = x + 1) {
      var v = SR * vec2<f32>(f32(x), f32(y));
      if (dot(v, v) <= 0.25) {
        let c = loadA(p + vec2<i32>(x, y), dim);
        var w: array<f32, 8>;
        var sum = 0.0;

        var vxx = zeta - eta * v.x * v.x;
        var vyy = zeta - eta * v.y * v.y;
        var z = max(0.0, v.y + vxx); w[0] = z * z; sum = sum + w[0];
        z = max(0.0, -v.x + vyy); w[2] = z * z; sum = sum + w[2];
        z = max(0.0, -v.y + vxx); w[4] = z * z; sum = sum + w[4];
        z = max(0.0, v.x + vyy); w[6] = z * z; sum = sum + w[6];
        v = sqrt(2.0) / 2.0 * vec2<f32>(v.x - v.y, v.x + v.y);
        vxx = zeta - eta * v.x * v.x;
        vyy = zeta - eta * v.y * v.y;
        z = max(0.0, v.y + vxx); w[1] = z * z; sum = sum + w[1];
        z = max(0.0, -v.x + vyy); w[3] = z * z; sum = sum + w[3];
        z = max(0.0, -v.y + vxx); w[5] = z * z; sum = sum + w[5];
        z = max(0.0, v.x + vyy); w[7] = z * z; sum = sum + w[7];

        let gw = exp(-3.125 * dot(v, v)) / max(sum, 1e-6);
        for (var k = 0; k < 8; k = k + 1) {
          let wk = w[k] * gw;
          m[k] = m[k] + vec4<f32>(c * wk, wk);
          s[k] = s[k] + c * c * wk;
        }
      }
    }
  }

  var output = vec4<f32>(0.0);
  for (var k = 0; k < 8; k = k + 1) {
    if (m[k].w > 1e-5) {
      let mrgb = m[k].rgb / m[k].w;
      let sk = abs(s[k] / m[k].w - mrgb * mrgb);
      let sigma2 = sk.r + sk.g + sk.b;
      let wgt = 1.0 / (1.0 + pow(abs(1000.0 * sigma2), 0.5 * u.q));
      output = output + vec4<f32>(mrgb * wgt, wgt);
    }
  }

  if (output.w < 1e-5) {
    return vec4<f32>(loadA(p, dim), 1.0);
  }
  return vec4<f32>(output.rgb / output.w, 1.0);
}
