// render.wgsl
// Draws the attractor as instanced 1px points, shaded by the brute-force
// ambient-occlusion grid sampled trilinearly. Port of InstancedParticle.shader.
//
// Positions are read from a storage buffer indexed by vertex_index; each
// instance pre-applies one of the top-level transforms (instanceCount =
// transformCount), exactly like the original instanced indirect draw.

struct Render {
  viewProj: mat4x4<f32>,
  particleColor: vec4<f32>,
  occlusionColor: vec4<f32>,
  gridSize: u32,
  transformCount: u32,
  gridBounds: f32,
  occlusionMultiplier: f32,
  occlusionAttenuation: f32,
  _p0: f32,
  _p1: f32,
  _p2: f32,
};

@group(0) @binding(0) var<storage, read> positions: array<vec3<f32>>;
// combined[i] = finalTransform * transforms[i], premultiplied by combine.wgsl
// so each of the ~25M instanced vertices applies a single matrix.
@group(0) @binding(1) var<storage, read> combined: array<mat4x4<f32>>;
@group(0) @binding(3) var occlusionTex: texture_3d<f32>;
@group(0) @binding(4) var<uniform> u: Render;
@group(0) @binding(5) var occlusionSampler: sampler;

struct VOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) worldPos: vec3<f32>,
  @location(1) oob: f32,
};

fn getTrilinearVoxel(pos: vec3<f32>) -> f32 {
  var v = 0.0;
  let boundsExtent = u.gridBounds;

  if (abs(pos.x) <= boundsExtent && abs(pos.y) <= boundsExtent && abs(pos.z) <= boundsExtent) {
    // Normalized grid coordinate, shifted half a texel so the hardware
    // trilinear filter interpolates on the voxel lattice exactly like the
    // manual 8-tap loop this replaces (voxel i's value lives at texel
    // center (i + 0.5) / gridSize).
    let uvw = (pos + vec3<f32>(u.gridBounds * 0.5)) / u.gridBounds
            + vec3<f32>(0.5 / f32(u.gridSize));
    v = textureSampleLevel(occlusionTex, occlusionSampler, uvw, 0.0).r;
  }
  return v;
}

@vertex
fn vs(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VOut {
  let basePos = positions[vid];
  let world = combined[iid] * vec4<f32>(basePos, 1.0);

  let halfBounds = u.gridBounds * 0.5;
  var oob = 0.0;
  if (any(world.xyz > vec3<f32>(halfBounds)) || any(world.xyz < vec3<f32>(-halfBounds))) {
    oob = 1.0;
  }

  var o: VOut;
  o.pos = u.viewProj * vec4<f32>(world.xyz, 1.0);
  o.worldPos = world.xyz;
  o.oob = oob;
  return o;
}

@fragment
fn fs(i: VOut) -> @location(0) vec4<f32> {
  if (i.oob > 0.5) { discard; }

  var occlusion = getTrilinearVoxel(i.worldPos);
  occlusion = pow(clamp(occlusion * u.occlusionMultiplier, 0.0, 1.0), u.occlusionAttenuation);

  let col = mix(u.occlusionColor.rgb, u.particleColor.rgb, occlusion);
  return vec4<f32>(col, 1.0);
}
