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
@group(0) @binding(1) var<storage, read> transforms: array<mat4x4<f32>>;
@group(0) @binding(2) var<storage, read> finalTransform: array<mat4x4<f32>>;
@group(0) @binding(3) var<storage, read> occlusionGrid: array<f32>;
@group(0) @binding(4) var<uniform> u: Render;

struct VOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) worldPos: vec3<f32>,
  @location(1) oob: f32,
};

fn to1D(p: vec3<u32>) -> u32 {
  return p.x + p.y * u.gridSize + p.z * u.gridSize * u.gridSize;
}

fn getTrilinearVoxel(pos: vec3<f32>) -> f32 {
  var v = 0.0;
  let boundsExtent = u.gridBounds;

  if (abs(pos.x) <= boundsExtent && abs(pos.y) <= boundsExtent && abs(pos.z) <= boundsExtent) {
    var seedPos = pos + vec3<f32>(u.gridBounds * 0.5);
    seedPos = seedPos / u.gridBounds;
    seedPos = seedPos * f32(u.gridSize);

    let vi = vec3<u32>(floor(seedPos));
    let g = u.gridSize;
    var value = 0.0;

    for (var i = 0u; i < 2u; i = i + 1u) {
      let w1 = 1.0 - min(abs(seedPos.x - f32(vi.x + i)), f32(g));
      for (var j = 0u; j < 2u; j = j + 1u) {
        let w2 = 1.0 - min(abs(seedPos.y - f32(vi.y + j)), f32(g));
        for (var k = 0u; k < 2u; k = k + 1u) {
          let w3 = 1.0 - min(abs(seedPos.z - f32(vi.z + k)), f32(g));
          let c = vi + vec3<u32>(i, j, k);
          if (c.x < g && c.y < g && c.z < g) {
            value = value + w1 * w2 * w3 * occlusionGrid[to1D(c)];
          }
        }
      }
    }
    v = value;
  }
  return v;
}

@vertex
fn vs(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VOut {
  let basePos = positions[vid];
  let world = finalTransform[0] * (transforms[iid] * vec4<f32>(basePos, 1.0));

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
