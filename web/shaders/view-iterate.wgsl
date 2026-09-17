// Generate directly inside visible IFS address prefixes. A prefix is selected
// once per workgroup; no particle pays to sample a rejected branch.
struct Chaos {
  count: u32, particleCount: u32, width: u32, iters: u32,
  batchSeed: u32, advance: u32, leafCount: u32, _pad: u32,
  seeds: array<vec4<f32>, 32>,
};
struct Point { x: f32, y: f32, z: f32 };
struct Leaf { matrix: mat4x4<f32>, endGroup: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read_write> positions: array<Point>;
@group(0) @binding(1) var<storage, read> transforms: array<mat4x4<f32>>;
@group(0) @binding(2) var<uniform> u: Chaos;
@group(0) @binding(3) var<storage, read> leaves: array<Leaf>;
var<workgroup> prefix: mat4x4<f32>;

fn pcg(v: u32) -> u32 {
  let state = v * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

@compute @workgroup_size(64)
fn iterateView(@builtin(global_invocation_id) gid: vec3<u32>,
               @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  if (lid == 0u) {
    let group = wid.y * (u.width / 64u) + wid.x;
    var lo = 0u;
    var hi = u.leafCount;
    while (lo < hi) {
      let mid = (lo + hi) / 2u;
      if (group < leaves[mid].endGroup) { hi = mid; } else { lo = mid + 1u; }
    }
    prefix = leaves[min(lo, u.leafCount - 1u)].matrix;
  }
  workgroupBarrier();
  let idx = gid.y * u.width + gid.x;
  if (idx >= u.particleCount) { return; }
  var h = pcg(idx + u.batchSeed * u.particleCount);
  var pos = u.seeds[h % u.count].xyz;
  for (var k = 0u; k < u.iters; k++) {
    h = pcg(h);
    pos = (transforms[h % u.count] * vec4<f32>(pos, 1.0)).xyz;
  }
  pos = (prefix * vec4<f32>(pos, 1.0)).xyz;
  positions[idx] = Point(pos.x, pos.y, pos.z);
}
