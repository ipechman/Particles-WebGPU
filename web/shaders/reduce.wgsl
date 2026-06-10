// reduce.wgsl
// Two-stage parallel reduction of the particle cloud to its min / max / sum.
// Stage 1 (reducePoints): PARTIALS workgroups grid-stride the positions and
// each writes one partial Bounds. Stage 2 (reduceBounds): a single workgroup
// folds the partials into result[0].
//
// The reduction runs on the exact cloud that will be rendered (it replaced
// the old origin-seeded low-detail "prediction" chain, which systematically
// under-estimated the attractor's true extent and let converged points fall
// outside the fitted box and clip).

struct Reduce {
  inputSize: u32,
  _p0: u32,
  _p1: u32,
  _p2: u32,
};

struct Bounds {
  mn: vec3<f32>,
  mx: vec3<f32>,
  sm: vec3<f32>,
};

@group(0) @binding(0) var<storage, read> points: array<vec3<f32>>;
@group(0) @binding(1) var<storage, read_write> outBounds: array<Bounds>;
@group(0) @binding(2) var<uniform> u: Reduce;
@group(0) @binding(3) var<storage, read> inBounds: array<Bounds>;

const WG: u32 = 256u;

var<workgroup> sMin: array<vec3<f32>, 256>;
var<workgroup> sMax: array<vec3<f32>, 256>;
var<workgroup> sSum: array<vec3<f32>, 256>;

// Shared-memory tree reduction (sequential addressing); result lands in
// slot 0 of each array.
fn fold(lid: u32) {
  var s = WG / 2u;
  loop {
    if (s == 0u) { break; }
    if (lid < s) {
      sMin[lid] = min(sMin[lid], sMin[lid + s]);
      sMax[lid] = max(sMax[lid], sMax[lid + s]);
      sSum[lid] = sSum[lid] + sSum[lid + s];
    }
    workgroupBarrier();
    s = s >> 1u;
  }
}

@compute @workgroup_size(256)
fn reducePoints(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let BIG = 1.0e30;
  var mn = vec3<f32>(BIG, BIG, BIG);
  var mx = vec3<f32>(-BIG, -BIG, -BIG);
  var sm = vec3<f32>(0.0, 0.0, 0.0);

  var i = wid.x * WG + lid.x;
  let step = nwg.x * WG;
  loop {
    if (i >= u.inputSize) { break; }
    let v = points[i];
    mn = min(mn, v);
    mx = max(mx, v);
    sm = sm + v;
    i = i + step;
  }

  sMin[lid.x] = mn;
  sMax[lid.x] = mx;
  sSum[lid.x] = sm;
  workgroupBarrier();
  fold(lid.x);

  if (lid.x == 0u) {
    outBounds[wid.x].mn = sMin[0];
    outBounds[wid.x].mx = sMax[0];
    outBounds[wid.x].sm = sSum[0];
  }
}

@compute @workgroup_size(256)
fn reduceBounds(@builtin(local_invocation_id) lid: vec3<u32>) {
  let BIG = 1.0e30;
  var mn = vec3<f32>(BIG, BIG, BIG);
  var mx = vec3<f32>(-BIG, -BIG, -BIG);
  var sm = vec3<f32>(0.0, 0.0, 0.0);

  var i = lid.x;
  loop {
    if (i >= u.inputSize) { break; }
    mn = min(mn, inBounds[i].mn);
    mx = max(mx, inBounds[i].mx);
    sm = sm + inBounds[i].sm;
    i = i + WG;
  }

  sMin[lid.x] = mn;
  sMax[lid.x] = mx;
  sSum[lid.x] = sm;
  workgroupBarrier();
  fold(lid.x);

  if (lid.x == 0u) {
    outBounds[0].mn = sMin[0];
    outBounds[0].mx = sMax[0];
    outBounds[0].sm = sSum[0];
  }
}
