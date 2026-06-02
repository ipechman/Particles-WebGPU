// reduce.wgsl
// Parallel reduction of the low-detail point cloud to its min / max / sum.
// A single workgroup does a grid-stride load followed by a shared-memory tree
// reduction (sequential addressing). This yields the same three quantities the
// original ParallelReduce.compute computes with its separate min/max/add
// reducers, condensed into one pass since the LOD set is small.

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

@group(0) @binding(0) var<storage, read> input: array<vec3<f32>>;
@group(0) @binding(1) var<storage, read_write> result: array<Bounds>;
@group(0) @binding(2) var<uniform> u: Reduce;

const WG: u32 = 256u;

var<workgroup> sMin: array<vec3<f32>, 256>;
var<workgroup> sMax: array<vec3<f32>, 256>;
var<workgroup> sSum: array<vec3<f32>, 256>;

@compute @workgroup_size(256)
fn reduce(@builtin(local_invocation_id) lid: vec3<u32>) {
  let BIG = 1.0e30;
  var mn = vec3<f32>(BIG, BIG, BIG);
  var mx = vec3<f32>(-BIG, -BIG, -BIG);
  var sm = vec3<f32>(0.0, 0.0, 0.0);

  var i = lid.x;
  loop {
    if (i >= u.inputSize) { break; }
    let v = input[i];
    mn = min(mn, v);
    mx = max(mx, v);
    sm = sm + v;
    i = i + WG;
  }

  sMin[lid.x] = mn;
  sMax[lid.x] = mx;
  sSum[lid.x] = sm;
  workgroupBarrier();

  var s = WG / 2u;
  loop {
    if (s == 0u) { break; }
    if (lid.x < s) {
      sMin[lid.x] = min(sMin[lid.x], sMin[lid.x + s]);
      sMax[lid.x] = max(sMax[lid.x], sMax[lid.x + s]);
      sSum[lid.x] = sSum[lid.x] + sSum[lid.x + s];
    }
    workgroupBarrier();
    s = s >> 1u;
  }

  if (lid.x == 0u) {
    result[0].mn = sMin[0];
    result[0].mx = sMax[0];
    result[0].sm = sSum[0];
  }
}
