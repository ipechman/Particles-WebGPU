// math.js
// Faithful ports of the linear-algebra used by the original Unity project.
// All 4x4 matrices are stored COLUMN-MAJOR (Float32Array length 16) so they
// can be uploaded directly into WGSL `mat4x4<f32>` (column-major) buffers.
//
// A matrix M transforms a point as  M * vec4(p, 1).  For an affine transform
// the translation therefore lives in the 4th column (indices 12,13,14).

export const DEG2RAD = Math.PI / 180.0;

// ----------------------------------------------------------------------------
// vec3 helpers (plain length-3 arrays)
// ----------------------------------------------------------------------------
export const v3 = {
  add: (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
  sub: (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]],
  mul: (a, b) => [a[0] * b[0], a[1] * b[1], a[2] * b[2]], // component-wise (Vector3.Scale)
  scale: (a, s) => [a[0] * s, a[1] * s, a[2] * s],
  // Unity's LerpUnclamped: a + (b-a)*t, no clamping of t.
  lerpUnclamped: (a, b, t) => [
    a[0] + (b[0] - a[0]) * t,
    a[1] + (b[1] - a[1]) * t,
    a[2] + (b[2] - a[2]) * t,
  ],
  clone: (a) => [a[0], a[1], a[2]],
  length: (a) => Math.hypot(a[0], a[1], a[2]),
  normalize: (a) => {
    const l = Math.hypot(a[0], a[1], a[2]) || 1;
    return [a[0] / l, a[1] / l, a[2] / l];
  },
  cross: (a, b) => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ],
  dot: (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
};

// ----------------------------------------------------------------------------
// mat4 (column-major Float32Array(16))
// ----------------------------------------------------------------------------
export const mat4 = {
  identity() {
    const m = new Float32Array(16);
    m[0] = m[5] = m[10] = m[15] = 1;
    return m;
  },

  // Build from logical ROWS (matches Unity's Matrix4x4.SetRow usage exactly).
  // r0..r3 are length-4 arrays.  Stored column-major: out[c*4 + r] = rowR[c].
  fromRows(r0, r1, r2, r3) {
    const m = new Float32Array(16);
    const rows = [r0, r1, r2, r3];
    for (let r = 0; r < 4; r++)
      for (let c = 0; c < 4; c++) m[c * 4 + r] = rows[r][c];
    return m;
  },

  // C = A * B  (column-major).  C[c*4+r] = sum_k A[k*4+r] * B[c*4+k]
  multiply(a, b) {
    const m = new Float32Array(16);
    for (let c = 0; c < 4; c++) {
      for (let r = 0; r < 4; r++) {
        let s = 0;
        for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
        m[c * 4 + r] = s;
      }
    }
    return m;
  },

  // ---- The exact constructors from AffineTransformations.cs --------------
  scale(s) {
    return mat4.fromRows(
      [s[0], 0, 0, 0],
      [0, s[1], 0, 0],
      [0, 0, s[2], 0],
      [0, 0, 0, 1]
    );
  },
  shearX(s) {
    return mat4.fromRows(
      [1, s[1], s[2], 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1]
    );
  },
  shearY(s) {
    return mat4.fromRows(
      [1, 0, 0, 0],
      [s[0], 1, s[2], 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1]
    );
  },
  shearZ(s) {
    return mat4.fromRows(
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [s[0], s[1], 1, 0],
      [0, 0, 0, 1]
    );
  },
  translate(t) {
    return mat4.fromRows(
      [1, 0, 0, t[0]],
      [0, 1, 0, t[1]],
      [0, 0, 1, t[2]],
      [0, 0, 0, 1]
    );
  },

  // Rotation matrix from a quaternion (x,y,z,w). Standard, row form below.
  fromQuat(q) {
    const [x, y, z, w] = q;
    const xx = x * x, yy = y * y, zz = z * z;
    const xy = x * y, xz = x * z, yz = y * z;
    const wx = w * x, wy = w * y, wz = w * z;
    return mat4.fromRows(
      [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy), 0],
      [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx), 0],
      [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy), 0],
      [0, 0, 0, 1]
    );
  },

  // ---- Camera matrices (column-major, WebGPU clip space z in [0,1]) ------
  perspective(fovY, aspect, near, far) {
    const f = 1.0 / Math.tan(fovY / 2);
    const nf = 1.0 / (near - far);
    const m = new Float32Array(16);
    m[0] = f / aspect;
    m[5] = f;
    m[10] = far * nf;
    m[11] = -1;
    m[14] = near * far * nf;
    return m;
  },

  lookAt(eye, center, up) {
    const z = v3.normalize(v3.sub(eye, center));
    const x = v3.normalize(v3.cross(up, z));
    const y = v3.cross(z, x);
    return mat4.fromRows(
      [x[0], x[1], x[2], -v3.dot(x, eye)],
      [y[0], y[1], y[2], -v3.dot(y, eye)],
      [z[0], z[1], z[2], -v3.dot(z, eye)],
      [0, 0, 0, 1]
    );
  },
};

// ----------------------------------------------------------------------------
// Quaternion (x,y,z,w).  Matches Unity's Quaternion conventions.
// ----------------------------------------------------------------------------
export const quat = {
  identity: () => [0, 0, 0, 1],

  // Hamilton product a * b
  mul(a, b) {
    const [ax, ay, az, aw] = a;
    const [bx, by, bz, bw] = b;
    return [
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw,
      aw * bw - ax * bx - ay * by - az * bz,
    ];
  },

  axisAngle(axis, deg) {
    const h = deg * DEG2RAD * 0.5;
    const s = Math.sin(h);
    return [axis[0] * s, axis[1] * s, axis[2] * s, Math.cos(h)];
  },

  // Unity's Quaternion.Euler(x,y,z): applies Z, then X, then Y => q = qy*qx*qz
  fromEuler(e) {
    const qx = quat.axisAngle([1, 0, 0], e[0]);
    const qy = quat.axisAngle([0, 1, 0], e[1]);
    const qz = quat.axisAngle([0, 0, 1], e[2]);
    return quat.mul(quat.mul(qy, qx), qz);
  },

  normalize(q) {
    const l = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
    return [q[0] / l, q[1] / l, q[2] / l, q[3] / l];
  },

  // Spherical interpolation (unclamped t allowed, shortest arc).
  slerp(a, b, t) {
    let [ax, ay, az, aw] = a;
    let [bx, by, bz, bw] = b;
    let cosom = ax * bx + ay * by + az * bz + aw * bw;
    if (cosom < 0) {
      cosom = -cosom;
      bx = -bx; by = -by; bz = -bz; bw = -bw;
    }
    let scale0, scale1;
    if (1.0 - cosom > 1e-6) {
      const omega = Math.acos(cosom);
      const sinom = Math.sin(omega);
      scale0 = Math.sin((1.0 - t) * omega) / sinom;
      scale1 = Math.sin(t * omega) / sinom;
    } else {
      // Nearly identical: linear interpolation.
      scale0 = 1.0 - t;
      scale1 = t;
    }
    return [
      scale0 * ax + scale1 * bx,
      scale0 * ay + scale1 * by,
      scale0 * az + scale1 * bz,
      scale0 * aw + scale1 * bw,
    ];
  },
};
