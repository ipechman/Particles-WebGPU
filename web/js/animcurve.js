// animcurve.js
// A small re-implementation of Unity's AnimationCurve (cubic Hermite between
// keyframes) used by SetBlender for "lerp smoothing" of the morph pacing.
// Keyframes and PingPong wrap mode are taken from the original SampleScene.

export const WRAP_CLAMP = 0;
export const WRAP_PINGPONG = 2;

// Keyframes copied from SampleScene.unity (the morph easing curve).
const DEFAULT_KEYS = [
  { time: 0.0,        value: 0.0,       inSlope: 0.58119947, outSlope: 0.58119947 },
  { time: 0.11869369, value: 0.16560625, inSlope: 1.8865476, outSlope: 1.8865476 },
  { time: 0.59942836, value: 0.95321834, inSlope: 0.736799,  outSlope: 0.736799 },
  { time: 0.99731445, value: 1.0000011,  inSlope: 0.0,        outSlope: 0.0 },
];

export class AnimationCurve {
  constructor(keys = DEFAULT_KEYS, preWrap = WRAP_PINGPONG, postWrap = WRAP_PINGPONG) {
    this.keys = keys;
    this.preWrap = preWrap;
    this.postWrap = postWrap;
  }

  get start() { return this.keys[0].time; }
  get end() { return this.keys[this.keys.length - 1].time; }

  _wrap(t) {
    const a = this.start, b = this.end;
    const range = b - a;
    if (range <= 0) return a;
    if (t >= a && t <= b) return t;

    if (t < a) {
      if (this.preWrap === WRAP_PINGPONG) return this._pingpong(t - a, range) + a;
      return a; // clamp
    }
    if (this.postWrap === WRAP_PINGPONG) return this._pingpong(t - a, range) + a;
    return b; // clamp
  }

  _pingpong(x, range) {
    const m = ((x % (2 * range)) + 2 * range) % (2 * range);
    return m <= range ? m : 2 * range - m;
  }

  evaluate(t) {
    const keys = this.keys;
    t = this._wrap(t);

    if (t <= keys[0].time) return keys[0].value;
    if (t >= keys[keys.length - 1].time) return keys[keys.length - 1].value;

    let i = 0;
    while (i < keys.length - 1 && t > keys[i + 1].time) i++;
    const k0 = keys[i], k1 = keys[i + 1];

    const dt = k1.time - k0.time;
    if (dt <= 0) return k0.value;

    const u = (t - k0.time) / dt;
    const u2 = u * u, u3 = u2 * u;

    // Cubic Hermite basis.
    const h00 = 2 * u3 - 3 * u2 + 1;
    const h10 = u3 - 2 * u2 + u;
    const h01 = -2 * u3 + 3 * u2;
    const h11 = u3 - u2;

    return (
      h00 * k0.value +
      h10 * dt * k0.outSlope +
      h01 * k1.value +
      h11 * dt * k1.inSlope
    );
  }
}
