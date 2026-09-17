// Work sizing and wall-clock metrics, shared with regression tests.
export const PARTICLE_STRIDE = 12; // Point { x, y, z: f32 }, no precision reduction

export function dispatchShape(threadCount, workgroupSize, maxDimension) {
  const groups = Math.max(1, Math.ceil(threadCount / workgroupSize));
  const y = Math.ceil(groups / maxDimension);
  const x = Math.ceil(groups / y);
  if (y > maxDimension) throw new RangeError("Particle dispatch exceeds device limits");
  return { x, y, width: x * workgroupSize };
}

export function lightingSchedule(particles, budget) {
  const stride = Math.max(1, Math.ceil(particles / Math.max(1, budget)));
  return { count: Math.ceil(particles / stride), stride };
}

export class FrameRate {
  constructor() { this.seconds = 0; this.frames = 0; }
  update(seconds) {
    if (!Number.isFinite(seconds) || seconds <= 0) return null;
    this.seconds += seconds;
    this.frames++;
    if (this.seconds < 0.5) return null;
    const fps = this.frames / this.seconds;
    this.seconds = 0;
    this.frames = 0;
    return fps;
  }
}
