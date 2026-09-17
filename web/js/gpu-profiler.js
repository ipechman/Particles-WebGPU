// Optional GPU pass timings. Readbacks run asynchronously in a fixed-size ring;
// a busy ring skips samples instead of delaying rendering or allocating buffers.
export class GPUProfiler {
  constructor(device, { enabled = true, sampleEvery = 30, ringSize = 3, maxPasses = 64 } = {}) {
    for (const [name, value, max] of [
      ["sampleEvery", sampleEvery, Number.MAX_SAFE_INTEGER],
      ["ringSize", ringSize, 8],
      ["maxPasses", maxPasses, 256],
    ]) {
      if (!Number.isSafeInteger(value) || value < 1 || value > max) {
        throw new RangeError(`${name} must be an integer from 1 to ${max}`);
      }
    }
    this.supported = device.features?.has("timestamp-query") ?? false;
    this.enabled = Boolean(enabled && this.supported);
    this.sampleEvery = sampleEvery;
    this.maxPasses = maxPasses;
    this.skippedSamples = 0;
    this.failedSamples = 0;
    this._frame = -1;
    this._latest = null;
    this._active = null;
    this._disposed = false;
    this._slots = [];

    if (!this.enabled) return;
    for (let i = 0; i < ringSize; i++) {
      this._slots.push({
        querySet: device.createQuerySet({ type: "timestamp", count: maxPasses * 2 }),
        resolve: device.createBuffer({
          label: `GPU timings resolve ${i}`, size: maxPasses * 16,
          usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        }),
        readback: device.createBuffer({
          label: `GPU timings readback ${i}`, size: maxPasses * 16,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        }),
        state: "free", labels: [], frame: -1, droppedPasses: 0,
      });
    }
    // Destroying a mapped or pending buffer cancels mapping. _readSlot catches
    // that rejection, including when the device is lost during a readback.
    device.lost?.then(() => this.dispose(), () => this.dispose());
  }

  // A frozen snapshot of the newest completed sample, or null before one arrives.
  // totalMs sums instrumented passes; it excludes copies and time between passes.
  get latest() { return this._latest; }

  beginFrame() {
    this._frame++;
    if (!this.enabled || this._disposed || this._active || this._frame % this.sampleEvery !== 0) {
      return false;
    }
    const slot = this._slots.find((candidate) => candidate.state === "free");
    if (!slot) {
      this.skippedSamples++;
      return false;
    }
    slot.state = "recording";
    slot.labels = [];
    slot.frame = this._frame;
    slot.droppedPasses = 0;
    this._active = slot;
    return true;
  }

  // Add this descriptor as timestampWrites when beginning a compute/render pass.
  // Repeated labels (e.g. bloom blur) are summed in the published snapshot.
  stamp(label) {
    const slot = this._active;
    if (!slot) return undefined;
    if (slot.labels.length >= this.maxPasses) {
      slot.droppedPasses++;
      return undefined;
    }
    const index = slot.labels.length * 2;
    slot.labels.push(String(label));
    return {
      querySet: slot.querySet,
      beginningOfPassWriteIndex: index,
      endOfPassWriteIndex: index + 1,
    };
  }

  // Call before encoder.finish(), after all timed passes have ended.
  finishFrame(encoder) {
    const slot = this._active;
    this._active = null;
    if (!slot) return;
    const queryCount = slot.labels.length * 2;
    if (!queryCount) {
      slot.state = "free";
      return;
    }
    encoder.resolveQuerySet(slot.querySet, 0, queryCount, slot.resolve, 0);
    encoder.copyBufferToBuffer(slot.resolve, 0, slot.readback, 0, queryCount * 8);
    slot.state = "ready";
  }

  // Call immediately after queue.submit(). No main-loop await or queue-wide wait.
  afterSubmit() {
    if (this._disposed) return;
    for (const slot of this._slots) {
      if (slot.state !== "ready") continue;
      slot.state = "mapping"; // reserve before starting mapAsync
      void this._readSlot(slot);
    }
  }

  async _readSlot(slot) {
    try {
      const bytes = slot.labels.length * 16;
      await slot.readback.mapAsync(GPUMapMode.READ, 0, bytes);
      if (this._disposed) return;
      const ticks = new BigUint64Array(slot.readback.getMappedRange(0, bytes));
      const passes = new Map();
      let totalMs = 0;
      for (let i = 0; i < slot.labels.length; i++) {
        const start = ticks[i * 2], end = ticks[i * 2 + 1];
        // Discard counter rollover/reset instead of publishing negative timings.
        if (end < start) continue;
        const ms = Number(end - start) / 1e6; // WebGPU timestamps are nanoseconds.
        if (!Number.isFinite(ms) || ms < 0) continue;
        const label = slot.labels[i];
        passes.set(label, (passes.get(label) ?? 0) + ms);
        totalMs += ms;
      }
      // Readbacks can finish out of order; keep the newest frame's timings.
      if (!this._latest || slot.frame > this._latest.frame) {
        this._latest = Object.freeze({
          frame: slot.frame,
          totalMs,
          passes: Object.freeze(Object.fromEntries(passes)),
          droppedPasses: slot.droppedPasses,
        });
      }
    } catch {
      // Mapping can fail on device loss or cancellation. Profiling must never
      // break rendering or leave an unhandled promise rejection behind.
      if (!this._disposed) this.failedSamples++;
    } finally {
      try { slot.readback.unmap(); } catch { /* lost/destroyed buffer */ }
      if (!this._disposed) slot.state = "free";
    }
  }

  dispose() {
    if (this._disposed) return;
    this._disposed = true;
    this.enabled = false;
    this._active = null;
    for (const slot of this._slots) {
      slot.state = "disposed";
      for (const resource of [slot.querySet, slot.resolve, slot.readback]) {
        try { resource.destroy(); } catch { /* already lost/destroyed */ }
      }
    }
  }
}
