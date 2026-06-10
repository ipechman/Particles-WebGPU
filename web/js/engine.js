// engine.js
// WebGPU engine: owns the device, buffers and pipelines, and runs the per-frame
// pipeline that mirrors IteratedFunctionSystem.Update:
//   predict final (auto-fit) transform -> iterate attractor -> voxelize +
//   ambient occlusion -> render instanced points.

import { mat4 } from "./math.js";
import { MAX_TRANSFORMS } from "./blender.js";

const WG = 64;                 // generic workgroup size
const REDUCE_PARTIALS = 1024;  // stage-1 workgroups of the bounds reduction
const SCENE_FORMAT = "rgba16float"; // HDR offscreen target for post-processing

const SHADER_FILES = {
  iterate: "shaders/iterate.wgsl",
  reduce: "shaders/reduce.wgsl",
  fit: "shaders/fit.wgsl",
  combine: "shaders/combine.wgsl",
  voxelize: "shaders/voxelize.wgsl",
  render: "shaders/render.wgsl",
  post: "shaders/post.wgsl",
  present: "shaders/present.wgsl",
  kuwahara: "shaders/kuwahara.wgsl",
};

export class Engine {
  constructor(canvas) {
    this.canvas = canvas;

    // ---- tunable configuration (defaults mirror the original project) ----
    this.particlesPerBatch = 1 << 23;     // 8,388,608 (8.4M)
    this.voxelGridDim = 128;              // grid resolution
    this.voxelBounds = 3.0;               // world-space box size
    this.scalePadding = 0.5;              // flagship fit padding

    this.particleColor = [0.93, 0.94, 0.96]; // neutral near-white highlight
    this.occlusionColor = [0.103773594, 0.014195448, 0.014195448];
    this.occlusionMultiplier = 1.0;
    this.occlusionAttenuation = 1.0;
    this.backgroundColor = [0.0, 0.0, 0.0];

    // ---- post-processing ----
    // Anisotropic Kuwahara filter (Acerola). Painterly, flow-aligned.
    this.kuwaharaEnabled = false;
    this.kuwaharaKernelSize = 4;    // window size; radius = kernelSize / 2
    this.kuwaharaSharpness = 5.0;   // _Q
    this.kuwaharaAlpha = 1.0;
    this.kuwaharaZeroCrossing = 0.58;
    this.kuwaharaBlurRadius = 2;
    // Retuned for the denser accumulated image: only genuinely bright areas
    // should glow, not the whole fractal. Note the prefilter's smoothstep
    // knee spans [threshold, threshold + 0.25] and lit pixels rarely exceed
    // ~0.8, so thresholds much above 0.4 kill the bloom entirely.
    this.bloomIntensity = 2.2;      // 0 disables bloom
    this.bloomThreshold = 0.4;
    this.bloomSpread = 2.0;
    this.bloomIterations = 3;       // blur passes -> width/softness of the glow

    // sizes currently realized in GPU buffers (for change detection)
    this._sizes = { particles: -1, voxels: -1 };

    // Cache for skipping the attractor/voxel/occlusion recompute when the
    // transforms (and the params they depend on) haven't changed since the last
    // computed frame. The render + post passes still run every frame.
    // Progressive accumulation: while the shape, camera and render params are
    // all unchanged, independent chaos batches are baked into the hidden back
    // scene buffer (one per frame), multiplying effective detail. The batch
    // budget scales inversely with the particle count so the resting image
    // always converges toward ~accumTargetPoints effective points (lower
    // particle counts buffer longer: 8 batches at 8.4M, 256 at 262K). The
    // front/back buffers swap at geometric checkpoints (4, 16, 64, 256
    // batches), each presenting a complete stable set. Once the budget is
    // reached the points pass is skipped entirely, so a fully idle frame
    // costs only the post chain.
    this.accumTargetPoints = 1 << 26; // ~67M effective points at rest
    this._accumCount = 0;      // batches baked into the current cycle
    this._accumCheckpoint = 4; // batch count at which front/back swap next
    this._backBatches = 0;     // batches in the back buffer (0 = needs seeding)
    this._batchSeed = 0;       // chaos batch the positions buffer holds
    this._cachedRenderU = null; // last drawn uRender bytes (camera, colors, ...)
    this._cachedBg = [NaN, NaN, NaN];
    this._cachedFbW = -1;
    this._cachedFbH = -1;

    this._cachedTransformData = null;
    this._cachedTransformCount = -1;
    this._cachedScalePadding = -1;
  }

  get voxelCount() {
    const d = this.voxelGridDim;
    return d * d * d;
  }

  async init() {
    if (!navigator.gpu) throw new Error("WebGPU is not available in this browser.");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("No suitable GPU adapter found.");

    // Request the largest storage-buffer / buffer sizes the adapter allows so
    // we can hold very large particle clouds (up to ~100M points) in one buffer.
    const lim = adapter.limits;
    this.device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize,
        maxBufferSize: lim.maxBufferSize,
      },
    });
    this.device.lost.then((info) => {
      if (info.reason !== "destroyed") console.error("WebGPU device lost:", info.message);
    });

    this.maxComputeDim = this.device.limits.maxComputeWorkgroupsPerDimension;
    // Largest particle count that fits a single 16-byte-per-point buffer.
    this.maxParticles = Math.floor(
      Math.min(this.device.limits.maxStorageBufferBindingSize, this.device.limits.maxBufferSize) / 16
    );

    this.ctx = this.canvas.getContext("webgpu");
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.ctx.configure({ device: this.device, format: this.format, alphaMode: "opaque" });

    await this._loadShaders();
    this._createLayouts();
    this._createPipelines();
    this._createStaticBuffers();
    this._rebuild();

    this.transformData = new Float32Array(MAX_TRANSFORMS * 16);
  }

  async _loadShaders() {
    const entries = await Promise.all(
      Object.entries(SHADER_FILES).map(async ([k, path]) => {
        const res = await fetch(path, { cache: "no-cache" });
        if (!res.ok) throw new Error(`Failed to load shader ${path}: ${res.status}`);
        return [k, await res.text()];
      })
    );
    this.modules = {};
    for (const [k, code] of entries) {
      this.modules[k] = this.device.createShaderModule({ code, label: k });
    }
  }

  // ---- bind group layouts -------------------------------------------------
  _createLayouts() {
    const d = this.device;
    const C = GPUShaderStage.COMPUTE;
    const VF = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT;

    const buf = (binding, vis, type, dyn = false) => ({
      binding, visibility: vis, buffer: { type, hasDynamicOffset: dyn },
    });

    this.bgl = {
      iter: d.createBindGroupLayout({
        entries: [
          buf(0, C, "storage"),
          buf(1, C, "read-only-storage"),
          buf(2, C, "uniform"),
        ],
      }),
      reduce: d.createBindGroupLayout({
        entries: [
          buf(0, C, "read-only-storage"),
          buf(1, C, "storage"),
          buf(2, C, "uniform"),
          buf(3, C, "read-only-storage"),
        ],
      }),
      fit: d.createBindGroupLayout({
        entries: [buf(0, C, "read-only-storage"), buf(1, C, "storage"), buf(2, C, "uniform")],
      }),
      combine: d.createBindGroupLayout({
        entries: [
          buf(0, C, "read-only-storage"),
          buf(1, C, "read-only-storage"),
          buf(2, C, "storage"),
          buf(3, C, "uniform"),
        ],
      }),
      grid: d.createBindGroupLayout({
        entries: [
          buf(0, C, "read-only-storage"),
          buf(1, C, "read-only-storage"),
          buf(2, C, "read-only-storage"),
          buf(3, C, "storage"),
          { binding: 4, visibility: C, storageTexture: { access: "write-only", format: "rgba16float", viewDimension: "3d" } },
          buf(5, C, "uniform"),
        ],
      }),
      render: d.createBindGroupLayout({
        entries: [
          buf(0, VF, "read-only-storage"),
          buf(1, VF, "read-only-storage"),
          buf(2, VF, "read-only-storage"),
          { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float", viewDimension: "3d" } },
          buf(4, VF, "uniform"),
          { binding: 5, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
        ],
      }),
      // Post passes: sampler + input texture + params.
      post: d.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
          { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
          { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        ],
      }),
      // Present: sampler + scene texture + bloom texture + params.
      present: d.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
          { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
          { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
          { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        ],
      }),
      // Anisotropic Kuwahara: two input textures (read via textureLoad) + params.
      kuw: d.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
          { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
          { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        ],
      }),
    };

    this.pl = {
      iter: d.createPipelineLayout({ bindGroupLayouts: [this.bgl.iter] }),
      reduce: d.createPipelineLayout({ bindGroupLayouts: [this.bgl.reduce] }),
      fit: d.createPipelineLayout({ bindGroupLayouts: [this.bgl.fit] }),
      combine: d.createPipelineLayout({ bindGroupLayouts: [this.bgl.combine] }),
      grid: d.createPipelineLayout({ bindGroupLayouts: [this.bgl.grid] }),
      render: d.createPipelineLayout({ bindGroupLayouts: [this.bgl.render] }),
      post: d.createPipelineLayout({ bindGroupLayouts: [this.bgl.post] }),
      present: d.createPipelineLayout({ bindGroupLayouts: [this.bgl.present] }),
      kuw: d.createPipelineLayout({ bindGroupLayouts: [this.bgl.kuw] }),
    };
  }

  _createPipelines() {
    const d = this.device;
    const comp = (mod, entry, layout) =>
      d.createComputePipeline({ layout, compute: { module: this.modules[mod], entryPoint: entry } });

    this.pipe = {
      iterate: comp("iterate", "iterate", this.pl.iter),
      reducePoints: comp("reduce", "reducePoints", this.pl.reduce),
      reduceBounds: comp("reduce", "reduceBounds", this.pl.reduce),
      fit: comp("fit", "fit", this.pl.fit),
      combine: comp("combine", "combine", this.pl.combine),
      clearGrids: comp("voxelize", "clearGrids", this.pl.grid),
      voxelize: comp("voxelize", "voxelize", this.pl.grid),
      occlusion: comp("voxelize", "occlusion", this.pl.grid),
    };

    this.pipe.render = d.createRenderPipeline({
      layout: this.pl.render,
      vertex: { module: this.modules.render, entryPoint: "vs" },
      fragment: {
        module: this.modules.render,
        entryPoint: "fs",
        targets: [{ format: SCENE_FORMAT }],
      },
      primitive: { topology: "point-list" },
      depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
    });

    // Post-processing pipelines (full-screen triangle).
    const post = (entry, format) =>
      d.createRenderPipeline({
        layout: this.pl.post,
        vertex: { module: this.modules.post, entryPoint: "vsFull" },
        fragment: { module: this.modules.post, entryPoint: entry, targets: [{ format }] },
        primitive: { topology: "triangle-list" },
      });

    this.pipe.prefilter = post("fsPrefilter", SCENE_FORMAT);
    this.pipe.blur = post("fsBlur", SCENE_FORMAT);

    // Anisotropic Kuwahara passes (own module + layout).
    const kuw = (entry) =>
      d.createRenderPipeline({
        layout: this.pl.kuw,
        vertex: { module: this.modules.kuwahara, entryPoint: "vsFull" },
        fragment: { module: this.modules.kuwahara, entryPoint: entry, targets: [{ format: SCENE_FORMAT }] },
        primitive: { topology: "triangle-list" },
      });
    this.pipe.kuwStructure = kuw("fsStructureTensor");
    this.pipe.kuwBlurH = kuw("fsTensorBlurH");
    this.pipe.kuwAniso = kuw("fsAnisotropy");
    this.pipe.kuwFilter = kuw("fsKuwahara");

    this.pipe.present = d.createRenderPipeline({
      layout: this.pl.present,
      vertex: { module: this.modules.present, entryPoint: "vsFull" },
      fragment: { module: this.modules.present, entryPoint: "fsPresent", targets: [{ format: this.format }] },
      primitive: { topology: "triangle-list" },
    });
  }

  // ---- buffers that never change size ------------------------------------
  _createStaticBuffers() {
    const d = this.device;
    const U = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
    const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;

    this.transformsBuf = d.createBuffer({ size: MAX_TRANSFORMS * 64, usage: S, label: "transforms" });
    this.finalTransformBuf = d.createBuffer({ size: 64, usage: S, label: "finalTransform" });
    this.reduceResultBuf = d.createBuffer({ size: 64, usage: GPUBufferUsage.STORAGE, label: "reduceResult" });
    // combined[i] = finalTransform * transforms[i], premultiplied on the GPU
    // after the fit pass (combine.wgsl) so render/voxelize apply one matrix.
    this.combinedBuf = d.createBuffer({ size: MAX_TRANSFORMS * 64, usage: GPUBufferUsage.STORAGE, label: "combined" });
    this.uCombine = d.createBuffer({ size: 16, usage: U, label: "uCombine" });

    // Chaos-game uniform: 32-byte header + per-transform fixed-point seeds.
    this.uChaos = d.createBuffer({ size: 32 + MAX_TRANSFORMS * 16, usage: U, label: "uChaos" });
    this.uChaosCPU = new ArrayBuffer(32 + MAX_TRANSFORMS * 16);

    // Two-stage bounds reduction over the real cloud: per-workgroup partials,
    // then a single-workgroup fold (see reduce.wgsl).
    this.partialsBuf = d.createBuffer({ size: REDUCE_PARTIALS * 48, usage: GPUBufferUsage.STORAGE, label: "reducePartials" });
    this.uReduce1 = d.createBuffer({ size: 16, usage: U, label: "uReduce1" });
    this.uReduce2 = d.createBuffer({ size: 16, usage: U, label: "uReduce2" });
    d.queue.writeBuffer(this.uReduce2, 0, new Uint32Array([REDUCE_PARTIALS, 0, 0, 0]));

    // Small single-use uniforms.
    this.uFit = d.createBuffer({ size: 16, usage: U, label: "uFit" });
    this.uGrid = d.createBuffer({ size: 32, usage: U, label: "uGrid" });
    this.uRender = d.createBuffer({ size: 128, usage: U, label: "uRender" });

    // Post-processing uniforms (PostParams / PresentParams are 32 bytes each).
    this.uKuw = d.createBuffer({ size: 32, usage: U, label: "uKuw" });
    this.uPre = d.createBuffer({ size: 32, usage: U, label: "uPre" });
    this.uBlurH = d.createBuffer({ size: 32, usage: U, label: "uBlurH" });
    this.uBlurV = d.createBuffer({ size: 32, usage: U, label: "uBlurV" });
    this.uPresent = d.createBuffer({ size: 32, usage: U, label: "uPresent" });

    this.sampler = d.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
    });

    // Init final transform to identity (used before the first fit dispatch).
    d.queue.writeBuffer(this.finalTransformBuf, 0, mat4.identity());

    this.bgFit = d.createBindGroup({
      layout: this.bgl.fit,
      entries: [
        { binding: 0, resource: { buffer: this.reduceResultBuf } },
        { binding: 1, resource: { buffer: this.finalTransformBuf } },
        { binding: 2, resource: { buffer: this.uFit } },
      ],
    });

    this.bgCombine = d.createBindGroup({
      layout: this.bgl.combine,
      entries: [
        { binding: 0, resource: { buffer: this.transformsBuf } },
        { binding: 1, resource: { buffer: this.finalTransformBuf } },
        { binding: 2, resource: { buffer: this.combinedBuf } },
        { binding: 3, resource: { buffer: this.uCombine } },
      ],
    });

    // Stage 2 of the bounds reduction: partials -> result[0]. (Binding 0 is
    // unused by the reduceBounds entry; the partials buffer stands in.)
    this.bgReduce2 = d.createBindGroup({
      layout: this.bgl.reduce,
      entries: [
        { binding: 0, resource: { buffer: this.partialsBuf } },
        { binding: 1, resource: { buffer: this.reduceResultBuf } },
        { binding: 2, resource: { buffer: this.uReduce2 } },
        { binding: 3, resource: { buffer: this.partialsBuf } },
      ],
    });
  }

  // ---- (re)create sized buffers + dependent bind groups ------------------
  _rebuild() {
    const d = this.device;
    const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;

    // New (empty) position/voxel buffers -> force a recompute next frame.
    this._computeDirty = true;

    // Positions buffer.
    this.positionsBuf?.destroy();
    this.positionsBuf = d.createBuffer({ size: this.particlesPerBatch * 16, usage: GPUBufferUsage.STORAGE, label: "positions" });

    // Stage 1 of the bounds reduction reads the whole positions buffer.
    d.queue.writeBuffer(this.uReduce1, 0, new Uint32Array([this.particlesPerBatch, 0, 0, 0]));

    // Voxel occupancy buffer + ambient-occlusion 3D texture. AO lives in a
    // real texture so the render pass gets hardware trilinear filtering.
    this.voxelGrid?.destroy();
    this.occlusionTex?.destroy();
    const vc = this.voxelCount;
    const dim = this.voxelGridDim;
    this.voxelGrid = d.createBuffer({ size: vc * 4, usage: S, label: "voxelGrid" });
    this.occlusionTex = d.createTexture({
      size: [dim, dim, dim],
      dimension: "3d",
      format: "rgba16float",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      label: "occlusionTex",
    });
    this.occlusionView = this.occlusionTex.createView();

    // Bind groups.
    this.bgIter = d.createBindGroup({
      layout: this.bgl.iter,
      entries: [
        { binding: 0, resource: { buffer: this.positionsBuf } },
        { binding: 1, resource: { buffer: this.transformsBuf } },
        { binding: 2, resource: { buffer: this.uChaos } },
      ],
    });

    // Stage 1 of the bounds reduction: positions -> per-workgroup partials.
    // Binding 3 is unused by the reducePoints entry, but it must not alias
    // the writable partials buffer (usage-conflict validation), so the
    // result buffer stands in.
    this.bgReduce1 = d.createBindGroup({
      layout: this.bgl.reduce,
      entries: [
        { binding: 0, resource: { buffer: this.positionsBuf } },
        { binding: 1, resource: { buffer: this.partialsBuf } },
        { binding: 2, resource: { buffer: this.uReduce1 } },
        { binding: 3, resource: { buffer: this.reduceResultBuf } },
      ],
    });

    this.bgGrid = d.createBindGroup({
      layout: this.bgl.grid,
      entries: [
        { binding: 0, resource: { buffer: this.positionsBuf } },
        { binding: 1, resource: { buffer: this.combinedBuf } },
        { binding: 2, resource: { buffer: this.finalTransformBuf } },
        { binding: 3, resource: { buffer: this.voxelGrid } },
        { binding: 4, resource: this.occlusionView },
        { binding: 5, resource: { buffer: this.uGrid } },
      ],
    });

    this.bgRender = d.createBindGroup({
      layout: this.bgl.render,
      entries: [
        { binding: 0, resource: { buffer: this.positionsBuf } },
        { binding: 1, resource: { buffer: this.combinedBuf } },
        { binding: 2, resource: { buffer: this.finalTransformBuf } },
        { binding: 3, resource: this.occlusionView },
        { binding: 4, resource: { buffer: this.uRender } },
        { binding: 5, resource: this.sampler },
      ],
    });

    this._sizes = { particles: this.particlesPerBatch, voxels: vc };
  }

  // Recreate buffers if a structural size changed.
  _ensureSizes() {
    // Never exceed what a single buffer/binding can hold on this device.
    this.particlesPerBatch = Math.min(this.particlesPerBatch, this.maxParticles);
    if (
      this._sizes.particles !== this.particlesPerBatch ||
      this._sizes.voxels !== this.voxelCount
    ) {
      this._rebuild();
    }
  }

  // Compute the chaos-game schedule for the current transform count.
  _schedule(transformCount) {
    const count = Math.max(transformCount, 1);

    // Chaos-game hops per particle. The seeds already lie exactly on the
    // attractor, so hops control sampling granularity, not convergence: pick
    // enough that distinct hop sequences comfortably outnumber the particles
    // (count^iters >= particleCount * count^8).
    const iters = count >= 2
      ? Math.min(40, Math.max(12, Math.ceil(Math.log(this.particlesPerBatch) / Math.log(count)) + 8))
      : 4;

    return { count, iters };
  }

  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.floor(this.canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    if (this._fbW === w && this._fbH === h) return;
    this._fbW = w;
    this._fbH = h;

    const d = this.device;
    const RT = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
    // Scene + depth are double-buffered: accumulation bakes extra batches into
    // the hidden back pair while the front pair keeps being presented, then
    // the roles swap. Both need copy usage to seed the back from the front.
    const CP = GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST;
    const hw = Math.max(1, w >> 1);
    const hh = Math.max(1, h >> 1);

    for (const t of this.depthTexs ?? []) t?.destroy();
    for (const t of this.sceneTexs ?? []) t?.destroy();
    for (const t of [this.kuwaharaTex, this.bloomA, this.bloomB, this.tensorA, this.tensorB]) t?.destroy();

    this.depthTexs = [0, 1].map((i) =>
      d.createTexture({ size: [w, h], format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT | CP, label: `depth${i}` }));
    this.sceneTexs = [0, 1].map((i) =>
      d.createTexture({ size: [w, h], format: SCENE_FORMAT, usage: RT | CP, label: `sceneTex${i}` }));
    this.depthViews = this.depthTexs.map((t) => t.createView());
    this.sceneViews = this.sceneTexs.map((t) => t.createView());
    this._front = 0;

    this.kuwaharaTex = d.createTexture({ size: [w, h], format: SCENE_FORMAT, usage: RT, label: "kuwaharaTex" });
    this.bloomA = d.createTexture({ size: [hw, hh], format: SCENE_FORMAT, usage: RT, label: "bloomA" });
    this.bloomB = d.createTexture({ size: [hw, hh], format: SCENE_FORMAT, usage: RT, label: "bloomB" });
    // Full-res scratch for the anisotropic Kuwahara structure tensor / flow map.
    this.tensorA = d.createTexture({ size: [w, h], format: SCENE_FORMAT, usage: RT, label: "tensorA" });
    this.tensorB = d.createTexture({ size: [w, h], format: SCENE_FORMAT, usage: RT, label: "tensorB" });

    this.kuwaharaView = this.kuwaharaTex.createView();
    this.bloomAView = this.bloomA.createView();
    this.bloomBView = this.bloomB.createView();
    this.tensorAView = this.tensorA.createView();
    this.tensorBView = this.tensorB.createView();

    const postBG = (tex, ubo) =>
      d.createBindGroup({
        layout: this.bgl.post,
        entries: [
          { binding: 0, resource: this.sampler },
          { binding: 1, resource: tex },
          { binding: 2, resource: { buffer: ubo } },
        ],
      });
    const presentBG = (sceneView) =>
      d.createBindGroup({
        layout: this.bgl.present,
        entries: [
          { binding: 0, resource: this.sampler },
          { binding: 1, resource: sceneView },
          { binding: 2, resource: this.bloomAView },
          { binding: 3, resource: { buffer: this.uPresent } },
        ],
      });

    // Bloom prefilter reads whichever image is the post-Kuwahara result; blur
    // ping-pongs between the two half-res targets. Bind groups that read the
    // scene exist per scene buffer (post always reads the front one).
    this.bgPreFromScene = this.sceneViews.map((v) => postBG(v, this.uPre));
    this.bgPreFromKuw = postBG(this.kuwaharaView, this.uPre);
    this.bgBlurH = postBG(this.bloomAView, this.uBlurH);
    this.bgBlurV = postBG(this.bloomBView, this.uBlurV);
    this.bgPresentScene = this.sceneViews.map((v) => presentBG(v));
    this.bgPresentKuw = presentBG(this.kuwaharaView);

    // Anisotropic Kuwahara passes (binding 1 is unused by the first three).
    const kuwBG = (a, b) =>
      d.createBindGroup({
        layout: this.bgl.kuw,
        entries: [
          { binding: 0, resource: a },
          { binding: 1, resource: b },
          { binding: 2, resource: { buffer: this.uKuw } },
        ],
      });
    this.bgKuwStructure = this.sceneViews.map((v) => kuwBG(v, v));                  // scene -> tensorA
    this.bgKuwBlurH = kuwBG(this.tensorAView, this.tensorAView);                    // tensorA -> tensorB
    this.bgKuwAniso = kuwBG(this.tensorBView, this.tensorBView);                    // tensorB -> tensorA (flow map)
    this.bgKuwFilter = this.sceneViews.map((v) => kuwBG(v, this.tensorAView));      // scene + flow map -> kuwaharaTex
  }

  // ---- the per-frame pipeline --------------------------------------------
  frame(blender, camera) {
    this._ensureSizes();
    this.resize();

    const transformCount = blender.getTransformCount();
    if (transformCount < 1) return;

    // Pack the blended affine matrices and decide whether the attractor / voxel
    // / occlusion / fit results would actually differ from last frame.
    blender.packMatrices(this.transformData);
    const recompute = this._computeStale(transformCount);

    const s = this._schedule(transformCount);

    // Classify the frame. The render-state check must run every frame so its
    // cache tracks what is actually on screen.
    const rb = this._buildRenderUniform(s, camera);
    const renderStale = this._renderStale(rb);
    const accumTarget = this._accumBatchTarget();
    let mode;
    if (recompute) {
      mode = "compute"; // rebuild attractor + grids, draw the front from scratch
      this._batchSeed = 0;
      this._accumCount = 1;
      this._backBatches = 0;
      this._accumCheckpoint = Math.min(4, accumTarget);
    } else if (renderStale) {
      mode = "redraw"; // same cloud, new camera/colors: draw the front from scratch
      this._accumCount = 1;
      this._backBatches = 0;
      this._accumCheckpoint = Math.min(4, accumTarget);
    } else if (this._accumCount < accumTarget) {
      mode = "accumulate"; // bake one more batch into the hidden back buffer
      this._batchSeed = this._accumCount;
    } else {
      mode = "idle"; // front holds the final accumulated image: post/present only
    }

    this._writeUniforms(s, rb);
    this._writePostUniforms();

    const enc = this.device.createCommandEncoder();
    const front = this._front;
    const back = 1 - front;
    if (mode === "compute") {
      // The attractor, voxel grid, occlusion and fit transform all depend only
      // on the transforms + scale, so they're rebuilt only when those change.
      // Otherwise the existing buffer contents are reused unchanged.
      this.device.queue.writeBuffer(this.transformsBuf, 0, this.transformData);
      this._encodeIterate(enc);
      this._encodeFit(enc);
      // Fold the freshly fitted final transform into each top-level transform.
      this._dispatch(enc, this.pipe.combine, this.bgCombine, 1);
      this._encodeVoxelize(enc);
      this._encodeRender(enc, transformCount, true, front);
    } else if (mode === "redraw") {
      // The positions buffer holds whichever batch was baked last, so the
      // next accumulation run may re-draw one batch. Harmless: same points.
      this._encodeRender(enc, transformCount, true, front);
    } else if (mode === "accumulate") {
      // Accumulation happens off-screen: the front image stays on display
      // untouched while batches bake into the back buffer, and the buffers
      // swap only at checkpoints, each presenting a complete stable set.
      // The user never sees a partially accumulated in-between state.
      if (this._backBatches === 0) {
        // Seed a fresh back segment with everything the front already shows.
        const size = [this._fbW, this._fbH, 1];
        enc.copyTextureToTexture({ texture: this.sceneTexs[front] }, { texture: this.sceneTexs[back] }, size);
        enc.copyTextureToTexture({ texture: this.depthTexs[front] }, { texture: this.depthTexs[back] }, size);
        this._backBatches = this._accumCount;
      }
      this._encodeIterate(enc);
      this._encodeRender(enc, transformCount, false, back);
      this._accumCount++;
      this._backBatches++;
      if (this._accumCount >= this._accumCheckpoint || this._accumCount >= accumTarget) {
        this._front = back; // checkpoint reached: present the denser image
        this._backBatches = 0;
        this._accumCheckpoint = Math.min(this._accumCheckpoint * 4, accumTarget);
      }
    }
    this._encodePost(enc);
    this.device.queue.submit([enc.finish()]);
  }

  // Accumulation batch budget for the current particle count: enough batches
  // that the resting image converges toward ~accumTargetPoints effective
  // points (lower particle counts buffer longer), capped at 256.
  _accumBatchTarget() {
    return Math.max(1, Math.min(256, Math.round(this.accumTargetPoints / this.particlesPerBatch)));
  }

  // True when the compute results would differ from the last computed frame.
  // Records the new state when stale so subsequent identical frames are skipped.
  _computeStale(transformCount) {
    const len = transformCount * 16;
    let stale =
      this._computeDirty ||
      !this._cachedTransformData ||
      transformCount !== this._cachedTransformCount ||
      this.scalePadding !== this._cachedScalePadding;

    if (!stale) {
      const a = this.transformData;
      const b = this._cachedTransformData;
      for (let i = 0; i < len; i++) {
        if (a[i] !== b[i]) { stale = true; break; }
      }
    }

    if (stale) {
      if (!this._cachedTransformData) this._cachedTransformData = new Float32Array(this.transformData.length);
      this._cachedTransformData.set(this.transformData);
      this._cachedTransformCount = transformCount;
      this._cachedScalePadding = this.scalePadding;
      this._computeDirty = false;
    }
    return stale;
  }

  // Build the uRender CPU bytes (viewProj, colors, grid/AO params) for this
  // frame. Kept separate from the upload so frame() can byte-compare it
  // against what is currently drawn into the scene texture.
  _buildRenderUniform(s, camera) {
    const dim = this.voxelGridDim;
    const gridBounds = 2 * this.voxelBounds * this.scalePadding;
    const aspect = this.canvas.width / Math.max(1, this.canvas.height);
    const vp = camera.viewProj(aspect);
    const rb = new ArrayBuffer(128);
    new Float32Array(rb, 0, 16).set(vp);
    new Float32Array(rb, 64, 4).set([...this.particleColor, 1]);
    new Float32Array(rb, 80, 4).set([...this.occlusionColor, 1]);
    new Uint32Array(rb, 96, 2).set([dim, s.count]);
    new Float32Array(rb, 104, 3).set([gridBounds, this.occlusionMultiplier, this.occlusionAttenuation]);
    return rb;
  }

  // True when the visible render state (uRender bytes, background color or
  // framebuffer size) differs from what the scene texture currently shows.
  // Records the new state so accumulation can resume next frame.
  _renderStale(rb) {
    const a = new Uint32Array(rb);
    const b = this._cachedRenderU;
    const bg = this.backgroundColor;
    let stale =
      !b ||
      this._cachedFbW !== this._fbW ||
      this._cachedFbH !== this._fbH ||
      this._cachedBg[0] !== bg[0] ||
      this._cachedBg[1] !== bg[1] ||
      this._cachedBg[2] !== bg[2];
    if (!stale) {
      for (let i = 0; i < a.length; i++) {
        if (a[i] !== b[i]) { stale = true; break; }
      }
    }
    if (stale) {
      if (!this._cachedRenderU) this._cachedRenderU = new Uint32Array(a.length);
      this._cachedRenderU.set(a);
      this._cachedBg = [bg[0], bg[1], bg[2]];
      this._cachedFbW = this._fbW;
      this._cachedFbH = this._fbH;
    }
    return stale;
  }

  _writeUniforms(s, rb) {
    const q = this.device.queue;
    const dim = this.voxelGridDim;

    // uChaos: header + per-transform fixed-point seeds (exact attractor points).
    new Uint32Array(this.uChaosCPU, 0, 5).set([
      s.count, this.particlesPerBatch, this._dims(this.particlesPerBatch).width, s.iters, this._batchSeed | 0,
    ]);
    new Float32Array(this.uChaosCPU, 32).set(this._fixedPoints(s.count));
    q.writeBuffer(this.uChaos, 0, this.uChaosCPU);

    // uCombine: transformCount
    q.writeBuffer(this.uCombine, 0, new Uint32Array([s.count, 0, 0, 0]));

    // uFit: targetBounds, scalePadding, particleCount. The fit scales the
    // fractal to radius = voxelBounds * scalePadding. The bounds now come
    // from the exact rendered cloud, so the only slack needed is a small
    // safety margin for float rounding at the clip boundary.
    q.writeBuffer(this.uFit, 0, new Float32Array([this.voxelBounds * 0.995, this.scalePadding, this.particlesPerBatch, 0]));

    // The voxel grid box tracks the fractal size (= its diameter) so the fixed
    // gridSize always spans the fractal: lighting resolution stays constant as
    // the Scale slider changes, and the fractal never clips against the box.
    const gridBounds = 2 * this.voxelBounds * this.scalePadding;

    // uGrid: gridSize, transformCount, particleCount, voxelCount, gridBounds, voxWidth
    const gridU = new ArrayBuffer(32);
    new Uint32Array(gridU, 0, 4).set([dim, s.count, this.particlesPerBatch, this.voxelCount]);
    new Float32Array(gridU, 16, 1).set([gridBounds]);
    new Uint32Array(gridU, 20, 1).set([this._dims(this.particlesPerBatch).width]);
    q.writeBuffer(this.uGrid, 0, gridU);

    // uRender (bytes prebuilt by _buildRenderUniform)
    q.writeBuffer(this.uRender, 0, rb);
  }

  // Run a single dispatch in its own compute pass. Each dependent step gets a
  // dedicated pass so WebGPU's automatic inter-pass barriers guarantee that
  // reads observe the previous step's writes (read-after-write hazards).
  _dispatch(enc, pipeline, bindGroup, groups, dynamicOffsets) {
    // `groups` is a workgroup count (1D) or [gx, gy] for a flattened 2D dispatch.
    const gx = Array.isArray(groups) ? groups[0] : groups;
    const gy = Array.isArray(groups) ? groups[1] : 1;
    if (gx <= 0 || gy <= 0) return;
    const p = enc.beginComputePass();
    p.setPipeline(pipeline);
    if (dynamicOffsets) p.setBindGroup(0, bindGroup, dynamicOffsets);
    else p.setBindGroup(0, bindGroup);
    p.dispatchWorkgroups(gx, gy);
    p.end();
  }

  // Split a thread count into a 2D workgroup dispatch that respects the
  // per-dimension workgroup limit. `width` is the number of threads the x
  // dimension spans, used by shaders to flatten (gid.y * width + gid.x).
  _dims(threadCount) {
    const groups = Math.max(1, Math.ceil(threadCount / WG));
    const x = Math.min(groups, this.maxComputeDim);
    const y = Math.ceil(groups / x);
    return { x, y, width: x * WG };
  }

  // Measure the freshly iterated cloud (min/max/sum in two reduction stages)
  // and build the auto-fit transform from it. Because the bounds come from
  // the exact points that will be rendered, the fractal cannot outgrow the
  // fitted box and clip.
  _encodeFit(enc) {
    this._dispatch(enc, this.pipe.reducePoints, this.bgReduce1, REDUCE_PARTIALS);
    this._dispatch(enc, this.pipe.reduceBounds, this.bgReduce2, 1);
    this._dispatch(enc, this.pipe.fit, this.bgFit, 1);
  }

  // Per-transform fixed points: solve (I - A) x = t for each affine map. The
  // fixed point of a contractive map lies exactly on the attractor, which
  // makes it an ideal chaos-game seed (no warm-up convergence needed). Falls
  // back to the origin when I - A is near-singular (non-contractive map).
  _fixedPoints(count) {
    const out = new Float32Array(MAX_TRANSFORMS * 4);
    const td = this.transformData;
    const det3 = (a, b, c, d, e, f, g, h, i) =>
      a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    for (let i = 0; i < count; i++) {
      const o = i * 16; // column-major mat4
      const m00 = 1 - td[o + 0], m01 = -td[o + 4], m02 = -td[o + 8];
      const m10 = -td[o + 1], m11 = 1 - td[o + 5], m12 = -td[o + 9];
      const m20 = -td[o + 2], m21 = -td[o + 6], m22 = 1 - td[o + 10];
      const tx = td[o + 12], ty = td[o + 13], tz = td[o + 14];
      const det = det3(m00, m01, m02, m10, m11, m12, m20, m21, m22);
      if (Math.abs(det) > 1e-6) {
        out[i * 4 + 0] = det3(tx, m01, m02, ty, m11, m12, tz, m21, m22) / det;
        out[i * 4 + 1] = det3(m00, tx, m02, m10, ty, m12, m20, tz, m22) / det;
        out[i * 4 + 2] = det3(m00, m01, tx, m10, m11, ty, m20, m21, tz) / det;
      }
    }
    return out;
  }

  _encodeIterate(enc) {
    const d = this._dims(this.particlesPerBatch);
    this._dispatch(enc, this.pipe.iterate, this.bgIter, [d.x, d.y]);
  }

  _encodeVoxelize(enc) {
    const vGroups = Math.ceil(this.voxelCount / WG); // voxel grid stays within the 1D cap
    const vd = this._dims(this.particlesPerBatch);
    this._dispatch(enc, this.pipe.clearGrids, this.bgGrid, vGroups);
    this._dispatch(enc, this.pipe.voxelize, this.bgGrid, [vd.x, vd.y]);
    this._dispatch(enc, this.pipe.occlusion, this.bgGrid, vGroups);
  }

  // Draw the point cloud into scene buffer `target`. With clear=false the
  // pass loads the existing color + depth instead, so a fresh chaos batch
  // adds to what previous batches already drew (progressive accumulation).
  _encodeRender(enc, transformCount, clear, target) {
    const bg = this.backgroundColor;
    const p = enc.beginRenderPass({
      colorAttachments: [{
        view: this.sceneViews[target],
        clearValue: { r: bg[0], g: bg[1], b: bg[2], a: 1 },
        loadOp: clear ? "clear" : "load",
        storeOp: "store",
      }],
      depthStencilAttachment: {
        view: this.depthViews[target],
        depthClearValue: 1.0,
        depthLoadOp: clear ? "clear" : "load",
        depthStoreOp: "store",
      },
    });
    p.setPipeline(this.pipe.render);
    p.setBindGroup(0, this.bgRender);
    // vertexCount = particlesPerBatch, instanceCount = transformCount
    p.draw(this.particlesPerBatch, transformCount);
    p.end();
  }

  // Run one full-screen post pass into the given target view.
  _fullscreen(enc, pipeline, bindGroup, targetView) {
    const p = enc.beginRenderPass({
      colorAttachments: [{ view: targetView, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
    });
    p.setPipeline(pipeline);
    p.setBindGroup(0, bindGroup);
    p.draw(3);
    p.end();
  }

  _writePostUniforms() {
    const q = this.device.queue;
    const tx = 1 / this._fbW;
    const ty = 1 / this._fbH;
    const htx = 1 / Math.max(1, this._fbW >> 1);
    const hty = 1 / Math.max(1, this._fbH >> 1);

    // KuwParams = { kernelSize, q, alpha, zeroCrossing, blurRadius, ... }
    q.writeBuffer(this.uKuw, 0, new Float32Array([
      this.kuwaharaKernelSize, this.kuwaharaSharpness, this.kuwaharaAlpha,
      this.kuwaharaZeroCrossing, this.kuwaharaBlurRadius, 0, 0, 0,
    ]));
    // PostParams = { p0: vec4 (texel.xy, radius/threshold, -), p1: vec4 (dir.xy, spread, -) }
    q.writeBuffer(this.uPre, 0, new Float32Array([tx, ty, this.bloomThreshold, 0, 0, 0, 0, 0]));
    q.writeBuffer(this.uBlurH, 0, new Float32Array([htx, hty, 0, 0, 1, 0, this.bloomSpread, 0]));
    q.writeBuffer(this.uBlurV, 0, new Float32Array([htx, hty, 0, 0, 0, 1, this.bloomSpread, 0]));
    q.writeBuffer(this.uPresent, 0, new Float32Array([this.bloomIntensity, 0, 0, 0, 0, 0, 0, 0]));
  }

  _encodePost(enc) {
    // Post always reads the FRONT scene buffer; accumulation renders into the
    // back one, so partially accumulated frames are never visible.
    const f = this._front;

    // 1) Optional anisotropic Kuwahara filter (4 passes -> kuwaharaTex).
    const useKuwahara = this.kuwaharaEnabled;
    if (useKuwahara) {
      this._fullscreen(enc, this.pipe.kuwStructure, this.bgKuwStructure[f], this.tensorAView); // scene -> tensorA
      this._fullscreen(enc, this.pipe.kuwBlurH, this.bgKuwBlurH, this.tensorBView);            // tensorA -> tensorB
      this._fullscreen(enc, this.pipe.kuwAniso, this.bgKuwAniso, this.tensorAView);            // tensorB -> tensorA (flow map)
      this._fullscreen(enc, this.pipe.kuwFilter, this.bgKuwFilter[f], this.kuwaharaView);      // scene + flow map -> kuwaharaTex
    }
    const bgPrefilter = useKuwahara ? this.bgPreFromKuw : this.bgPreFromScene[f];
    const bgPresent = useKuwahara ? this.bgPresentKuw : this.bgPresentScene[f];

    // 2) Optional bloom: bright-pass, then several separable blur iterations
    //    (each H+V pass widens and softens the glow).
    if (this.bloomIntensity > 0.0001) {
      this._fullscreen(enc, this.pipe.prefilter, bgPrefilter, this.bloomAView); // -> bloomA
      for (let i = 0; i < this.bloomIterations; i++) {
        this._fullscreen(enc, this.pipe.blur, this.bgBlurH, this.bloomBView);   // A -> B (horizontal)
        this._fullscreen(enc, this.pipe.blur, this.bgBlurV, this.bloomAView);   // B -> A (vertical)
      }
    }

    // 3) Composite to the swap chain.
    const out = this.ctx.getCurrentTexture().createView();
    this._fullscreen(enc, this.pipe.present, bgPresent, out);
  }
}
