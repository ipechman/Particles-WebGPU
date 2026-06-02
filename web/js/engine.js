// engine.js
// WebGPU engine: owns the device, buffers and pipelines, and runs the per-frame
// pipeline that mirrors IteratedFunctionSystem.Update:
//   predict final (auto-fit) transform -> iterate attractor -> voxelize +
//   ambient occlusion -> render instanced points.

import { mat4 } from "./math.js";
import { MAX_TRANSFORMS } from "./blender.js";

const WG = 64;                 // generic workgroup size
const SLOT = 256;              // dynamic uniform slot stride (alignment)
const MAX_SLOTS = 40;          // generations / LOD levels headroom
const MAX_LOD = 1 << 20;       // cap on low-detail point count
const SCENE_FORMAT = "rgba16float"; // HDR offscreen target for post-processing

const SHADER_FILES = {
  iterate: "shaders/iterate.wgsl",
  lod: "shaders/lod.wgsl",
  reduce: "shaders/reduce.wgsl",
  fit: "shaders/fit.wgsl",
  voxelize: "shaders/voxelize.wgsl",
  render: "shaders/render.wgsl",
  post: "shaders/post.wgsl",
  present: "shaders/present.wgsl",
};

export class Engine {
  constructor(canvas) {
    this.canvas = canvas;

    // ---- tunable configuration (defaults mirror the original project) ----
    this.particlesPerBatch = 1 << 20;     // 1,048,576
    this.voxelGridDim = 128;              // grid resolution
    this.voxelBounds = 3.0;               // world-space box size
    this.lowDetailGenerations = 8;
    this.scalePadding = 0.5;              // flagship fit padding

    this.particleColor = [0.93, 0.94, 0.96]; // neutral near-white highlight
    this.occlusionColor = [0.103773594, 0.014195448, 0.014195448];
    this.occlusionMultiplier = 1.0;
    this.occlusionAttenuation = 1.0;
    this.backgroundColor = [0.0, 0.0, 0.0];

    // ---- post-processing ----
    this.kuwaharaEnabled = false;
    this.kuwaharaRadius = 4;        // window radius for the original Kuwahara filter
    this.bloomIntensity = 0.5;      // 0 disables bloom
    this.bloomThreshold = 0.3;      // low enough that bloom is visible on most fractals
    this.bloomSpread = 2.0;
    this.bloomIterations = 3;       // blur passes -> width/softness of the glow

    // sizes currently realized in GPU buffers (for change detection)
    this._sizes = { particles: -1, lod: -1, voxels: -1 };
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
        const res = await fetch(path);
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
          buf(2, C, "uniform", true),
        ],
      }),
      lod: d.createBindGroupLayout({
        entries: [
          buf(0, C, "read-only-storage"),
          buf(1, C, "storage"),
          buf(2, C, "read-only-storage"),
          buf(3, C, "uniform", true),
        ],
      }),
      reduce: d.createBindGroupLayout({
        entries: [buf(0, C, "read-only-storage"), buf(1, C, "storage"), buf(2, C, "uniform")],
      }),
      fit: d.createBindGroupLayout({
        entries: [buf(0, C, "read-only-storage"), buf(1, C, "storage"), buf(2, C, "uniform")],
      }),
      grid: d.createBindGroupLayout({
        entries: [
          buf(0, C, "read-only-storage"),
          buf(1, C, "read-only-storage"),
          buf(2, C, "read-only-storage"),
          buf(3, C, "storage"),
          buf(4, C, "storage"),
          buf(5, C, "uniform"),
        ],
      }),
      render: d.createBindGroupLayout({
        entries: [
          buf(0, VF, "read-only-storage"),
          buf(1, VF, "read-only-storage"),
          buf(2, VF, "read-only-storage"),
          buf(3, VF, "read-only-storage"),
          buf(4, VF, "uniform"),
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
    };

    this.pl = {
      iter: d.createPipelineLayout({ bindGroupLayouts: [this.bgl.iter] }),
      lod: d.createPipelineLayout({ bindGroupLayouts: [this.bgl.lod] }),
      reduce: d.createPipelineLayout({ bindGroupLayouts: [this.bgl.reduce] }),
      fit: d.createPipelineLayout({ bindGroupLayouts: [this.bgl.fit] }),
      grid: d.createPipelineLayout({ bindGroupLayouts: [this.bgl.grid] }),
      render: d.createPipelineLayout({ bindGroupLayouts: [this.bgl.render] }),
      post: d.createPipelineLayout({ bindGroupLayouts: [this.bgl.post] }),
      present: d.createPipelineLayout({ bindGroupLayouts: [this.bgl.present] }),
    };
  }

  _createPipelines() {
    const d = this.device;
    const comp = (mod, entry, layout) =>
      d.createComputePipeline({ layout, compute: { module: this.modules[mod], entryPoint: entry } });

    this.pipe = {
      iterate: comp("iterate", "iterate", this.pl.iter),
      lodFirst: comp("lod", "lodFirst", this.pl.lod),
      lodIterate: comp("lod", "lodIterate", this.pl.lod),
      reduce: comp("reduce", "reduce", this.pl.reduce),
      fit: comp("fit", "fit", this.pl.fit),
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

    this.pipe.kuwahara = post("fsKuwahara", SCENE_FORMAT);
    this.pipe.prefilter = post("fsPrefilter", SCENE_FORMAT);
    this.pipe.blur = post("fsBlur", SCENE_FORMAT);

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

    // Dynamic-offset uniform rings.
    this.uIter = d.createBuffer({ size: SLOT * MAX_SLOTS, usage: U, label: "uIter" });
    this.uLod = d.createBuffer({ size: SLOT * MAX_SLOTS, usage: U, label: "uLod" });
    this.uIterCPU = new Uint32Array((SLOT * MAX_SLOTS) / 4);
    this.uLodCPU = new Uint32Array((SLOT * MAX_SLOTS) / 4);

    // Small single-use uniforms.
    this.uReduce = d.createBuffer({ size: 16, usage: U, label: "uReduce" });
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
  }

  // ---- (re)create sized buffers + dependent bind groups ------------------
  _rebuild() {
    const d = this.device;
    const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;

    this.lodCount = MAX_LOD; // upper bound; recomputed per preset in _schedule

    // Positions buffer.
    this.positionsBuf?.destroy();
    this.positionsBuf = d.createBuffer({ size: this.particlesPerBatch * 16, usage: GPUBufferUsage.STORAGE, label: "positions" });

    // LOD ping-pong buffers (sized to the cap; actual N <= cap).
    this.lodA?.destroy();
    this.lodB?.destroy();
    this.lodA = d.createBuffer({ size: MAX_LOD * 16, usage: GPUBufferUsage.STORAGE, label: "lodA" });
    this.lodB = d.createBuffer({ size: MAX_LOD * 16, usage: GPUBufferUsage.STORAGE, label: "lodB" });

    // Voxel + occlusion grids.
    this.voxelGrid?.destroy();
    this.occlusionGrid?.destroy();
    const vc = this.voxelCount;
    this.voxelGrid = d.createBuffer({ size: vc * 4, usage: S, label: "voxelGrid" });
    this.occlusionGrid = d.createBuffer({ size: vc * 4, usage: GPUBufferUsage.STORAGE, label: "occlusionGrid" });

    // Bind groups.
    this.bgIter = d.createBindGroup({
      layout: this.bgl.iter,
      entries: [
        { binding: 0, resource: { buffer: this.positionsBuf } },
        { binding: 1, resource: { buffer: this.transformsBuf } },
        { binding: 2, resource: { buffer: this.uIter, size: SLOT } },
      ],
    });

    const lodEntries = (inBuf, outBuf) => ({
      layout: this.bgl.lod,
      entries: [
        { binding: 0, resource: { buffer: inBuf } },
        { binding: 1, resource: { buffer: outBuf } },
        { binding: 2, resource: { buffer: this.transformsBuf } },
        { binding: 3, resource: { buffer: this.uLod, size: SLOT } },
      ],
    });
    this.bgLodAB = d.createBindGroup(lodEntries(this.lodA, this.lodB)); // in A -> out B
    this.bgLodBA = d.createBindGroup(lodEntries(this.lodB, this.lodA)); // in B -> out A

    this.bgReduceA = d.createBindGroup({
      layout: this.bgl.reduce,
      entries: [
        { binding: 0, resource: { buffer: this.lodA } },
        { binding: 1, resource: { buffer: this.reduceResultBuf } },
        { binding: 2, resource: { buffer: this.uReduce } },
      ],
    });
    this.bgReduceB = d.createBindGroup({
      layout: this.bgl.reduce,
      entries: [
        { binding: 0, resource: { buffer: this.lodB } },
        { binding: 1, resource: { buffer: this.reduceResultBuf } },
        { binding: 2, resource: { buffer: this.uReduce } },
      ],
    });

    this.bgGrid = d.createBindGroup({
      layout: this.bgl.grid,
      entries: [
        { binding: 0, resource: { buffer: this.positionsBuf } },
        { binding: 1, resource: { buffer: this.transformsBuf } },
        { binding: 2, resource: { buffer: this.finalTransformBuf } },
        { binding: 3, resource: { buffer: this.voxelGrid } },
        { binding: 4, resource: { buffer: this.occlusionGrid } },
        { binding: 5, resource: { buffer: this.uGrid } },
      ],
    });

    this.bgRender = d.createBindGroup({
      layout: this.bgl.render,
      entries: [
        { binding: 0, resource: { buffer: this.positionsBuf } },
        { binding: 1, resource: { buffer: this.transformsBuf } },
        { binding: 2, resource: { buffer: this.finalTransformBuf } },
        { binding: 3, resource: { buffer: this.occlusionGrid } },
        { binding: 4, resource: { buffer: this.uRender } },
      ],
    });

    this._sizes = { particles: this.particlesPerBatch, lod: MAX_LOD, voxels: vc };
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

  // Compute the iteration / LOD schedule for the current transform count.
  _schedule(transformCount) {
    const count = Math.max(transformCount, 1);

    // Iterated-system generations tile [0, particlesPerBatch) contiguously.
    const gens = [];
    let iterated = count;
    let prev = count;
    gens.push({ offset: 0, limit: count });
    while (iterated < this.particlesPerBatch) {
      const genSize = prev * count;
      const limit = Math.min(iterated + genSize, this.particlesPerBatch);
      gens.push({ offset: iterated, limit });
      iterated += genSize;
      prev = genSize;
      if (gens.length >= MAX_SLOTS) break;
    }

    // LOD generations, capped so the point count stays bounded.
    let lodGens = this.lowDetailGenerations;
    if (count >= 2) {
      const maxByCap = Math.floor(Math.log(MAX_LOD) / Math.log(count));
      lodGens = Math.min(lodGens, Math.max(1, maxByCap));
    }
    lodGens = Math.max(1, Math.min(lodGens, MAX_SLOTS - 1));
    const N = Math.min(Math.pow(count, lodGens), MAX_LOD);
    const levels = lodGens - 1; // number of lodIterate dispatches
    const lodFinalIsA = levels % 2 === 0; // lodFirst writes A; each level swaps

    // 2D dispatch dims for each generation (large counts exceed the 1D cap).
    const genDims = gens.map((g) => this._dims(g.limit - g.offset));

    return { count, gens, genDims, lodGens, N, levels, lodFinalIsA };
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
    const hw = Math.max(1, w >> 1);
    const hh = Math.max(1, h >> 1);

    this.depthTex?.destroy();
    this.depthTex = d.createTexture({ size: [w, h], format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT });
    this.depthView = this.depthTex.createView();

    for (const t of [this.sceneTex, this.kuwaharaTex, this.bloomA, this.bloomB]) t?.destroy();
    this.sceneTex = d.createTexture({ size: [w, h], format: SCENE_FORMAT, usage: RT, label: "sceneTex" });
    this.kuwaharaTex = d.createTexture({ size: [w, h], format: SCENE_FORMAT, usage: RT, label: "kuwaharaTex" });
    this.bloomA = d.createTexture({ size: [hw, hh], format: SCENE_FORMAT, usage: RT, label: "bloomA" });
    this.bloomB = d.createTexture({ size: [hw, hh], format: SCENE_FORMAT, usage: RT, label: "bloomB" });

    this.sceneView = this.sceneTex.createView();
    this.kuwaharaView = this.kuwaharaTex.createView();
    this.bloomAView = this.bloomA.createView();
    this.bloomBView = this.bloomB.createView();

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

    // Kuwahara reads the raw scene; bloom prefilter reads whichever image is the
    // post-Kuwahara result; blur ping-pongs between the two half-res targets.
    this.bgKuwahara = postBG(this.sceneView, this.uKuw);
    this.bgPreFromScene = postBG(this.sceneView, this.uPre);
    this.bgPreFromKuw = postBG(this.kuwaharaView, this.uPre);
    this.bgBlurH = postBG(this.bloomAView, this.uBlurH);
    this.bgBlurV = postBG(this.bloomBView, this.uBlurV);
    this.bgPresentScene = presentBG(this.sceneView);
    this.bgPresentKuw = presentBG(this.kuwaharaView);
  }

  // ---- the per-frame pipeline --------------------------------------------
  frame(blender, camera) {
    this._ensureSizes();
    this.resize();

    const transformCount = blender.getTransformCount();
    if (transformCount < 1) return;

    // Upload the freshly blended affine matrices.
    blender.packMatrices(this.transformData);
    this.device.queue.writeBuffer(this.transformsBuf, 0, this.transformData);

    const s = this._schedule(transformCount);
    this._writeUniforms(s, camera);
    this._writePostUniforms();

    const enc = this.device.createCommandEncoder();
    this._encodePredict(enc, s);
    this._encodeIterate(enc, s);
    this._encodeVoxelize(enc);
    this._encodeRender(enc, transformCount);
    this._encodePost(enc);
    this.device.queue.submit([enc.finish()]);
  }

  _writeUniforms(s, camera) {
    const q = this.device.queue;
    const dim = this.voxelGridDim;

    // uIter slots (one per generation): count, genOffset, genLimit, dispatchWidth.
    this.uIterCPU.fill(0);
    for (let g = 0; g < s.gens.length; g++) {
      const o = (g * SLOT) / 4;
      this.uIterCPU[o + 0] = s.count;
      this.uIterCPU[o + 1] = s.gens[g].offset;
      this.uIterCPU[o + 2] = s.gens[g].limit;
      this.uIterCPU[o + 3] = s.genDims[g].width;
    }
    q.writeBuffer(this.uIter, 0, this.uIterCPU.buffer, 0, s.gens.length * SLOT || SLOT);

    // uLod slots: slot 0 = lodFirst (inCount = count); slots 1.. = levels.
    this.uLodCPU.fill(0);
    {
      const o0 = 0;
      this.uLodCPU[o0 + 0] = s.count;
      this.uLodCPU[o0 + 1] = s.count;
      let inCount = s.count;
      for (let l = 1; l <= s.levels; l++) {
        const o = (l * SLOT) / 4;
        this.uLodCPU[o + 0] = s.count;
        this.uLodCPU[o + 1] = inCount; // size of the input generation
        inCount = inCount * s.count;
      }
    }
    q.writeBuffer(this.uLod, 0, this.uLodCPU.buffer, 0, (s.levels + 1) * SLOT);

    // uReduce: inputSize = N
    q.writeBuffer(this.uReduce, 0, new Uint32Array([s.N, 0, 0, 0]));

    // uFit: targetBounds, scalePadding, particleCount(=N as float). The fit
    // scales the fractal to radius = voxelBounds * scalePadding.
    q.writeBuffer(this.uFit, 0, new Float32Array([this.voxelBounds, this.scalePadding, s.N, 0]));

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

    // uRender
    const aspect = this.canvas.width / Math.max(1, this.canvas.height);
    const vp = camera.viewProj(aspect);
    const rb = new ArrayBuffer(128);
    new Float32Array(rb, 0, 16).set(vp);
    new Float32Array(rb, 64, 4).set([...this.particleColor, 1]);
    new Float32Array(rb, 80, 4).set([...this.occlusionColor, 1]);
    new Uint32Array(rb, 96, 2).set([dim, s.count]);
    new Float32Array(rb, 104, 3).set([gridBounds, this.occlusionMultiplier, this.occlusionAttenuation]);
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

  _encodePredict(enc, s) {
    // Low-detail generation (ping-pong between A and B).
    this._dispatch(enc, this.pipe.lodFirst, this.bgLodBA, Math.ceil(s.count / WG), [0]); // out = A

    let outIsA = true;
    let inCount = s.count;
    for (let l = 1; l <= s.levels; l++) {
      outIsA = !outIsA;
      const bg = outIsA ? this.bgLodBA : this.bgLodAB;
      this._dispatch(enc, this.pipe.lodIterate, bg, Math.ceil(inCount / WG), [l * SLOT]);
      inCount = inCount * s.count;
    }

    // Reduce final LOD buffer -> min/max/sum, then build the auto-fit transform.
    this._dispatch(enc, this.pipe.reduce, s.lodFinalIsA ? this.bgReduceA : this.bgReduceB, 1);
    this._dispatch(enc, this.pipe.fit, this.bgFit, 1);
  }

  _encodeIterate(enc, s) {
    for (let g = 0; g < s.gens.length; g++) {
      const { offset, limit } = s.gens[g];
      if (limit - offset <= 0) continue;
      const d = s.genDims[g];
      this._dispatch(enc, this.pipe.iterate, this.bgIter, [d.x, d.y], [g * SLOT]);
    }
  }

  _encodeVoxelize(enc) {
    const vGroups = Math.ceil(this.voxelCount / WG); // voxel grid stays within the 1D cap
    const vd = this._dims(this.particlesPerBatch);
    this._dispatch(enc, this.pipe.clearGrids, this.bgGrid, vGroups);
    this._dispatch(enc, this.pipe.voxelize, this.bgGrid, [vd.x, vd.y]);
    this._dispatch(enc, this.pipe.occlusion, this.bgGrid, vGroups);
  }

  _encodeRender(enc, transformCount) {
    const bg = this.backgroundColor;
    const p = enc.beginRenderPass({
      colorAttachments: [{ view: this.sceneView, clearValue: { r: bg[0], g: bg[1], b: bg[2], a: 1 }, loadOp: "clear", storeOp: "store" }],
      depthStencilAttachment: {
        view: this.depthView,
        depthClearValue: 1.0,
        depthLoadOp: "clear",
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

    // PostParams = { p0: vec4 (texel.xy, radius/threshold, -), p1: vec4 (dir.xy, spread, -) }
    q.writeBuffer(this.uKuw, 0, new Float32Array([tx, ty, this.kuwaharaRadius, 0, 0, 0, 0, 0]));
    q.writeBuffer(this.uPre, 0, new Float32Array([tx, ty, this.bloomThreshold, 0, 0, 0, 0, 0]));
    q.writeBuffer(this.uBlurH, 0, new Float32Array([htx, hty, 0, 0, 1, 0, this.bloomSpread, 0]));
    q.writeBuffer(this.uBlurV, 0, new Float32Array([htx, hty, 0, 0, 0, 1, this.bloomSpread, 0]));
    q.writeBuffer(this.uPresent, 0, new Float32Array([this.bloomIntensity, 0, 0, 0, 0, 0, 0, 0]));
  }

  _encodePost(enc) {
    // 1) Optional Kuwahara filter (scene -> kuwaharaTex).
    const useKuwahara = this.kuwaharaEnabled;
    if (useKuwahara) {
      this._fullscreen(enc, this.pipe.kuwahara, this.bgKuwahara, this.kuwaharaView);
    }
    const bgPrefilter = useKuwahara ? this.bgPreFromKuw : this.bgPreFromScene;
    const bgPresent = useKuwahara ? this.bgPresentKuw : this.bgPresentScene;

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
