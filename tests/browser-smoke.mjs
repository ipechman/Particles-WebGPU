// Run with `npm run test:browser` after `npx playwright install chromium`.
// CI uses Chromium's software WebGPU adapter, so no physical GPU is required.
// These are rendering/correctness checks, not representative GPU benchmarks.
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { createHash } from "node:crypto";
import { readFile, mkdir, writeFile } from "node:fs/promises";
import { dirname, extname, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";
import { PNG } from "pngjs";
import { batchPoints, fixedPoint, fullDepth, qualityFixtures } from "./helpers/sampling.mjs";
import { dispatchShape } from "../web/js/performance.js";
import { runViewChecks } from "./helpers/browser-view-checks.mjs";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "../web");
const artifacts = resolve(root, "../test-results/browser");
await mkdir(artifacts, { recursive: true });
const messages = [];
const checks = [];
const errors = [];
const mimeTypes = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".wgsl": "text/plain; charset=utf-8",
};
const server = createServer(async (req, res) => {
  try {
    const pathname = decodeURIComponent(new URL(req.url, "http://localhost").pathname);
    if (pathname === "/favicon.ico") {
      res.writeHead(204).end();
      return;
    }
    const path = resolve(root, `.${pathname === "/" ? "/index.html" : pathname}`);
    if (!path.startsWith(root + sep)) {
      res.writeHead(403).end();
      return;
    }
    const content = await readFile(path);
    res.writeHead(200, { "Content-Type": mimeTypes[extname(path)] || "application/octet-stream" });
    res.end(content);
  } catch {
    res.writeHead(404).end();
  }
});
await new Promise((done) => server.listen(0, "127.0.0.1", done));
const base = `http://127.0.0.1:${server.address().port}`;
let browser;
let page;
let browserVersion;

const state = () => page.evaluate(() => {
  const { engine, blender } = window.__app;
  return {
    mode: engine.frameMode,
    accumulation: engine._accumCount,
    target: engine._accumBatchTarget(),
    particles: engine.particlesPerBatch,
    transformCount: blender.getTransformCount(),
    revision: engine.sceneRevision,
    postStats: { ...engine.postStats },
    profile: engine.profiler?.latest ?? null,
    view: engine.viewStats,
  };
});

async function healthy() {
  const gpuErrors = await page.evaluate(() => window.__gpuErrors);
  assert.deepEqual(errors, [], `Browser errors:\n${errors.join("\n")}`);
  assert.deepEqual(gpuErrors, [], `WebGPU errors:\n${gpuErrors.join("\n")}`);
  assert.equal(await page.locator("#error").evaluate((el) => el.classList.contains("hidden")), true);
}

async function settled() {
  // Cross a RAF boundary before checking: the UI event may have changed state
  // while frameMode still describes the preceding frame.
  await page.evaluate(() => new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(done))));
  await page.waitForFunction(() => {
    const e = window.__app.engine;
    return e.frameMode === "idle" && (e.samplingMode !== "view" || e._fitReadbackFailed ||
      (e._fitCPU && e._viewKey && !e._fitPending && !e._viewPending));
  }, null, { timeout: 90_000 });
  await page.evaluate(() => window.__app.engine.device.queue.onSubmittedWorkDone());
  await healthy();
  const current = await state();
  assert.equal(current.accumulation, current.target, "Accumulation must reach its configured budget");
  return current;
}

async function renderDiagnostics() {
  return page.evaluate(async () => {
    const { engine: e, camera } = window.__app;
    const d = e.device, width = e._fbW, height = e._fbH, front = e._front;
    const rowBytes = Math.ceil(width * 8 / 256) * 256;
    const textureReadback = d.createBuffer({ size: rowBytes * height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const presentRowBytes = Math.ceil(width * 4 / 256) * 256;
    const presentTexture = d.createTexture({ size: [width, height], format: e.format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    const presentReadback = d.createBuffer({ size: presentRowBytes * height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const packed = d.createBuffer({ size: 176, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const metadata = d.createBuffer({ size: 320, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    try {
      // Read-only bindings inspect buffers that intentionally lack COPY_SRC.
      // The website's pipelines, resource usages, and uniforms stay untouched.
      const module = d.createShaderModule({ code: `
        @group(0) @binding(0) var<storage, read> fit: array<f32>;
        @group(0) @binding(1) var<storage, read> combined: array<f32>;
        @group(0) @binding(2) var<storage, read> bounds: array<f32>;
        @group(0) @binding(3) var<storage, read_write> output: array<f32>;
        @compute @workgroup_size(1) fn copyMetadata() {
          for (var i = 0u; i < 16u; i++) { output[i] = fit[i]; output[16u+i] = combined[i]; }
          for (var i = 0u; i < 12u; i++) { output[32u+i] = bounds[i]; }
        }` });
      const pipeline = d.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "copyMetadata" } });
      const bindGroup = d.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries:
        [e.finalTransformBuf, e.combinedBuf, e.reduceResultBuf, packed].map((buffer, binding) => ({ binding, resource: { buffer } })) });
      const encoder = d.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, bindGroup); pass.dispatchWorkgroups(1); pass.end();
      encoder.copyBufferToBuffer(packed, 0, metadata, 0, 176);
      encoder.copyBufferToBuffer(e.positionsBuf, 0, metadata, 176, 144);
      encoder.copyTextureToBuffer({ texture: e.sceneTexs[front] }, { buffer: textureReadback, bytesPerRow: rowBytes }, [width, height, 1]);
      e._fullscreen(encoder, e.pipe.present, e.kuwaharaEnabled ? e.bgPresentKuw
        : e.displayMode === "detail" ? e.bgPresentDetail : e.bgPresentScene[front], presentTexture.createView());
      encoder.copyTextureToBuffer({ texture: presentTexture }, { buffer: presentReadback, bytesPerRow: presentRowBytes }, [width, height, 1]);
      d.queue.submit([encoder.finish()]);
      await Promise.all([textureReadback.mapAsync(GPUMapMode.READ), metadata.mapAsync(GPUMapMode.READ), presentReadback.mapAsync(GPUMapMode.READ)]);
      const half = (bits) => {
        const sign = bits & 0x8000 ? -1 : 1, exponent = (bits >>> 10) & 31, mantissa = bits & 1023;
        return exponent === 31 ? (mantissa ? NaN : sign * Infinity)
          : sign * (exponent === 0 ? mantissa * 2 ** -24 : (1 + mantissa / 1024) * 2 ** (exponent - 15));
      };
      const texels = new Uint16Array(textureReadback.getMappedRange());
      let minimum = Infinity, maximum = -Infinity, positivePixels = 0, nonfiniteChannels = 0;
      for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
        let positive = false;
        for (let c = 0; c < 3; c++) {
          const value = half(texels[y * rowBytes / 2 + x * 4 + c]);
          if (!Number.isFinite(value)) { nonfiniteChannels++; continue; }
          minimum = Math.min(minimum, value); maximum = Math.max(maximum, value);
          positive ||= value > 0.001;
        }
        if (positive) positivePixels++;
      }
      const data = new Float32Array(metadata.getMappedRange());
      const presented = new Uint8Array(presentReadback.getMappedRange());
      let presentMaximum = 0, presentPositivePixels = 0;
      for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
        const i = y * presentRowBytes + x * 4;
        const value = Math.max(presented[i], presented[i + 1], presented[i + 2]);
        presentMaximum = Math.max(presentMaximum, value);
        if (value > 25) presentPositivePixels++;
      }
      return {
        offscreenPresent: { maximum: presentMaximum, positivePixels: presentPositivePixels },
        scene: { width, height, front, revision: e.sceneRevision, minimum, maximum, positivePixels, nonfiniteChannels },
        finalTransform: Array.from(data.slice(0, 16)), combinedFirst: Array.from(data.slice(16, 32)),
        boundsWithPadding: Array.from(data.slice(32, 44)), firstPositions: Array.from(data.slice(44)),
        renderUniformF32: e._cachedRenderU ? Array.from(new Float32Array(e._cachedRenderU.buffer)) : null,
        camera: { yaw: camera.yaw, pitch: camera.pitch, distance: camera.distance, target: [...camera.target] },
        mode: e.frameMode, accumulation: e._accumCount, batchSeed: e._batchSeed,
        gpuErrors: [...window.__gpuErrors],
      };
    } finally {
      for (const buffer of [textureReadback, packed, metadata, presentReadback]) buffer.destroy();
      presentTexture.destroy();
    }
  });
}

async function screenshot(name) {
  // Hide only overlays in the screenshot. A visible control panel must not
  // cause the pixel test to pass when the actual WebGPU image is blank.
  const png = await page.locator("#gpu").screenshot({
    path: resolve(artifacts, `${name}.png`),
    style: "#ui, #hint, #error { visibility: hidden !important; }",
  });
  const { data, width, height } = PNG.sync.read(png);
  let bright = 0;
  let darkest = 255;
  let brightest = 0;
  for (let i = 0; i < data.length; i += 4) {
    const value = Math.max(data[i], data[i + 1], data[i + 2]);
    if (value > 25) bright++;
    darkest = Math.min(darkest, value);
    brightest = Math.max(brightest, value);
  }
  if (bright <= width * height * 0.001 || brightest - darkest <= 25) {
    let diagnostics;
    try { diagnostics = await renderDiagnostics(); }
    catch (error) { diagnostics = { diagnosticError: error.stack || String(error) }; }
    await writeFile(resolve(artifacts, `${name}-render-diagnostics.json`), JSON.stringify(diagnostics, null, 2));
    console.log(`Render diagnostics ${name}: ${JSON.stringify(diagnostics)}`);
  }
  assert.ok(bright > width * height * 0.001, `${name}: fractal image is blank (${bright} bright pixels)`);
  assert.ok(brightest - darkest > 25, `${name}: image is uniform`);
  return { width, height, brightPixels: bright, range: brightest - darkest, hash: createHash("sha256").update(data).digest("hex") };
}

async function check(name, run) {
  const started = performance.now();
  console.log(`START ${name}`);
  let timeout;
  try {
    await Promise.race([
      run(),
      new Promise((_, reject) => {
        timeout = setTimeout(() => reject(new Error(`${name}: exceeded the 120s check budget`)), 120_000);
      }),
    ]);
  } finally {
    clearTimeout(timeout);
  }
  await healthy();
  checks.push({ name, durationMs: Math.round(performance.now() - started), state: await state() });
  console.log(`PASS  ${name} (${checks.at(-1).durationMs} ms)`);
}

async function slider(id, value) {
  await page.locator(`#${id}`).evaluate((el, next) => {
    el.value = String(next);
    el.dispatchEvent(new Event("input", { bubbles: true }));
  }, value);
}

try {
  browser = await chromium.launch({
    // Linux headless Chromium captures WebGPU canvases as black even when GPU
    // readbacks are correct. Use a real compositor surface (Xvfb in CI).
    channel: "chromium",
    headless: false,
    // Software shader compilation/execution can exceed Chromium's hardware
    // watchdog budget on a shared CI CPU. JS/GPU errors and suite timeouts
    // remain fatal; this does not disable WebGPU validation.
    args: [
      "--enable-unsafe-webgpu", "--use-gl=angle", "--use-angle=swiftshader",
      "--use-vulkan=swiftshader", "--enable-features=Vulkan",
      "--disable-vulkan-surface", "--enable-unsafe-swiftshader", "--disable-gpu-watchdog",
    ],
  });
  browserVersion = browser.version();
  console.log(`Chromium ${browserVersion}`);
  page = await browser.newPage({ viewport: { width: 640, height: 480 }, deviceScaleFactor: 1 });
  page.setDefaultTimeout(90_000);
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => {
    messages.push({ type: message.type(), text: message.text() });
    if (message.type() === "error") errors.push(message.text());
    if (message.text().startsWith("WebGPU adapter:")) console.log(message.text());
  });
  await page.addInitScript(() => {
    window.__gpuErrors = [];
    window.__gpuStartup = { userAgent: navigator.userAgent, adapters: [] };
    window.addEventListener("unhandledrejection", (event) => {
      window.__gpuErrors.push(`Unhandled rejection: ${event.reason?.message || event.reason}`);
    });
    // Attach before Engine.init() creates shaders/pipelines, not after the
    // debug handle appears. This catches early validation failures too.
    if (typeof GPUAdapter !== "undefined") {
      const requestDevice = GPUAdapter.prototype.requestDevice;
      GPUAdapter.prototype.requestDevice = async function (...args) {
        const info = this.info;
        const adapter = {
          vendor: info?.vendor, architecture: info?.architecture,
          device: info?.device, description: info?.description,
          fallback: this.isFallbackAdapter ?? info?.isFallbackAdapter,
          features: [...this.features],
          requiredFeatures: [...(args[0]?.requiredFeatures || [])],
          requiredLimits: args[0]?.requiredLimits,
          requestedAtMs: performance.now(),
        };
        window.__gpuStartup.adapters.push(adapter);
        console.info(`WebGPU adapter: ${JSON.stringify(adapter)}`);
        const device = await requestDevice.apply(this, args);
        adapter.createdAtMs = performance.now();
        device.addEventListener("uncapturederror", (event) => {
          window.__gpuErrors.push(event.error.message);
        });
        device.lost.then((info) => {
          adapter.lost = { reason: info.reason, message: info.message, atMs: performance.now() };
          if (info.reason !== "destroyed") window.__gpuErrors.push(`Device lost: ${info.message}`);
        });
        return device;
      };
    }
  });
  // Test-sized startup allocation avoids creating an initial 8M-particle
  // cloud before UI controls can lower the count. Most cases use a small grid;
  // a separate case below exercises the production 128^3 lighting grid.
  await page.goto(`${base}/?particles=32768&accum=131072&grid=32&profile=1`);
  await page.waitForFunction(() => Boolean(window.__app));
  await page.locator("#animate").uncheck();
  // Stop the startup procedural transition before numerical readbacks. A CPU
  // adapter otherwise accumulates hundreds of expensive morph frames in its
  // queue while those asynchronous readbacks wait for earlier submissions.
  await page.locator("#preset").selectOption("SierpinskiTriangle2D");

  await check("all WGSL modules compile", async () => {
    const compilation = await page.evaluate(async () => {
      const entries = await Promise.all(Object.entries(window.__app.engine.modules).map(async ([name, module]) => {
        const info = await module.getCompilationInfo();
        return [name, info.messages.map(({ type, message, lineNum, linePos }) => ({ type, message, lineNum, linePos }))];
      }));
      return Object.fromEntries(entries);
    });
    await writeFile(resolve(artifacts, "shader-compilation.json"), JSON.stringify(compilation, null, 2));
    assert.ok(Object.keys(compilation).length >= 11);
    for (const [name, diagnostics] of Object.entries(compilation)) {
      assert.deepEqual(diagnostics.filter((d) => d.type === "error"), [], `${name} WGSL errors`);
    }
  });

  await check("GPU chaos samples match the independent CPU oracle", async () => {
    const particles = 509; // Deliberately leaves padding in the final workgroup.
    const dispatch = dispatchShape(particles, 64, 3); // Exercise both dispatch dimensions.
    const fixtures = qualityFixtures.filter(({ name }) => ["triangle-2d", "pyramid-3d", "procedural-8"].includes(name));
    const cases = fixtures.map(({ name, matrices }) => ({
      name,
      matrices: matrices.map((m) => Array.from(m)),
      seeds: matrices.map(fixedPoint),
      steps: [
        { batch: 0, hops: fullDepth(matrices.length, particles), advance: false },
        { batch: 1, hops: 4, advance: true },
        { batch: Math.floor(0x100000000 / particles) - 1, hops: fullDepth(matrices.length, particles), advance: false },
      ],
    }));
    const results = await page.evaluate(async ({ cases, particles, dispatch }) => {
      const { device, pipe, bgl } = window.__app.engine;
      const results = [];
      for (const fixture of cases) {
        // Separate buffers prevent this numerical fixture from perturbing the
        // concurrently running website or requiring another GPU device.
        const positions = device.createBuffer({ size: particles * 12, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const transforms = device.createBuffer({ size: fixture.matrices.length * 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        const uniform = device.createBuffer({ size: 544, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const readback = device.createBuffer({ size: particles * 12, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        try {
          device.queue.writeBuffer(transforms, 0, new Float32Array(fixture.matrices.flat()));
          const group = device.createBindGroup({ layout: bgl.iter, entries: [
            { binding: 0, resource: { buffer: positions } },
            { binding: 1, resource: { buffer: transforms } },
            { binding: 2, resource: { buffer: uniform } },
          ] });
          const samples = [];
          for (const step of fixture.steps) {
            const bytes = new ArrayBuffer(544);
            new Uint32Array(bytes, 0, 6).set([fixture.matrices.length, particles, dispatch.width, step.hops, step.batch, Number(step.advance)]);
            fixture.seeds.forEach((seed, i) => new Float32Array(bytes, 32 + i * 16, 3).set(seed));
            device.queue.writeBuffer(uniform, 0, bytes);
            const encoder = device.createCommandEncoder();
            const pass = encoder.beginComputePass();
            pass.setPipeline(pipe.iterate);
            pass.setBindGroup(0, group);
            pass.dispatchWorkgroups(dispatch.x, dispatch.y);
            pass.end();
            encoder.copyBufferToBuffer(positions, 0, readback, 0, particles * 12);
            device.queue.submit([encoder.finish()]);
            await readback.mapAsync(GPUMapMode.READ);
            samples.push(Array.from(new Float32Array(readback.getMappedRange())));
            readback.unmap();
          }
          results.push({ name: fixture.name, samples });
        } finally {
          for (const resource of [positions, transforms, uniform, readback]) resource.destroy();
        }
      }
      return results;
    }, { cases, particles, dispatch });
    const comparisons = [];
    for (let c = 0; c < cases.length; c++) {
      const fixture = cases[c];
      let previous = null;
      for (let s = 0; s < fixture.steps.length; s++) {
        const step = fixture.steps[s];
        const expected = batchPoints(fixture.matrices, particles, step.batch, step.hops, step.advance ? previous : null);
        const actual = results[c].samples[s];
        assert.equal(actual.length, expected.length);
        let maximumError = 0;
        for (let i = 0; i < expected.length; i++) {
          assert.ok(Number.isFinite(actual[i]), `${fixture.name}: nonfinite GPU coordinate ${i}`);
          maximumError = Math.max(maximumError, Math.abs(actual[i] - expected[i]));
        }
        // CPU arithmetic and backend matrix operations can fuse/reorder sums;
        // tolerate float32 rounding, never changed random-transform choices.
        assert.ok(maximumError < 1e-5, `${fixture.name} batch ${step.batch}: GPU/CPU error ${maximumError}`);
        comparisons.push({ name: fixture.name, ...step, maximumError });
        previous = expected;
      }
    }
    await writeFile(resolve(artifacts, "gpu-sampling-oracle.json"), JSON.stringify({ particles, dispatch, comparisons }, null, 2));
  });

  for (const [preset, count] of [["SierpinskiTriangle2D", 3], ["SierpinskiTriangle3D", 5], ["SierpinskiCarpet3D", 20]]) {
    await check(`${preset}: renders and accumulates`, async () => {
      await page.locator("#preset").selectOption(preset);
      const current = await settled();
      assert.equal(current.transformCount, count);
      await screenshot(preset);
    });
  }

  await check("packed position buffer contains finite geometry", async () => {
    const geometry = await page.evaluate(async () => {
      const engine = window.__app.engine;
      const count = engine.particlesPerBatch;
      const bytes = count * 12;
      const readback = engine.device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const encoder = engine.device.createCommandEncoder();
      encoder.copyBufferToBuffer(engine.positionsBuf, 0, readback, 0, bytes);
      engine.device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const points = new Float32Array(readback.getMappedRange());
      let finite = true;
      let min = Infinity;
      let max = -Infinity;
      for (const value of points) {
        finite &&= Number.isFinite(value);
        min = Math.min(min, value);
        max = Math.max(max, value);
      }
      readback.unmap();
      readback.destroy();
      return { finite, min, max, bytes: engine.positionsBuf.size, expected: bytes };
    });
    assert.equal(geometry.bytes, geometry.expected);
    assert.equal(geometry.finite, true);
    assert.ok(geometry.max > geometry.min);
  });

  await page.locator("#preset").selectOption("SierpinskiTriangle3D");
  for (const theme of ["ivory", "ember", "gilded-lagoon", "amber-fern", "glacial-ember"]) {
    await check(`theme ${theme}`, async () => {
      await page.locator("#theme").selectOption(theme);
      await settled();
      assert.equal(await page.locator("#paletteAccents input").count(), theme.includes("-") ? 4 : 0);
      await screenshot(`theme-${theme}`);
    });
  }

  for (const mode of ["independent", "reuse"]) {
    await check(`accumulation ${mode}`, async () => {
      await page.locator("#refinement").selectOption(mode);
      // A camera change ensures an entire fresh accumulation cycle is tested.
      await page.evaluate(() => { window.__app.camera.yaw += 0.1; });
      await settled();
      assert.equal(await page.evaluate(() => window.__app.engine.accumulationMode), mode);
      await screenshot(`accumulation-${mode}`);
    });
  }

  await check("orbit, pan, and zoom controls redraw", async () => {
    const before = await page.evaluate(() => ({ yaw: window.__app.camera.yaw, distance: window.__app.camera.distance, target: [...window.__app.camera.target] }));
    const revision = (await state()).revision;
    // Keep the drag outside the control panel's hit area.
    await page.mouse.move(520, 200);
    await page.mouse.down();
    await page.mouse.move(590, 240, { steps: 4 });
    await page.mouse.up();
    await page.mouse.down({ button: "right" });
    await page.mouse.move(570, 230, { steps: 3 });
    await page.mouse.up({ button: "right" });
    await page.mouse.wheel(0, -150);
    await settled();
    const after = await page.evaluate(() => ({ yaw: window.__app.camera.yaw, distance: window.__app.camera.distance, target: [...window.__app.camera.target] }));
    assert.notEqual(after.yaw, before.yaw);
    assert.ok(after.distance < before.distance);
    assert.notDeepEqual(after.target, before.target);
    assert.ok((await state()).revision > revision);
    await screenshot("camera-controls");
  });

  await check("resize rebuilds framebuffer and post processing", async () => {
    await page.setViewportSize({ width: 720, height: 540 });
    await settled();
    const dimensions = await page.locator("#gpu").evaluate((canvas) => [canvas.width, canvas.height]);
    assert.deepEqual(dimensions, [720, 540]);
    await screenshot("resized");
    await page.setViewportSize({ width: 640, height: 480 });
    await settled();
  });

  await check("bloom and Kuwahara toggles invalidate cached effects", async () => {
    await slider("bloom", 0);
    await settled();
    const off = await screenshot("bloom-off");
    const before = (await state()).postStats;
    await slider("bloom", 2.2);
    await settled();
    assert.equal((await state()).postStats.bloom, before.bloom, "Reenabling an unchanged bloom should reuse its cached texture");
    const on = await screenshot("bloom-on");
    assert.notEqual(on.hash, off.hash, "Bloom intensity must still change the composite image");
    await page.evaluate(() => { window.__app.engine.bloomThreshold = 0.2; });
    await settled();
    assert.ok((await state()).postStats.bloom > before.bloom, "Changing the threshold must invalidate bloom");
    await page.locator("#kuwahara").check();
    await settled();
    assert.ok((await state()).postStats.kuwahara > before.kuwahara);
    const filtered = await screenshot("kuwahara-on");
    assert.notEqual(filtered.hash, off.hash, "Effect toggles must change the image");
    await slider("kuwSize", 6);
    await slider("kuwSharp", 4);
    await settled();
    await page.locator("#kuwahara").uncheck();
    await settled();
    await screenshot("kuwahara-off");
  });

  await check("idle frames reuse bloom and Kuwahara results", async () => {
    await page.locator("#kuwahara").check();
    const before = await settled();
    await page.evaluate(async () => {
      for (let i = 0; i < 8; i++) await new Promise(requestAnimationFrame);
      await window.__app.engine.device.queue.onSubmittedWorkDone();
    });
    const after = await state();
    assert.equal(after.mode, "idle");
    assert.equal(after.revision, before.revision);
    assert.deepEqual(after.postStats, before.postStats, "Idle frames must not rerun cached post chains");
    await page.locator("#kuwahara").uncheck();
  });

  await check("hidden accumulation batches reuse the visible post result", async () => {
    const frames = await page.evaluate(async () => {
      const { engine, camera } = window.__app;
      engine.accumTargetPoints = engine.particlesPerBatch * 16;
      camera.yaw += 0.01;
      const samples = [];
      for (let i = 0; i < 18; i++) {
        await new Promise(requestAnimationFrame);
        samples.push({ mode: engine.frameMode, revision: engine.sceneRevision, postStats: { ...engine.postStats } });
      }
      return samples;
    });
    let cachedFrames = 0;
    for (let i = 1; i < frames.length; i++) {
      if (frames[i].mode === "accumulate" && frames[i].revision === frames[i - 1].revision) {
        assert.deepEqual(frames[i].postStats, frames[i - 1].postStats);
        cachedFrames++;
      }
    }
    assert.ok(cachedFrames >= 2, "Exercise hidden accumulation between visible checkpoints");
    await settled();
    await page.evaluate(() => {
      window.__app.engine.accumTargetPoints = 131072;
      window.__app.camera.yaw += 0.01;
    });
    await settled();
  });

  await check("particle resizing and sampled lighting budget", async () => {
    await page.locator("#preset").selectOption("SierpinskiTriangle2D");
    await page.locator("#particles").selectOption("1048576");
    await page.locator("#lighting").selectOption("524288");
    await settled();
    assert.equal((await state()).particles, 1048576);
    assert.equal(await page.evaluate(() => window.__app.engine.lightingParticleBudget), 524288);
    await screenshot("sampled-lighting");
    await page.locator("#lighting").selectOption("full");
    await settled();
    assert.equal(await page.evaluate(() => window.__app.engine.lightingParticleBudget === Infinity), true);
    await screenshot("full-lighting");
    await page.locator("#particles").selectOption("32768");
    await settled();
  });

  await check("production 128 cubed lighting grid", async () => {
    await page.evaluate(() => { window.__app.engine.voxelGridDim = 128; });
    await settled();
    await screenshot("grid-128");
  });

  await check("morph animation updates geometry and returns to rest", async () => {
    await page.locator("#preset").selectOption("Procedural");
    await page.evaluate(() => { window.__app.engine.voxelGridDim = 32; });
    await page.locator("#animate").check();
    await page.locator("#randomize").click();
    const before = (await state()).revision;
    await page.waitForFunction((revision) => window.__app.engine.sceneRevision > revision + 3, before);
    assert.equal(await page.evaluate(() => window.__app.blender.animate), true);
    await page.locator("#animate").uncheck();
    // Pausing stops new targets; the current morph still finishes normally.
    // Raise the existing speed control so this correctness test need not wait
    // several seconds for the asymptotic animation to settle on a CPU adapter.
    await slider("speed", 6);
    await settled();
    await screenshot("procedural-paused");
  });

  await check("optional GPU timing returns finite pass measurements", async () => {
    const supported = await page.evaluate(() => window.__app.engine.profiler.supported);
    if (supported) {
      await page.waitForFunction(() => window.__app.engine.profiler.latest !== null);
      const profile = await page.evaluate(() => {
        const profiler = window.__app.engine.profiler;
        return { latest: profiler.latest, failed: profiler.failedSamples };
      });
      assert.equal(profile.failed, 0);
      assert.ok(Number.isFinite(profile.latest.totalMs) && profile.latest.totalMs >= 0);
      assert.ok(Object.keys(profile.latest.passes).length > 0);
      for (const ms of Object.values(profile.latest.passes)) assert.ok(Number.isFinite(ms) && ms >= 0);
      await writeFile(resolve(artifacts, "gpu-profile.json"), JSON.stringify(profile, null, 2));
    } else {
      assert.equal(await page.evaluate(() => window.__app.engine.profiler.enabled), false);
      console.log("  Adapter does not expose timestamp-query; rendering fallback remains healthy.");
    }
  });

  await runViewChecks({ page, check, settled, screenshot, slider, artifacts });

  await check("unavailable WebGPU presents the existing error screen", async () => {
    const unsupported = await browser.newPage({ viewport: { width: 640, height: 480 } });
    try {
      await unsupported.addInitScript(() => Object.defineProperty(navigator, "gpu", { value: undefined }));
      await unsupported.goto(base);
      await unsupported.locator("#error").waitFor({ state: "visible" });
      assert.match(await unsupported.locator("#error p").textContent(), /WebGPU is not available/);
      assert.equal(await unsupported.evaluate(() => Boolean(window.__app)), false);
      await unsupported.screenshot({ path: resolve(artifacts, "webgpu-unavailable.png") });
    } finally {
      await unsupported.close();
    }
  });

  await healthy();
  console.log(`Completed ${checks.length} WebGPU regression checks.`);
} catch (error) {
  errors.push(error.stack || String(error));
  if (page) {
    try { await page.screenshot({ path: resolve(artifacts, "failure.png") }); } catch { /* Keep the original failure. */ }
  }
  console.error(error);
  process.exitCode = 1;
} finally {
  let gpuErrors = [];
  let startup;
  if (page) {
    try { gpuErrors = await page.evaluate(() => window.__gpuErrors); } catch { /* Page may have crashed. */ }
    try { startup = await page.evaluate(() => window.__gpuStartup); } catch { /* Page may have crashed. */ }
  }
  await writeFile(resolve(artifacts, "report.json"), JSON.stringify({ browserVersion, startup, checks, errors, gpuErrors, messages }, null, 2));
  await browser?.close();
  await new Promise((done) => server.close(done));
}
