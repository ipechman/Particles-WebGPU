import assert from "node:assert/strict";
import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { PNG } from "pngjs";
import { qualityFixtures, fixedPoint } from "./sampling.mjs";
import { focusedPoints } from "./view-quality.mjs";
import { multiply64, packViewLeaves } from "../../web/js/view-sampling.js";

export async function runViewChecks({ page, check, settled, screenshot, slider, artifacts }) {
  const readPNG = async name => PNG.sync.read(await readFile(resolve(artifacts, `${name}.png`)));
  const mask = png => Uint8Array.from({ length: png.width * png.height }, (_, i) =>
    Number(Math.max(...png.data.subarray(i * 4, i * 4 + 3)) > 25));
  const clipped = png => {
    let count = 0;
    for (let i = 0; i < png.data.length; i += 4) if (Math.max(...png.data.subarray(i, i + 3)) === 255) count++;
    return count;
  };

  await check("view shader matches CPU points across prefixes and padded 2D dispatch", async () => {
    const particles = 509;
    const comparisons = [];
    for (const fixture of qualityFixtures.filter(f => ["triangle-2d", "pyramid-3d", "procedural-8"].includes(f.name))) {
      const leaves = [fixture.matrices[0], fixture.matrices[1], multiply64(fixture.matrices[0], fixture.matrices[2])]
        .map(matrix => ({ matrix, area: 100 }));
      const packed = packViewLeaves(leaves, particles);
      for (const batch of [0, Math.floor(0x100000000 / particles) - 1]) {
        const expected = focusedPoints(fixture.matrices, { leaves }, particles, batch);
        const actual = await page.evaluate(async ({ matrices, seeds, packed, particles, batch }) => {
          const e = window.__app.engine, d = e.device;
          const buffers = [
            d.createBuffer({ size: particles * 12, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC }),
            d.createBuffer({ size: matrices.length * 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }),
            d.createBuffer({ size: 544, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
            d.createBuffer({ size: packed.length, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }),
            d.createBuffer({ size: particles * 12, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
          ];
          try {
            d.queue.writeBuffer(buffers[1], 0, new Float32Array(matrices.flat()));
            d.queue.writeBuffer(buffers[3], 0, new Uint8Array(packed));
            const u = new ArrayBuffer(544);
            const hops = Math.min(40, Math.max(12, Math.ceil(Math.log(particles) / Math.log(matrices.length)) + 8));
            new Uint32Array(u, 0, 8).set([matrices.length, particles, 192, hops, batch, 0, 3, 0]);
            seeds.forEach((p, i) => new Float32Array(u, 32 + i * 16, 3).set(p));
            d.queue.writeBuffer(buffers[2], 0, u);
            const bg = d.createBindGroup({ layout: e.bgl.viewIter, entries: buffers.slice(0, 4)
              .map((buffer, binding) => ({ binding, resource: { buffer } })) });
            const enc = d.createCommandEncoder(), pass = enc.beginComputePass();
            pass.setPipeline(e.pipe.iterateView); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(3, 3); pass.end();
            enc.copyBufferToBuffer(buffers[0], 0, buffers[4], 0, particles * 12);
            d.queue.submit([enc.finish()]);
            await buffers[4].mapAsync(GPUMapMode.READ);
            return Array.from(new Float32Array(buffers[4].getMappedRange()));
          } finally { buffers.forEach(b => b.destroy()); }
        }, { matrices: fixture.matrices.map(m => Array.from(m)), seeds: fixture.matrices.map(fixedPoint),
          packed: Array.from(new Uint8Array(packed)), particles, batch });
        let maxError = 0;
        actual.forEach((value, i) => {
          assert.ok(Number.isFinite(value));
          maxError = Math.max(maxError, Math.abs(value - expected[i]));
        });
        assert.ok(maxError < 2e-5, `${fixture.name}: ${maxError}`);
        comparisons.push({ name: fixture.name, batch, maxError });
      }
    }
    await writeFile(resolve(artifacts, "view-gpu-oracle.json"), JSON.stringify(comparisons, null, 2));
  });

  await check("depth detail preserves flat surfaces and background while shading a depth step", async () => {
    const result = await page.evaluate(async () => {
      const e = window.__app.engine, d = e.device;
      const rt = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
      const scene = d.createTexture({ size: [32, 32], format: "rgba16float", usage: rt });
      const depth = d.createTexture({ size: [32, 32], format: "depth24plus", usage: rt });
      const out = d.createTexture({ size: [32, 32], format: "rgba16float", usage: rt | GPUTextureUsage.COPY_SRC });
      const u = d.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const readback = d.createBuffer({ size: 8192, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      try {
        const module = d.createShaderModule({ code: `
          @vertex fn vs(@builtin(vertex_index) id:u32) -> @builtin(position) vec4<f32> {
            return vec4<f32>(vec2<f32>(f32((id<<1u)&2u),f32(id&2u))*2.0-1.0,0.0,1.0);
          }
          struct Out { @location(0) color:vec4<f32>, @builtin(frag_depth) depth:f32 }
          @fragment fn fs(@builtin(position) p:vec4<f32>) -> Out {
            var o:Out;
            o.color=vec4<f32>(0.5,0.5,0.5,1.0);
            let distance=select(2.0,1.8,p.x>=16.0);
            o.depth=100.0/99.9-10.0/(99.9*distance);
            if(p.x<4.0) { o.color=vec4<f32>(0.0,0.0,0.0,1.0); o.depth=1.0; }
            return o;
          }` });
        const pipe = d.createRenderPipeline({ layout: "auto", vertex: { module, entryPoint: "vs" },
          fragment: { module, entryPoint: "fs", targets: [{ format: "rgba16float" }] },
          depthStencil: { format: "depth24plus", depthCompare: "always", depthWriteEnabled: true } });
        const bg = d.createBindGroup({ layout: e.bgl.detail, entries: [
          { binding: 0, resource: scene.createView() }, { binding: 1, resource: depth.createView() },
          { binding: 2, resource: { buffer: u } },
        ] });
        d.queue.writeBuffer(u, 0, new Float32Array([0.1, 100, 2 * Math.tan(Math.PI / 6) / 32, 1, 0, 0, 0, 0]));
        const enc = d.createCommandEncoder();
        const pass = enc.beginRenderPass({ colorAttachments: [{ view: scene.createView(), loadOp: "clear", storeOp: "store" }],
          depthStencilAttachment: { view: depth.createView(), depthLoadOp: "clear", depthStoreOp: "store", depthClearValue: 1 } });
        pass.setPipeline(pipe); pass.draw(3); pass.end();
        e._fullscreen(enc, e.pipe.detail, bg, out.createView());
        enc.copyTextureToBuffer({ texture: out }, { buffer: readback, bytesPerRow: 256 }, [32, 32]);
        d.queue.submit([enc.finish()]); await readback.mapAsync(GPUMapMode.READ);
        const values = new Uint16Array(readback.getMappedRange());
        const half = bits => (bits >>> 10) === 0 ? (bits & 1023) * 2 ** -24 : (1 + (bits & 1023) / 1024) * 2 ** ((bits >>> 10) - 15);
        const at = x => half(values[(16 * 32 + x) * 4]);
        return { background: at(1), flat: at(9), cavity: at(14), foreground: at(23) };
      } finally { [scene, depth, out, u, readback].forEach(r => r.destroy()); }
    });
    assert.equal(result.background, 0);
    assert.equal(result.flat, 0.5);
    assert.equal(result.foreground, 0.5);
    assert.ok(result.cavity < 0.49 && result.cavity > 0.2);
    await writeFile(resolve(artifacts, "depth-detail-oracle.json"), JSON.stringify(result, null, 2));
  });

  // Geometry comparisons disable post effects and use constant point color.
  // Global gets its original M instanced copies; focused draws exactly N.
  const comparisons = [];
  for (const [preset, offset] of [
    ["SierpinskiTriangle2D", [0.015, 0.01, 0.09]],
    ["SierpinskiTriangle3D", [0.045, 0.045, 0.09]],
    ["Vicsek3D", [0.015, 0.01, 0.09]],
  ]) {
    await check(`equal particle budget close-up and 32x reference: ${preset}`, async () => {
      await page.locator("#sampling").selectOption("view");
      await page.locator("#preset").selectOption(preset);
      await page.locator("#displayMode").selectOption("classic");
      await page.locator("#refinement").selectOption("independent");
      await page.evaluate(() => {
        const { engine: e, camera: c } = window.__app;
        e.particlesPerBatch = 32768; e.accumTargetPoints = 32768;
        e.bloomIntensity = 0; e.kuwaharaEnabled = false; e.paletteStops = [];
        e.particleColor = [1, 1, 1]; e.occlusionColor = [1, 1, 1]; e.backgroundColor = [0, 0, 0];
        c.distance = 4.5; c.target = [0, 0, 0]; c.yaw = 0.6; c.pitch = 0.45;
      });
      await settled();
      await page.evaluate(offset => {
        const { engine: e, camera: c } = window.__app;
        const seed = e._fixedPoints(window.__app.blender.getTransformCount());
        const fit = e._fitCPU;
        c.target = [0, 1, 2].map(r => fit[r] * seed[0] + fit[4 + r] * seed[1] + fit[8 + r] * seed[2] + fit[12 + r]);
        const length = Math.hypot(...offset);
        c.distance = fit[0] * length; c.yaw = Math.atan2(offset[2], offset[0]); c.pitch = Math.asin(offset[1] / length);
      }, offset);
      await settled();
      assert.equal(await page.evaluate(() => window.__app.engine._viewActive), true);
      const fit = await page.evaluate(() => Array.from(window.__app.engine._fitCPU));
      await page.locator("#sampling").selectOption("global");
      await settled();
      const global = await screenshot(`${preset}-close-global`);
      await page.locator("#sampling").selectOption("view");
      const focusedState = await settled();
      const focused = await screenshot(`${preset}-close-focused`);
      assert.equal(focusedState.particles, 32768);
      assert.equal(focusedState.accumulation, 1);
      assert.equal(focusedState.view.active, true);
      assert.ok(focused.brightPixels > global.brightPixels * 1.5, `${focused.brightPixels} vs ${global.brightPixels}`);
      assert.deepEqual(await page.evaluate(() => Array.from(window.__app.engine._fitCPU)), fit, "Focusing must not change fit");
      await page.locator("#sampling").selectOption("global");
      await page.evaluate(() => { window.__app.engine.accumTargetPoints = 32768 * 32; });
      await settled();
      await screenshot(`${preset}-close-reference`);
      const a = mask(await readPNG(`${preset}-close-global`));
      const b = mask(await readPNG(`${preset}-close-focused`));
      const refPNG = await readPNG(`${preset}-close-reference`), reference = mask(refPNG);
      const nearReference = i => {
        const x = i % refPNG.width, y = Math.floor(i / refPNG.width);
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
          if (x + dx >= 0 && x + dx < refPNG.width && y + dy >= 0 && y + dy < refPNG.height &&
              reference[(y + dy) * refPNG.width + x + dx]) return true;
        }
        return false;
      };
      let globalSupported = 0, focusedSupported = 0;
      for (let i = 0; i < a.length; i++) if (nearReference(i)) { globalSupported += a[i]; focusedSupported += b[i]; }
      assert.ok(focusedSupported / focused.brightPixels > 0.95, "Focused pixels must agree with the dense reference within one pixel");
      assert.ok(focusedSupported > globalSupported * 1.5);
      const result = { preset, offset, fit, particles: 32768, batches: 1, globalVertices: 32768 * focusedState.transformCount,
        focusedVertices: 32768, globalPixels: global.brightPixels, focusedPixels: focused.brightPixels,
        coverageRatio: focused.brightPixels / global.brightPixels, referenceBatches: 32,
        focusedReferenceAgreement: focusedSupported / focused.brightPixels, planning: focusedState.view };
      comparisons.push(result);
      console.log(JSON.stringify(result));
      await writeFile(resolve(artifacts, "view-quality.json"), JSON.stringify(comparisons, null, 2));
    });
  }

  await check("focused accumulation, camera changes, resize, and global return preserve the fit", async () => {
    await page.locator("#sampling").selectOption("view");
    await page.evaluate(() => { window.__app.engine.accumTargetPoints = 32768 * 4; });
    await settled();
    const fit = await page.evaluate(() => Array.from(window.__app.engine._fitCPU));
    await page.locator("#refinement").selectOption("reuse");
    await settled();
    const refit = await page.evaluate(() => Array.from(window.__app.engine._fitCPU));
    // Changing refinement intentionally rebuilds the global reference once.
    assert.ok(refit.every(Number.isFinite) && fit.every(Number.isFinite));
    await page.evaluate(() => { const c = window.__app.camera; c.yaw += 0.02; c.distance *= 0.95; c.target[0] += c.distance * 0.01; });
    await settled();
    assert.deepEqual(await page.evaluate(() => Array.from(window.__app.engine._fitCPU)), refit);
    await page.setViewportSize({ width: 720, height: 540 }); await settled();
    assert.deepEqual(await page.evaluate(() => Array.from(window.__app.engine._fitCPU)), refit);
    await screenshot("focused-resize");
    await page.setViewportSize({ width: 640, height: 480 }); await settled();
    await page.evaluate(() => { window.__app.camera.target = [100, 100, 100]; });
    await settled();
    assert.equal(await page.evaluate(() => window.__app.engine._viewPlan.leaves.length), 0);
    await page.evaluate(() => { const c = window.__app.camera; c.target = [0, 0, 0]; c.distance = 4.5; });
    await settled();
    assert.equal(await page.evaluate(() => window.__app.engine._viewActive), false);
    await screenshot("focused-return-overview");
  });

  await check("detail display preserves highlights and caches depth shading", async () => {
    await page.locator("#preset").selectOption("SierpinskiTriangle3D");
    await page.locator("#theme").selectOption("gilded-lagoon");
    await page.locator("#displayMode").selectOption("classic");
    await slider("bloom", 2.2);
    await page.evaluate(() => {
      const { engine: e, camera: c } = window.__app;
      e.bloomThreshold = 0.4; c.target = [0, 0, 0]; c.distance = 4.5; c.yaw = 0.6; c.pitch = 0.45;
    });
    await settled();
    await screenshot("detail-compare-classic");
    const before = clipped(await readPNG("detail-compare-classic"));
    assert.ok(before > 100);
    // Same bloom tests tone mapping independently of the lower new default.
    await page.locator("#displayMode").selectOption("detail");
    await settled();
    await screenshot("detail-compare-same-bloom");
    const compressed = clipped(await readPNG("detail-compare-same-bloom"));
    assert.ok(compressed < before * 0.1, `${compressed} vs ${before} clipped pixels`);
    await slider("bloom", 0.25);
    const initial = await settled();
    const display = await screenshot("detail-compare-default");
    const after = clipped(await readPNG("detail-compare-default"));
    assert.equal(after, 0);
    await slider("exposure", 1.3); await settled();
    assert.notEqual((await screenshot("detail-exposure")).hash, display.hash);
    const stats = await page.evaluate(() => ({ ...window.__app.engine.postStats }));
    assert.equal(stats.detail, initial.postStats.detail, "Exposure only recomposites the cached image");
    await page.evaluate(async () => { for (let i = 0; i < 5; i++) await new Promise(requestAnimationFrame); });
    assert.deepEqual(await page.evaluate(() => ({ ...window.__app.engine.postStats })), stats);
    await writeFile(resolve(artifacts, "detail-quality.json"), JSON.stringify({ before, sameBloom: compressed, newDefault: after }, null, 2));
  });
}
