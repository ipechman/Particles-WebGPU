import assert from "node:assert/strict";
import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { PNG } from "pngjs";
import { runFocusChecks } from "./browser-focus-checks.mjs";

export async function runViewChecks({ page, check, settled, screenshot, slider, artifacts }) {
  const readPNG = async name => PNG.sync.read(await readFile(resolve(artifacts, `${name}.png`)));
  const clipped = png => {
    let count = 0;
    for (let i = 0; i < png.data.length; i += 4) if (Math.max(...png.data.subarray(i, i + 3)) === 255) count++;
    return count;
  };

  await runFocusChecks({ page, check, settled, screenshot, slider, artifacts });

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
    assert.equal(await page.evaluate(() => window.__app.engine._viewActive), true);
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
