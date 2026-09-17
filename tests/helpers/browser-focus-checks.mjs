import assert from "node:assert/strict";
import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { PNG } from "pngjs";
import { qualityFixtures, batchPoints, fullDepth } from "./sampling.mjs";
import { transformPoint } from "./view-quality.mjs";
import { mat4 } from "../../web/js/math.js";
import { multiply64, packViewDraws } from "../../web/js/view-sampling.js";

export async function runFocusChecks({ page, check, settled, screenshot, slider, artifacts }) {
  await check("GPU worldPoint matches independent coordinates and all child maps", async () => {
    const comparisons = [];
    const n = 509; // Unaligned buffer boundaries must not lose or duplicate a range.
    for (const { name, matrices } of qualityFixtures.filter(f => ["triangle-2d", "pyramid-3d", "procedural-8"].includes(f.name))) {
      const base = batchPoints(matrices, n, 37, fullDepth(matrices.length, n));
      const fit = mat4.identity(); fit[0] = fit[5] = fit[10] = 1.3; fit[12] = -0.2;
      const leaves = [matrices[0], matrices[1], multiply64(matrices[0], matrices[2])].map(matrix => ({ matrix }));
      const packed = packViewDraws(leaves, n, matrices.length, fit, [1, 2, 3]);
      const queries = [], expected = [];
      packed.draws.forEach((draw, iid) => {
        const matrix = new Float32Array(packed.bytes, iid * 80, 16);
        for (let vid = draw.first; vid < draw.first + draw.count; vid++) {
          queries.push(vid, iid);
          const child = matrices[(vid + draw.copy) % matrices.length];
          const inner = transformPoint(child, base.subarray(vid * 3, vid * 3 + 3)).map(Math.fround);
          expected.push(...transformPoint(matrix, inner), 1);
        }
      });
      const combined = matrices.map(m => multiply64(fit, m));
      const actual = await page.evaluate(async ({ base, matrices, combined, draws, queries, count }) => {
        const e = window.__app.engine, d = e.device;
        const source = await (await fetch("shaders/render.wgsl")).text();
        // Invoke the same worldPoint function the production vertex entry uses.
        // The numerical oracle on the Node side is independent of this WGSL.
        const module = d.createShaderModule({ code: source + `
          @group(1) @binding(0) var<storage,read> queries:array<vec2<u32>>;
          @group(1) @binding(1) var<storage,read_write> results:array<vec4<f32>>;
          @compute @workgroup_size(64) fn verify(@builtin(global_invocation_id) id:vec3<u32>) {
            if(id.x<arrayLength(&queries)) { results[id.x]=worldPoint(queries[id.x].x,queries[id.x].y); }
          }` });
        const pipeline = d.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "verify" } });
        const resources = [];
        const upload = (data, usage = GPUBufferUsage.STORAGE) => {
          const b = d.createBuffer({ size: data.byteLength, usage: usage | GPUBufferUsage.COPY_DST });
          resources.push(b); d.queue.writeBuffer(b, 0, data); return b;
        };
        try {
          const u = new ArrayBuffer(224); new Uint32Array(u, 100, 1)[0] = count; new Uint32Array(u, 120, 1)[0] = 1;
          const buffers = new Map([
            [0, upload(new Float32Array(base))], [1, upload(new Float32Array(combined.flat()))],
            [4, upload(new Uint8Array(u), GPUBufferUsage.UNIFORM)],
            [6, upload(new Uint8Array(draws))], [7, upload(new Float32Array(matrices.flat()))],
          ]);
          const size = queries.length / 2 * 16;
          const q = upload(new Uint32Array(queries));
          const result = d.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
          const readback = d.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
          resources.push(result, readback);
          const g0 = d.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [...buffers].map(([binding, buffer]) => ({ binding, resource: { buffer } })) });
          const g1 = d.createBindGroup({ layout: pipeline.getBindGroupLayout(1), entries: [q, result].map((buffer, binding) => ({ binding, resource: { buffer } })) });
          const enc = d.createCommandEncoder(), pass = enc.beginComputePass();
          pass.setPipeline(pipeline); pass.setBindGroup(0, g0); pass.setBindGroup(1, g1);
          pass.dispatchWorkgroups(Math.ceil(queries.length / 128)); pass.end();
          enc.copyBufferToBuffer(result, 0, readback, 0, size); d.queue.submit([enc.finish()]);
          await readback.mapAsync(GPUMapMode.READ);
          return Array.from(new Float32Array(readback.getMappedRange()));
        } finally { resources.forEach(r => r.destroy()); }
      }, { base: Array.from(base), matrices: matrices.map(m => Array.from(m)), combined: combined.map(m => Array.from(m)),
        draws: Array.from(new Uint8Array(packed.bytes)), queries, count: matrices.length });
      let maxError = 0;
      actual.forEach((v, i) => { assert.ok(Number.isFinite(v)); maxError = Math.max(maxError, Math.abs(v - expected[i])); });
      assert.ok(maxError < 2e-5, `${name}: ${maxError}`);
      comparisons.push({ name, particles: n, vertices: queries.length / 2, maxError });
    }
    await writeFile(resolve(artifacts, "focus-coordinate-oracle.json"), JSON.stringify(comparisons, null, 2));
  });

  const fixtures = [
    ["carpet-2d", "SierpinskiCarpet2D", 1], ["pyramid-3d", "SierpinskiTriangle3D", 1],
    ["procedural-3", "Procedural", 2], ["procedural-8", "Procedural", 2],
  ];
  const comparisons = [];
  await page.setViewportSize({ width: 1280, height: 720 });
  for (const [name, preset, distance] of fixtures) {
    await check(`Focus changes a normal centered wheel zoom at 262K: ${name}`, async () => {
      await page.locator("#animate").uncheck();
      await page.locator("#preset").selectOption(preset);
      if (preset === "Procedural") {
        // Frozen ordinary procedural forms make the regression reproducible.
        // Camera orientation/target remain the startup values for every case.
        const fixture = qualityFixtures.find(f => f.name === name);
        await page.evaluate(instructions => window.__app.blender.loadShape(instructions), fixture.instructions);
      }
      await page.locator("#particles").selectOption("262144");
      await page.locator("#refinePasses").selectOption("2");
      await page.locator("#refinement").selectOption("independent");
      await page.locator("#refinePasses").selectOption("1");
      await page.locator("#displayMode").selectOption("classic");
      await page.locator("#theme").selectOption("ivory");
      await slider("bloom", 0);
      await page.locator("#kuwahara").uncheck();
      for (const id of ["pColor", "oColor"]) await page.locator(`#${id}`).evaluate(el => {
        el.value = "#ffffff"; el.dispatchEvent(new Event("input", { bubbles: true }));
      });
      await page.evaluate(() => {
        const { camera: c } = window.__app;
        c.target = [0, 0, 0]; c.yaw = 0.6; c.pitch = 0.45; c.distance = 4.5;
      });
      await page.locator("#sampling").selectOption("view"); await settled();
      assert.equal(await page.locator("#samplingStatus").textContent(), "Focus active", "Focus must already work at the default camera");
      await page.mouse.move(1100, 400);
      await page.mouse.wheel(0, Math.log(distance / 4.5) / 0.0015);
      await settled();
      const initial = await page.evaluate(() => {
        const e = window.__app.engine;
        window.__focusTestBuffer = e.positionsBuf;
        return { fit: Array.from(e._fitCPU), seed: e._batchSeed, particles: e.particlesPerBatch };
      });
      await page.locator("#sampling").selectOption("global"); await settled();
      assert.equal(await page.locator("#samplingStatus").textContent(), "Global sampling selected");
      const global = await screenshot(`normal-${name}-global`);
      await page.locator("#sampling").selectOption("view");
      const current = await settled();
      assert.equal(await page.locator("#samplingStatus").textContent(), "Focus active");
      const focused = await screenshot(`normal-${name}-focus`);
      assert.equal(current.view.vertices, 262144 * current.transformCount, "Focus must preserve the full drawing budget");
      assert.notEqual(focused.hash, global.hash, "The Sampling control must change the rendered geometry");
      assert.ok(focused.brightPixels > global.brightPixels * 1.025, `${name}: ${focused.brightPixels} vs ${global.brightPixels}`);
      const after = await page.evaluate(() => ({ fit: Array.from(window.__app.engine._fitCPU), seed: window.__app.engine._batchSeed,
        sameBuffer: window.__app.engine.positionsBuf === window.__focusTestBuffer }));
      assert.deepEqual(after.fit, initial.fit); assert.equal(after.seed, initial.seed); assert.equal(after.sameBuffer, true);

      await page.locator("#sampling").selectOption("global");
      await page.locator("#refinePasses").selectOption("8");
      await settled(); await screenshot(`normal-${name}-reference`);
      const ref = PNG.sync.read(await readFile(resolve(artifacts, `normal-${name}-reference.png`)));
      const f = PNG.sync.read(await readFile(resolve(artifacts, `normal-${name}-focus.png`)));
      const mask = png => Uint8Array.from({ length: png.width * png.height }, (_, i) =>
        Number(Math.max(...png.data.subarray(i * 4, i * 4 + 3)) > 25));
      const a = mask(ref), b = mask(f); let supported = 0;
      for (let y = 0; y < f.height; y++) for (let x = 0; x < f.width; x++) if (b[y * f.width + x]) {
        let found = false;
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
          if (x + dx >= 0 && x + dx < f.width && y + dy >= 0 && y + dy < f.height && a[(y + dy) * f.width + x + dx]) found = true;
        }
        supported += Number(found);
      }
      assert.ok(supported / focused.brightPixels > 0.95, "Focused geometry must agree with the denser reference");
      const result = { name, viewport: [1280, 720], distance, target: [0, 0, 0], yaw: 0.6, pitch: 0.45,
        particles: 262144, batches: 1, verticesPerMode: current.view.vertices, globalPixels: global.brightPixels,
        focusedPixels: focused.brightPixels, coverageRatio: focused.brightPixels / global.brightPixels,
        referenceBatches: 8, referenceAgreement: supported / focused.brightPixels, planningMs: current.view.planningMs };
      comparisons.push(result); console.log(JSON.stringify(result));
      await writeFile(resolve(artifacts, "normal-focus-quality.json"), JSON.stringify(comparisons, null, 2));
    });
  }

  await check("paused normal view visibly changes with the actual palette and Detail display", async () => {
    await page.locator("#theme").selectOption("gilded-lagoon");
    await page.locator("#displayMode").selectOption("detail"); await slider("bloom", 0.25);
    await page.locator("#refinePasses").selectOption("1");
    await page.locator("#sampling").selectOption("global"); await settled();
    const global = await screenshot("normal-shaded-global");
    await page.locator("#sampling").selectOption("view"); await settled();
    const focus = await screenshot("normal-shaded-focus");
    assert.notEqual(global.hash, focus.hash);
  });
  await page.locator("#particles").selectOption("32768");
  await page.locator("#refinePasses").selectOption("auto");
  await page.setViewportSize({ width: 640, height: 480 }); await settled();
}
