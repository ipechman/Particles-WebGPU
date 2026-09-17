import assert from "node:assert/strict";
import { writeFile } from "node:fs/promises";
import { resolve } from "node:path";

export async function runRefinementChecks({ page, check, settled, screenshot, slider, artifacts }) {
  await check("public refinement budget adds visible samples and gates the batch method", async () => {
    await page.locator("#preset").selectOption("SierpinskiCarpet2D");
    await page.locator("#sampling").selectOption("global");
    await page.locator("#theme").selectOption("ivory");
    await page.locator("#displayMode").selectOption("classic");
    await slider("bloom", 0);
    for (const id of ["pColor", "oColor"]) await page.locator(`#${id}`).evaluate(el => {
      el.value = "#ffffff"; el.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await page.locator("#refinePasses").selectOption("1");
    await page.mouse.move(520, 200);
    await page.mouse.wheel(0, Math.log(1 / 4.5) / 0.0015);
    const one = await settled();
    assert.equal(one.target, 1); assert.equal(one.accumulation, 1);
    assert.equal(await page.locator("#refinement").isEnabled(), false);
    assert.match(await page.locator("#refinementStatus").textContent(), /Single pass/);
    const sparse = await screenshot("refinement-one-pass");

    // Observe the uniforms consumed by the real compute passes. No sampling,
    // camera, budget, or pause state is altered through this instrumentation.
    await page.evaluate(() => {
      const e = window.__app.engine, original = e._encodeIterate.bind(e);
      window.__refinementBuffer = e.positionsBuf;
      window.__refinementTrace = [];
      e._encodeIterate = enc => {
        const u = new Uint32Array(e.uChaosCPU);
        window.__refinementTrace.push({ particles: u[1], hops: u[3], seed: u[4], advance: u[5] });
        original(enc);
      };
      window.__restoreRefinementTrace = () => { e._encodeIterate = original; };
    });
    const traces = {};
    try {
      await page.locator("#animate").check();
      await page.locator("#refinePasses").selectOption("4");
      const four = await settled();
      assert.equal(await page.locator("#animate").isChecked(), false, "Selecting refinement must pause for inspection");
      assert.equal(four.accumulation, 4); assert.equal(four.target, 4);
      assert.equal(await page.locator("#refinement").isEnabled(), true);
      assert.equal(await page.locator("#refinementStatus").textContent(), "Complete · 4 passes");
      traces.fast = await page.evaluate(() => window.__refinementTrace);
      assert.equal(traces.fast.length, 3);
      assert.ok(traces.fast.every(p => p.hops === 4 && p.advance === 1));
      const dense = await screenshot("refinement-four-passes");
      assert.notEqual(dense.hash, sparse.hash);
      assert.ok(dense.brightPixels > sparse.brightPixels * 1.05, `${dense.brightPixels} vs ${sparse.brightPixels}`);

      await page.evaluate(() => { window.__refinementTrace.length = 0; });
      await page.locator("#refinement").selectOption("independent");
      await settled();
      traces.independent = await page.evaluate(() => window.__refinementTrace);
      assert.equal(traces.independent.length, 4);
      assert.ok(traces.independent.every(p => p.hops > 4 && p.advance === 0));
      assert.equal(await page.evaluate(() => window.__app.engine.positionsBuf === window.__refinementBuffer), true);
      await page.screenshot({ path: resolve(artifacts, "refinement-controls.png") });

      await page.locator("#refinePasses").selectOption("1");
      assert.equal((await settled()).accumulation, 1, "Lowering the budget must discard extra accumulated passes");
      assert.equal(await page.locator("#refinement").isEnabled(), false);
      await writeFile(resolve(artifacts, "refinement-controls.json"), JSON.stringify({ particles: one.particles,
        onePassPixels: sparse.brightPixels, fourPassPixels: dense.brightPixels,
        coverageRatio: dense.brightPixels / sparse.brightPixels, traces }, null, 2));
    } finally {
      await page.evaluate(() => window.__restoreRefinementTrace());
    }
    await page.locator("#refinePasses").selectOption("auto");
    await page.locator("#sampling").selectOption("view");
    await page.locator("#displayMode").selectOption("detail");
    await page.locator("#theme").selectOption("ivory");
    await slider("bloom", 0.25);
    await page.mouse.move(520, 200);
    await page.mouse.wheel(0, Math.log(4.5) / 0.0015);
    await settled();
  });
}
