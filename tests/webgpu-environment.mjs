// Temporary CI environment probe. This is deliberately NOT a regression gate:
// each configuration records failures and the probe exits successfully.
import { createServer } from "node:http";
import { chromium } from "playwright";
import { PNG } from "pngjs";

const server = createServer((req, res) => {
  res.writeHead(200, { "Content-Type": "text/html" });
  res.end('<!doctype html><style>html,body{margin:0;background:black}canvas{display:block}</style><canvas width="64" height="64"></canvas>');
});
await new Promise((done) => server.listen(0, "127.0.0.1", done));
const base = ["--enable-unsafe-webgpu", "--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--disable-gpu-watchdog"];
const vulkan = [...base, "--use-vulkan=swiftshader", "--enable-features=Vulkan", "--disable-vulkan-surface"];
const variants = [
  ["vulkan-current", vulkan],
  ["vulkan-no-gpu-compositing", [...vulkan, "--disable-gpu-compositing"]],
  ["vulkan-new-screenshot-surface", [...vulkan.filter((s) => !s.startsWith("--enable-features=")), "--enable-features=Vulkan,CDPScreenshotNewSurface"]],
  ["graphite-swiftshader", [...base, "--enable-skia-graphite", "--skia-graphite-dawn-backend=swiftshader"]],
  ["default-unsafe-webgpu", ["--enable-unsafe-webgpu", "--disable-gpu-watchdog"]],
  ["native-vulkan", ["--enable-unsafe-webgpu", "--use-gl=angle", "--use-angle=vulkan", "--use-vulkan=native", "--enable-features=Vulkan", "--disable-vulkan-surface", "--disable-gpu-watchdog"]],
  ["native-vulkan", ["--use-gl=angle", "--use-angle=vulkan", "--use-vulkan=native", "--enable-features=Vulkan", "--disable-vulkan-surface", "--enable-unsafe-webgpu", "--disable-gpu-watchdog"]],
];
const pixels = (buffer) => {
  const { data } = PNG.sync.read(buffer);
  let red = 0, green = 0, max = 0;
  for (let i = 0; i < data.length; i += 4) {
    if (data[i] > 200 && data[i + 1] < 30) red++;
    if (data[i + 1] > 200 && data[i] < 30) green++;
    max = Math.max(max, data[i], data[i + 1], data[i + 2]);
  }
  return { red, green, max };
};
for (const [name, args] of variants) {
  let browser, timer;
  const result = { name, args, errors: [] };
  console.log(`ENVIRONMENT PROBE START ${name}`);
  try {
    await Promise.race([new Promise((_, reject) => { timer = setTimeout(() => reject(new Error("20s environment probe timeout")), 20_000); }), (async () => {
      browser = await chromium.launch({ channel: "chromium", headless: true, args, timeout: 15_000 });
      const page = await browser.newPage({ viewport: { width: 64, height: 64 } });
      page.on("pageerror", (error) => result.errors.push(error.message));
      page.on("console", (message) => { if (message.type() === "error") result.errors.push(message.text()); });
      await page.goto(`http://127.0.0.1:${server.address().port}`);
      result.adapter = await page.evaluate(async () => {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) throw new Error("No WebGPU adapter");
        const device = await adapter.requestDevice();
        window.gpuErrors = [];
        device.addEventListener("uncapturederror", (event) => window.gpuErrors.push(event.error.message));
        device.lost.then((info) => window.gpuErrors.push(`Device lost: ${info.message}`));
        const context = document.querySelector("canvas").getContext("webgpu"), format = navigator.gpu.getPreferredCanvasFormat();
        context.configure({ device, format, alphaMode: "opaque" });
        const module = device.createShaderModule({ code: `@vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4f { let p=array(vec2f(-1,-1),vec2f(3,-1),vec2f(-1,3)); return vec4f(p[i],0,1); } @fragment fn fs()->@location(0) vec4f { return vec4f(0,1,0,1); }` });
        const pipeline = device.createRenderPipeline({ layout: "auto", vertex: { module, entryPoint: "vs" }, fragment: { module, entryPoint: "fs", targets: [{ format }] } });
        function frame() {
          const encoder = device.createCommandEncoder();
          const pass = encoder.beginRenderPass({ colorAttachments: [{ view: context.getCurrentTexture().createView(), clearValue: [1, 0, 0, 1], loadOp: "clear", storeOp: "store" }] });
          if (window.drawGreen) { pass.setPipeline(pipeline); pass.draw(3); }
          pass.end(); device.queue.submit([encoder.finish()]); requestAnimationFrame(frame);
        }
        requestAnimationFrame(frame);
        await new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(done)));
        await device.queue.onSubmittedWorkDone();
        return { vendor: adapter.info.vendor, architecture: adapter.info.architecture, description: adapter.info.description };
      });
      result.clear = pixels(await page.locator("canvas").screenshot({ timeout: 5000 }));
      await page.evaluate(() => { window.drawGreen = true; return new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(done))); });
      result.draw = pixels(await page.locator("canvas").screenshot({ timeout: 5000 }));
      result.gpuErrors = await page.evaluate(() => window.gpuErrors);
    })()]);
  } catch (error) { result.failure = error.message; }
  finally { clearTimeout(timer); if (browser) await browser.close().catch(() => {}); }
  console.log(`ENVIRONMENT PROBE RESULT ${JSON.stringify(result)}`);
}
await new Promise((done) => server.close(done));
