// main.js — entry point and render loop.

import { Engine } from "./engine.js";
import { Blender } from "./blender.js";
import { OrbitCamera } from "./camera.js";
import { buildUI } from "./ui.js";
import { FrameRate } from "./performance.js";

const canvas = document.getElementById("gpu");

function fail(message) {
  const el = document.getElementById("error");
  el.querySelector("p").textContent = message;
  el.classList.remove("hidden");
  console.error(message);
}

const engine = new Engine(canvas);
// Reproducible validation/profiling settings; ordinary visits keep the defaults.
const params = new URLSearchParams(location.search);
const particles = Number(params.get("particles"));
if (Number.isSafeInteger(particles) && particles >= 64 && particles <= 100000000) {
  engine.particlesPerBatch = particles;
}
const accum = Number(params.get("accum"));
if (Number.isSafeInteger(accum) && accum >= 64 && accum <= 0x40000000) engine.accumTargetPoints = accum;
const grid = Number(params.get("grid"));
if ([16, 32, 64, 128].includes(grid)) engine.voxelGridDim = grid;
engine.profilingEnabled = params.get("profile") === "1";

try {
  await engine.init();
} catch (e) {
  fail(e.message || String(e));
  throw e;
}

const blender = new Blender();
const camera = new OrbitCamera(canvas);
const ui = buildUI(engine, blender, camera);

// Expose for debugging / external control.
window.__app = { engine, blender, camera, ui };

let last = performance.now();
const frameRate = new FrameRate();

function loop(now) {
  const elapsed = (now - last) / 1000;
  const dt = Math.min(elapsed, 0.05);
  last = now;

  blender.update(dt);
  engine.frame(blender, camera);
  ui.updateStatus();

  const fps = frameRate.update(elapsed);
  if (fps !== null) ui.setFps(fps);

  requestAnimationFrame(loop);
}

requestAnimationFrame(loop);
