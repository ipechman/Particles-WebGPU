// main.js — entry point and render loop.

import { Engine } from "./engine.js";
import { Blender } from "./blender.js";
import { OrbitCamera } from "./camera.js";
import { buildUI } from "./ui.js";

const canvas = document.getElementById("gpu");

function fail(message) {
  const el = document.getElementById("error");
  el.querySelector("p").textContent = message;
  el.classList.remove("hidden");
  console.error(message);
}

const engine = new Engine(canvas);

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
let acc = 0;
let frames = 0;

function loop(now) {
  const dt = Math.min((now - last) / 1000, 0.05);
  last = now;

  blender.update(dt);
  engine.frame(blender, camera);

  acc += dt;
  frames++;
  if (acc >= 0.5) {
    ui.setFps(frames / acc);
    acc = 0;
    frames = 0;
  }

  requestAnimationFrame(loop);
}

requestAnimationFrame(loop);
