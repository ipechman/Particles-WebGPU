// ui.js
// Wires the minimalist control panel to the engine + blender.

import { PRESET_LABELS } from "./presets.js";
import { MORPH_FUNCTIONS } from "./blender.js";
import { THEMES, findTheme } from "./themes.js";

const hex2rgb = (h) => {
  const n = parseInt(h.slice(1), 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
};
const rgb2hex = (c) => {
  const to = (x) => Math.max(0, Math.min(255, Math.round(x * 255))).toString(16).padStart(2, "0");
  return "#" + to(c[0]) + to(c[1]) + to(c[2]);
};

export function buildUI(engine, blender, camera) {
  const $ = (id) => document.getElementById(id);

  // Populate preset dropdown.
  const preset = $("preset");
  for (const [value, label] of PRESET_LABELS) {
    const o = document.createElement("option");
    o.value = value;
    o.textContent = label;
    preset.appendChild(o);
  }
  preset.value = blender.preset;

  // Morph-function dropdown (first entry is the default).
  const morphFn = $("morphFn");
  for (const [value, label] of MORPH_FUNCTIONS) {
    const o = document.createElement("option");
    o.value = value;
    o.textContent = label;
    morphFn.appendChild(o);
  }
  morphFn.value = blender.morphMode;

  // Particle-count options (filtered to what a single GPU buffer can hold).
  const particles = $("particles");
  const COUNTS = [
    262144, 524288, 1048576, 2097152, 4194304, 8388608,
    16777216, 33554432, 67108864, 100000000,
  ];
  const fmt = (n) => (n >= 1e6 ? `${+(n / 1e6).toFixed(n % 1e6 ? 1 : 0)}M` : `${(n / 1024).toFixed(0)}K`);
  for (const n of COUNTS) {
    if (n > engine.maxParticles) continue;
    const o = document.createElement("option");
    o.value = String(n);
    o.textContent = fmt(n);
    particles.appendChild(o);
  }
  particles.value = String(engine.particlesPerBatch);

  // Initial control values.
  const pcount = $("pcount");
  const pcountVal = $("pcountVal");
  pcount.value = String(blender.proceduralCount);
  pcountVal.textContent = blender.proceduralCount;

  $("animate").checked = blender.animate;
  const speedVal = $("speedVal");
  $("speed").value = String(blender.speed);
  speedVal.textContent = blender.speed.toFixed(1);
  $("pColor").value = rgb2hex(engine.particleColor);
  $("oColor").value = rgb2hex(engine.occlusionColor);
  $("bgColor").value = rgb2hex(engine.backgroundColor);

  // Color theme dropdown (+ a "Custom" entry shown when the user edits a swatch).
  const theme = $("theme");
  for (const t of THEMES) {
    const o = document.createElement("option");
    o.value = t.id;
    o.textContent = t.label;
    theme.appendChild(o);
  }
  const customOpt = document.createElement("option");
  customOpt.value = "custom";
  customOpt.textContent = "Custom";
  theme.appendChild(customOpt);
  theme.value = "ivory"; // matches the engine defaults

  const applyTheme = (id) => {
    const t = findTheme(id);
    if (!t) return;
    engine.particleColor = t.particle.slice();
    engine.occlusionColor = t.shadow.slice();
    engine.backgroundColor = t.bg.slice();
    $("pColor").value = rgb2hex(t.particle);
    $("oColor").value = rgb2hex(t.shadow);
    $("bgColor").value = rgb2hex(t.bg);
  };
  $("occMul").value = String(engine.occlusionMultiplier);
  $("occAtt").value = String(engine.occlusionAttenuation);
  const padVal = $("padVal");
  $("pad").value = String(engine.scalePadding);
  padVal.textContent = engine.scalePadding.toFixed(2);

  $("kuwahara").checked = engine.kuwaharaEnabled;
  const bloom = $("bloom");
  const bloomVal = $("bloomVal");
  bloom.value = String(engine.bloomIntensity);
  bloomVal.textContent = engine.bloomIntensity.toFixed(2);

  const updateCountRow = () => {
    $("countRow").style.display = blender.preset === "Procedural" ? "" : "none";
  };
  updateCountRow();

  // ---- events ----
  preset.addEventListener("change", () => {
    blender.setPreset(preset.value, blender.proceduralCount);
    updateCountRow();
  });
  pcount.addEventListener("input", () => {
    const n = parseInt(pcount.value, 10);
    pcountVal.textContent = n;
    blender.setPreset(blender.preset, n);
  });
  particles.addEventListener("change", () => {
    engine.particlesPerBatch = parseInt(particles.value, 10);
  });
  $("animate").addEventListener("change", (e) => (blender.animate = e.target.checked));
  morphFn.addEventListener("change", (e) => blender.setMorphMode(e.target.value));
  $("randomize").addEventListener("click", () => blender.randomizeTarget());
  $("speed").addEventListener("input", (e) => {
    blender.speed = parseFloat(e.target.value);
    speedVal.textContent = blender.speed.toFixed(1);
  });
  theme.addEventListener("change", (e) => {
    if (e.target.value !== "custom") applyTheme(e.target.value);
  });
  $("pColor").addEventListener("input", (e) => {
    engine.particleColor = hex2rgb(e.target.value);
    theme.value = "custom";
  });
  $("oColor").addEventListener("input", (e) => {
    engine.occlusionColor = hex2rgb(e.target.value);
    theme.value = "custom";
  });
  $("bgColor").addEventListener("input", (e) => {
    engine.backgroundColor = hex2rgb(e.target.value);
    theme.value = "custom";
  });
  $("occMul").addEventListener("input", (e) => (engine.occlusionMultiplier = parseFloat(e.target.value)));
  $("occAtt").addEventListener("input", (e) => (engine.occlusionAttenuation = parseFloat(e.target.value)));
  $("pad").addEventListener("input", (e) => {
    engine.scalePadding = parseFloat(e.target.value);
    padVal.textContent = engine.scalePadding.toFixed(2);
  });
  $("kuwahara").addEventListener("change", (e) => (engine.kuwaharaEnabled = e.target.checked));
  bloom.addEventListener("input", (e) => {
    engine.bloomIntensity = parseFloat(e.target.value);
    bloomVal.textContent = engine.bloomIntensity.toFixed(2);
  });

  // Collapse / hide.
  const ui = $("ui");
  $("toggle").addEventListener("click", () => ui.classList.toggle("collapsed"));

  // Keyboard shortcuts (mirroring the original: space = animate, f = new).
  window.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    if (e.key === " ") {
      blender.animate = !blender.animate;
      $("animate").checked = blender.animate;
      e.preventDefault();
    } else if (e.key === "f" || e.key === "F") {
      blender.randomizeTarget();
    } else if (e.key === "h" || e.key === "H") {
      ui.classList.toggle("hidden");
      $("hint").classList.toggle("hidden");
    }
  });

  const fpsEl = $("fps");
  return {
    setFps(fps) {
      fpsEl.textContent = `${fps.toFixed(0)} fps · ${(engine.particlesPerBatch * blender.getTransformCount()).toLocaleString()} pts`;
    },
  };
}
