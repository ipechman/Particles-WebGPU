// ui.js
// Wires the minimalist control panel to the engine + blender.

import { PRESET_LABELS } from "./presets.js";
import { MORPH_FUNCTIONS } from "./blender.js";
import { THEMES, applyTheme } from "./themes.js";
import { getSavedShapes, addSavedShape, deleteSavedShape } from "./storage.js";

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
  // Nonstandard counts can be supplied by reproducible profiling URLs.
  if (!particles.value) {
    const option = document.createElement("option");
    option.value = String(engine.particlesPerBatch);
    option.textContent = fmt(engine.particlesPerBatch);
    particles.appendChild(option);
    particles.value = option.value;
  }
  $("refinement").value = engine.accumulationMode;
  $("refinement").addEventListener("change", (e) => { engine.accumulationMode = e.target.value; });
  $("sampling").value = engine.samplingMode;
  $("sampling").addEventListener("change", (e) => { engine.samplingMode = e.target.value; });
  $("displayMode").value = engine.displayMode;
  const updateExposure = () => { $("exposureRow").style.display = engine.displayMode === "detail" ? "" : "none"; };
  $("displayMode").addEventListener("change", (e) => { engine.displayMode = e.target.value; updateExposure(); });
  $("exposure").value = String(engine.exposure);
  $("exposureVal").textContent = engine.exposure.toFixed(2);
  $("exposure").addEventListener("input", (e) => {
    engine.exposure = Number(e.target.value);
    $("exposureVal").textContent = engine.exposure.toFixed(2);
  });
  updateExposure();
  $("lighting").addEventListener("change", (e) => {
    engine.lightingParticleBudget = e.target.value === "full" ? Infinity : Number(e.target.value);
  });

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

  const palettePreview = $("palettePreview");
  const paletteAccents = $("paletteAccents");
  const updatePalettePreview = () => {
    const stops = engine.paletteStops.length >= 2 ? engine.paletteStops : [
      { position: 0, color: engine.occlusionColor },
      { position: 1, color: engine.particleColor },
    ];
    const colors = stops.map((stop, i) => rgb2hex(i === 0 ? engine.occlusionColor
      : i === stops.length - 1 ? engine.particleColor : stop.color));
    palettePreview.style.background = `linear-gradient(to right, ${stops.map((stop, i) =>
      `${colors[i]} ${stop.position * 100}%`).join(", ")})`;
    palettePreview.setAttribute("aria-label", `Shadow to highlight: ${colors.join(", ")}`);
  };
  const refreshPaletteControls = () => {
    paletteAccents.replaceChildren();
    paletteAccents.hidden = engine.paletteStops.length <= 2;
    if (!paletteAccents.hidden) {
      const label = document.createElement("span");
      label.textContent = "Accents";
      paletteAccents.appendChild(label);
      engine.paletteStops.slice(1, -1).forEach((stop, i) => {
        const input = document.createElement("input");
        input.type = "color";
        input.value = rgb2hex(stop.color);
        input.title = `Accent ${i + 1} (${Math.round(stop.position * 100)}% light)`;
        input.setAttribute("aria-label", `Palette accent ${i + 1}`);
        input.addEventListener("input", (e) => {
          stop.color = hex2rgb(e.target.value);
          theme.value = "custom";
          updatePalettePreview();
        });
        paletteAccents.appendChild(input);
      });
    }
    updatePalettePreview();
  };
  const selectTheme = (id) => {
    applyTheme(engine, id);
    $("pColor").value = rgb2hex(engine.particleColor);
    $("oColor").value = rgb2hex(engine.occlusionColor);
    $("bgColor").value = rgb2hex(engine.backgroundColor);
    refreshPaletteControls();
  };
  refreshPaletteControls();
  $("occMul").value = String(engine.occlusionMultiplier);
  $("occAtt").value = String(engine.occlusionAttenuation);
  const padVal = $("padVal");
  $("pad").value = String(engine.scalePadding);
  padVal.textContent = engine.scalePadding.toFixed(2);

  $("kuwahara").checked = engine.kuwaharaEnabled;
  const kuwSize = $("kuwSize");
  const kuwSizeVal = $("kuwSizeVal");
  const kuwSharp = $("kuwSharp");
  const kuwSharpVal = $("kuwSharpVal");
  kuwSize.value = String(engine.kuwaharaKernelSize);
  kuwSizeVal.textContent = engine.kuwaharaKernelSize;
  kuwSharp.value = String(engine.kuwaharaSharpness);
  kuwSharpVal.textContent = engine.kuwaharaSharpness;
  const updateKuwRows = () => {
    const show = $("kuwahara").checked ? "" : "none";
    $("kuwSizeRow").style.display = show;
    $("kuwSharpRow").style.display = show;
  };
  updateKuwRows();

  const bloom = $("bloom");
  const bloomVal = $("bloomVal");
  bloom.value = String(engine.bloomIntensity);
  bloomVal.textContent = engine.bloomIntensity.toFixed(2);

  const updateCountRow = () => {
    $("countRow").style.display = blender.preset === "Procedural" ? "" : "none";
  };
  updateCountRow();

  // ---- saved shapes (localStorage) ----
  const shapeList = $("shapeList");
  let savedShapes = getSavedShapes();
  const refreshShapeList = (sel = -1) => {
    shapeList.innerHTML = "";
    const ph = document.createElement("option");
    ph.value = "-1";
    ph.textContent = savedShapes.length ? "— load saved —" : "— none saved —";
    shapeList.appendChild(ph);
    savedShapes.forEach((s, i) => {
      const o = document.createElement("option");
      o.value = String(i);
      o.textContent = s.name || `Shape ${i + 1}`;
      shapeList.appendChild(o);
    });
    shapeList.value = String(sel);
  };
  refreshShapeList();

  const syncAfterLoad = () => {
    preset.value = "Procedural";
    updateCountRow();
    pcount.value = String(blender.proceduralCount);
    pcountVal.textContent = blender.proceduralCount;
    $("animate").checked = false;
  };

  shapeList.addEventListener("change", () => {
    const idx = parseInt(shapeList.value, 10);
    if (idx < 0 || !savedShapes[idx]) return;
    blender.loadShape(savedShapes[idx].transforms);
    syncAfterLoad();
  });
  $("saveShape").addEventListener("click", () => {
    const def = `Shape ${savedShapes.length + 1}`;
    const name = window.prompt("Name this shape:", def);
    if (name === null) return; // cancelled
    savedShapes = addSavedShape(name.trim() || def, blender.getCurrentShape());
    refreshShapeList(savedShapes.length - 1);
  });
  $("deleteShape").addEventListener("click", () => {
    const idx = parseInt(shapeList.value, 10);
    if (idx < 0) return;
    savedShapes = deleteSavedShape(idx);
    refreshShapeList(-1);
  });

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
    if (e.target.value !== "custom") selectTheme(e.target.value);
  });
  $("pColor").addEventListener("input", (e) => {
    engine.particleColor = hex2rgb(e.target.value);
    theme.value = "custom";
    updatePalettePreview();
  });
  $("oColor").addEventListener("input", (e) => {
    engine.occlusionColor = hex2rgb(e.target.value);
    theme.value = "custom";
    updatePalettePreview();
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
  $("kuwahara").addEventListener("change", (e) => {
    engine.kuwaharaEnabled = e.target.checked;
    updateKuwRows();
  });
  kuwSize.addEventListener("input", (e) => {
    engine.kuwaharaKernelSize = parseInt(e.target.value, 10);
    kuwSizeVal.textContent = engine.kuwaharaKernelSize;
  });
  kuwSharp.addEventListener("input", (e) => {
    engine.kuwaharaSharpness = parseFloat(e.target.value);
    kuwSharpVal.textContent = engine.kuwaharaSharpness;
  });
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
  const samplingStatus = $("samplingStatus");
  return {
    updateStatus() {
      let message;
      if (engine.samplingMode === "global") message = "Global sampling selected";
      else if (engine._fitReadbackFailed) message = "Focus unavailable: could not read the shape fit";
      else if (engine.viewStats.reason === "uncertified bounds") message = "Focus unavailable for this shape; using Global";
      else if (engine._viewPending || engine._fitPending) message = "Preparing focus…";
      else if (engine._viewActive) message = engine.viewStats.vertices ? "Focus active" : "Focus active — shape outside the view";
      else message = blender.animate ? "Pause morphing to focus the current shape" : "Preparing focus…";
      if (samplingStatus.textContent !== message) samplingStatus.textContent = message;
    },
    setFps(fps) {
      // Accumulated batches multiply the points actually baked into the frame.
      const perBatch = engine._viewActive ? engine.viewStats.vertices : engine.particlesPerBatch * blender.getTransformCount();
      const pts = perBatch * Math.max(1, engine._accumCount);
      fpsEl.textContent = `${fps.toFixed(0)} fps · ${pts.toLocaleString()} pts`;
      if (engine.samplingMode === "view") fpsEl.textContent += engine._viewActive ? " · view focused" : " · global";
      const timing = engine.profiler?.latest;
      if (timing) fpsEl.textContent += ` · GPU passes ${timing.totalMs.toFixed(1)} ms`;
    },
  };
}
