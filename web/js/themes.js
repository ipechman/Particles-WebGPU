// themes.js
// Curated color themes. Each sets the highlight (particle), shadow (occlusion)
// and background colors, with optional AO-driven color stops. The first entry
// matches the engine defaults. Stop colors use the same RGB space as the swatches.

// Keep in sync with Render.paletteStops in render.wgsl.
export const MAX_PALETTE_STOPS = 6;

const rgb = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
const rampTheme = (id, label, stops) => {
  const paletteStops = stops.map(([position, hex]) => ({ position, color: rgb(hex) }));
  return {
    id, label, paletteStops,
    shadow: paletteStops[0].color,
    particle: paletteStops[paletteStops.length - 1].color,
    bg: [0, 0, 0],
  };
};

export const THEMES = [
  { id: "ivory",   label: "Ivory",        particle: [0.93, 0.94, 0.96], shadow: [0.104, 0.014, 0.014], bg: [0.0, 0.0, 0.0] },
  { id: "acerola", label: "Acerola",      particle: [0.854, 1.0, 0.895], shadow: [0.104, 0.014, 0.014], bg: [0.0, 0.0, 0.0] },
  { id: "ember",   label: "Ember",        particle: [1.0, 0.74, 0.38],  shadow: [0.16, 0.02, 0.0],     bg: [0.03, 0.008, 0.0] },
  { id: "ice",     label: "Ice",          particle: [0.78, 0.92, 1.0],  shadow: [0.0, 0.05, 0.16],     bg: [0.01, 0.02, 0.05] },
  { id: "neon",    label: "Neon",         particle: [1.0, 0.5, 0.95],   shadow: [0.10, 0.0, 0.20],     bg: [0.03, 0.0, 0.06] },
  { id: "gold",    label: "Gold",         particle: [1.0, 0.86, 0.5],   shadow: [0.14, 0.06, 0.0],     bg: [0.02, 0.015, 0.0] },
  { id: "viridis", label: "Viridis",      particle: [0.7, 1.0, 0.55],   shadow: [0.12, 0.0, 0.22],     bg: [0.0, 0.01, 0.03] },
  { id: "mono",    label: "Monochrome",   particle: [1.0, 1.0, 1.0],    shadow: [0.04, 0.04, 0.05],    bg: [0.0, 0.0, 0.0] },
  // Reference 1: turquoise shadows, amber edges and luminous yellow-cream tips.
  rampTheme("gilded-lagoon", "Gilded Lagoon", [
    [0.00, "#071b26"], [0.24, "#276b79"], [0.46, "#85b9aa"],
    [0.60, "#b67929"], [0.78, "#ead270"], [1.00, "#fbfcde"],
  ]),
  // Reference 2: deeper shadows with copper, warm gold and pale fern highlights.
  rampTheme("amber-fern", "Amber Fern", [
    [0.00, "#07131a"], [0.25, "#244650"], [0.43, "#a35b20"],
    [0.62, "#dda443"], [0.82, "#f8dc79"], [1.00, "#f9fad0"],
  ]),
  // Reference 3: orange embers in the recesses, blue-teal bodies and icy tips.
  rampTheme("glacial-ember", "Glacial Ember", [
    [0.00, "#030c1b"], [0.22, "#bd4c0a"], [0.36, "#e79b30"],
    [0.50, "#1b526b"], [0.76, "#66a4a8"], [1.00, "#eef8e8"],
  ]),
];

export function findTheme(id) {
  return THEMES.find((t) => t.id === id);
}

// Clone editable colors so customizing a theme never changes its preset.
export function applyTheme(engine, id) {
  const t = findTheme(id);
  if (!t) return;
  engine.particleColor = t.particle.slice();
  engine.occlusionColor = t.shadow.slice();
  engine.backgroundColor = t.bg.slice();
  engine.paletteStops = (t.paletteStops || []).map((stop) => ({
    position: stop.position, color: stop.color.slice(),
  }));
}
