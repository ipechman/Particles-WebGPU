// themes.js
// Curated color themes. Each sets the highlight (particle), shadow (occlusion)
// and background colors. The first entry matches the engine defaults.

export const THEMES = [
  { id: "ivory",   label: "Ivory",        particle: [0.93, 0.94, 0.96], shadow: [0.104, 0.014, 0.014], bg: [0.0, 0.0, 0.0] },
  { id: "acerola", label: "Acerola",      particle: [0.854, 1.0, 0.895], shadow: [0.104, 0.014, 0.014], bg: [0.0, 0.0, 0.0] },
  { id: "ember",   label: "Ember",        particle: [1.0, 0.74, 0.38],  shadow: [0.16, 0.02, 0.0],     bg: [0.03, 0.008, 0.0] },
  { id: "ice",     label: "Ice",          particle: [0.78, 0.92, 1.0],  shadow: [0.0, 0.05, 0.16],     bg: [0.01, 0.02, 0.05] },
  { id: "neon",    label: "Neon",         particle: [1.0, 0.5, 0.95],   shadow: [0.10, 0.0, 0.20],     bg: [0.03, 0.0, 0.06] },
  { id: "gold",    label: "Gold",         particle: [1.0, 0.86, 0.5],   shadow: [0.14, 0.06, 0.0],     bg: [0.02, 0.015, 0.0] },
  { id: "viridis", label: "Viridis",      particle: [0.7, 1.0, 0.55],   shadow: [0.12, 0.0, 0.22],     bg: [0.0, 0.01, 0.03] },
  { id: "mono",    label: "Monochrome",   particle: [1.0, 1.0, 1.0],    shadow: [0.04, 0.04, 0.05],    bg: [0.0, 0.0, 0.0] },
];

export function findTheme(id) {
  return THEMES.find((t) => t.id === id);
}
