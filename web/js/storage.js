// storage.js
// Persist user-saved procedural shapes (their affine transform sets) in
// localStorage so they survive reloads.

const KEY = "pcf.savedShapes.v1";

export function getSavedShapes() {
  try {
    const arr = JSON.parse(localStorage.getItem(KEY) || "[]");
    return Array.isArray(arr) ? arr : [];
  } catch {
    return [];
  }
}

export function setSavedShapes(shapes) {
  try {
    localStorage.setItem(KEY, JSON.stringify(shapes));
  } catch (e) {
    console.warn("Could not persist shapes:", e);
  }
}

export function addSavedShape(name, transforms) {
  const shapes = getSavedShapes();
  shapes.push({ name, transforms, savedAt: Date.now() });
  setSavedShapes(shapes);
  return shapes;
}

export function deleteSavedShape(index) {
  const shapes = getSavedShapes();
  if (index >= 0 && index < shapes.length) shapes.splice(index, 1);
  setSavedShapes(shapes);
  return shapes;
}
