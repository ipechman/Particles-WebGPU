// camera.js
// A minimal orbit camera (drag to rotate, wheel to zoom, right/middle-drag to
// pan). Produces a column-major view-projection matrix for WGSL.

import { mat4, v3 } from "./math.js";

export class OrbitCamera {
  constructor(canvas) {
    this.canvas = canvas;
    this.target = [0, 0, 0];
    this.distance = 4.5;
    this.yaw = 0.6;     // radians
    this.pitch = 0.45;  // radians
    this.fov = (60 * Math.PI) / 180;
    this.near = 0.01;
    this.far = 1000;

    this.minDistance = 0.25;
    this.maxDistance = 40;

    this._attach();
  }

  eye() {
    const cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw);
    const dir = [cp * cy, sp, cp * sy];
    return v3.add(this.target, v3.scale(dir, this.distance));
  }

  viewProj(aspect) {
    const view = mat4.lookAt(this.eye(), this.target, [0, 1, 0]);
    const proj = mat4.perspective(this.fov, aspect, this.near, this.far);
    return mat4.multiply(proj, view);
  }

  _attach() {
    const el = this.canvas;
    let dragging = false;
    let panning = false;
    let lastX = 0, lastY = 0;

    const onDown = (e) => {
      el.setPointerCapture?.(e.pointerId);
      lastX = e.clientX;
      lastY = e.clientY;
      if (e.button === 0) dragging = true;
      else if (e.button === 1 || e.button === 2) panning = true;
    };
    const onUp = (e) => {
      dragging = false;
      panning = false;
      el.releasePointerCapture?.(e.pointerId);
    };
    const onMove = (e) => {
      const dx = e.clientX - lastX;
      const dy = e.clientY - lastY;
      lastX = e.clientX;
      lastY = e.clientY;
      if (dragging) {
        this.yaw -= dx * 0.005;
        this.pitch += dy * 0.005;
        const lim = Math.PI / 2 - 0.01;
        this.pitch = Math.max(-lim, Math.min(lim, this.pitch));
      } else if (panning) {
        // Pan in the camera's screen plane.
        const eye = this.eye();
        const fwd = v3.normalize(v3.sub(this.target, eye));
        const right = v3.normalize(v3.cross(fwd, [0, 1, 0]));
        const up = v3.cross(right, fwd);
        const s = this.distance * 0.0015;
        this.target = v3.add(this.target, v3.scale(right, -dx * s));
        this.target = v3.add(this.target, v3.scale(up, dy * s));
      }
    };
    const onWheel = (e) => {
      e.preventDefault();
      const factor = Math.exp(e.deltaY * 0.001);
      this.distance = Math.max(this.minDistance, Math.min(this.maxDistance, this.distance * factor));
    };

    el.addEventListener("pointerdown", onDown);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointermove", onMove);
    el.addEventListener("wheel", onWheel, { passive: false });
    el.addEventListener("contextmenu", (e) => e.preventDefault());
  }
}
