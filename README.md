# Point Cloud Fractals — WebGPU

A pure **WebGPU** reimplementation of Acerola's *Point Cloud Fractals* tech demo.
It draws millions of particles to approximate the attractor of an iterated
function system (IFS), with a real-time brute-force ambient-occlusion lighting
solution, and continuously morphs between fractal forms.

Everything runs in the browser — no Unity, no engine, no build step. Just static
HTML, JavaScript modules, and WGSL compute/render shaders.

![fractal](./Examples/flagship.png)
![fractal](./Examples/f17.png)

## Run it

WebGPU requires a secure context, so serve the `web/` folder over
`http://localhost` (opening `index.html` from disk won't work):

```bash
# any static server works, e.g.:
python -m http.server 8080 --directory web
# then open http://localhost:8080
```

Needs a WebGPU-capable browser (recent Chrome / Edge / Chromium).

## Controls

- **Drag** — orbit · **scroll** — zoom · **right-drag** — pan
- **Space** — toggle morphing · **F** — randomize target form · **H** — hide UI
- The panel exposes the fractal preset, transform count, particle count, colors,
  ambient-occlusion strength/falloff, the **morph function** (Lerp Smoothing /
  Linear / Smoothstep / Spring) and its **speed**, and **scale** (how large the
  fractal is sized inside the lighting box).
- **Post-processing**: a toggle for the original **Kuwahara filter** (Acerola's
  edge-preserving painterly filter) and a **bloom** slider (0 disables it).

## What it does (preserved from the original)

The graphical pipeline mirrors the Unity project one-to-one:

1. **Generating functions** — the six classic presets (Sierpinski Triangle /
   Vicsek / Sierpinski Carpet, 2D and 3D) plus a procedural random generator,
   each producing a set of affine transforms.
2. **Iterated system** — a compute shader builds the attractor point cloud by
   recursively applying every transform, one generation per dispatch.
3. **Auto-fit** — a low-detail copy of the attractor is reduced (min/max/sum on
   the GPU) to predict a bounding box, and a "final transform" rescales and
   recenters the fractal to fill the view.
4. **Voxelization + ambient occlusion** — the cloud is splatted into a 3D voxel
   grid and a brute-force AO approximation (3×3×3 neighbourhood) is computed,
   then sampled trilinearly in the particle shader for shading.
5. **Morphing / smoothing functions** — frame-rate-independent exponential
   "lerp smoothing" with an easing curve, and quaternion slerp for rotations,
   continuously eases the current form toward freshly generated targets. The
   morph function is selectable (Lerp Smoothing, Linear, Smoothstep, Spring).

Post-processing (new): the scene renders to an HDR offscreen target, then an
optional **Kuwahara filter** (the original four-quadrant minimum-variance
filter) and an optional **bloom** pass (bright-pass + separable Gaussian blur,
controlled by an intensity slider) composite to the screen.

## Layout

```
web/
  index.html            page + minimalist UI
  css/style.css
  js/
    main.js             entry point + render loop
    engine.js           WebGPU device, buffers, pipelines, per-frame pipeline
    math.js             column-major mat4 / quaternion / vec3 (Unity-faithful)
    presets.js          generating functions (presets + procedural)
    blender.js          morphing/smoothing + affine matrix construction
    animcurve.js        AnimationCurve (cubic Hermite + PingPong)
    camera.js           orbit camera
    ui.js               control panel wiring
  shaders/
    iterate.wgsl        attractor iteration
    lod.wgsl            low-detail generation for bounds prediction
    reduce.wgsl         parallel min/max/sum reduction
    fit.wgsl            auto-fit final transform
    voxelize.wgsl       voxel grid + ambient occlusion
    render.wgsl         instanced point rendering with trilinear AO
    post.wgsl           Kuwahara filter + bloom prefilter/blur
    present.wgsl        final composite (scene + bloom) to the swap chain
```

## Credits

Original tech and concept by **Acerola**. Iterated function systems and the
chaos game: <https://paulbourke.net/fractals/ifs/> ·
<https://en.wikipedia.org/wiki/Chaos_game>. Lerp smoothing:
[@acegikmo](https://www.youtube.com/@acegikmo).
