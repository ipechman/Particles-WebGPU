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

### Reference color palettes

The **Theme** dropdown includes three palettes inspired by the supplied fractal
references, in the same order:

| Theme | Palette |
| --- | --- |
| Gilded Lagoon | Deep navy, turquoise, seafoam, amber, pale gold, cream |
| Amber Fern | Blue-black, muted teal, copper, warm gold, yellow, pale cream |
| Glacial Ember | Midnight blue, burnt orange, amber, blue-teal, aqua, icy white |

These themes map ambient-occlusion lighting through six color stops on a black
background. The gradient below the color controls previews the shadow-to-highlight
sequence. **Color** and **Shadow** edit its endpoints; the four **Accents** swatches
edit the intermediate colors. Editing any swatch selects **Custom**; reselecting a
theme restores its original colors. The original themes retain their two-color
blend. Palette selection leaves the shape, lighting and post-processing settings
as they are, so the exact appearance depends on the current fractal and AO/bloom.

Run the palette/renderer regression checks with `node --test tests/*.test.mjs`
(Node.js 22.7 or newer; no dependencies or build step required).

### Performance and refinement

- **Refinement — Fast** (default) generates the first batch at full depth, then
  advances each stored point by four chaos-game hops while the shape and view
  remain still. **Independent** generates every batch at full depth for comparison.
  Shape changes restart sampling; moving the camera restarts image accumulation.
  Both modes refine toward the existing 67M base-sample target. These are samples,
  not a guarantee of 67M distinct visible pixels; rendering multiplies the count
  by the number of transforms.
- **Lighting — Full** (default) retains all particles for voxel occupancy.
  **Balanced** caps lighting samples at about 2M and **Fast** at 524K, without
  lowering the number of rendered particles. Reduced lighting budgets may change
  shadows in sparse regions. Lighting stays fixed during one accumulation cycle.
- Positions retain float32 precision in packed 12-byte records (25% less storage
  than the previous 16-byte layout). Compute dispatches avoid mostly empty rows,
  batches use disjoint counter ranges, voxel occupancy uses atomic writes, and AO
  reuses a shared-memory neighborhood tile. Bloom/Kuwahara results are reused
  whenever the displayed image and relevant effect settings are unchanged.

Open `?profile=1` to enable optional GPU timestamps, sampled every 30 frames.
`window.__app.engine.profiler.latest` contains timings by pass and their sum.
The displayed **GPU passes** value excludes copies and gaps between passes; it
is not total frame latency. Unsupported adapters render normally without timings.
Readbacks are asynchronous and bounded; a busy profiling ring skips a sample.
FPS uses actual wall time, independently of the simulation timestep clamp.

For repeatable tests, URL parameters can set `particles` (64–100M), `accum`
(target base samples, 64–1,073,741,824), and `grid` (16, 32, 64, or 128).
For example: `?particles=262144&accum=1048576&grid=32&profile=1`.
Normal visits retain the original 8.4M particles, 128³ grid, and 67M target.

### Validation

```sh
npm ci
npm test
npx playwright install --with-deps chromium
npm run test:browser
node tests/helpers/sampling.mjs --benchmark
```

The browser suite runs the real app in Chromium with software WebGPU. It checks
WGSL compilation, GPU errors, nonblank rendering, presets/palettes, accumulation,
camera controls, resizing, lighting, effects, cache invalidation, and profiling.
Screenshots and a JSON report are written under `test-results/browser/` and
uploaded by the regression workflow. Software-adapter timings are not hardware
performance benchmarks. No npm packages are loaded by the deployed site.

The CPU sampling benchmark compares six frozen 2D, 3D, and procedural shapes.
At 32,768 particles × eight batches and a 1024² projection, four-hop refinement
retained 99.708%–100.053% of the independent reference's occupied-pixel count,
using 31.94%–39.42% as many affine operations. These checks are not pixel-perfect
equivalence or measured FPS gains; camera views, zoom, and shapes outside the
fixtures may differ. The Independent option remains available for comparison.

## What it does (preserved from the original)

The graphical pipeline mirrors the Unity project one-to-one:

1. **Generating functions** — the six classic presets (Sierpinski Triangle /
   Vicsek / Sierpinski Carpet, 2D and 3D) plus a procedural random generator,
   each producing a set of affine transforms.
2. **Iterated system** — a compute shader builds the attractor point cloud
   with a deterministic per-particle chaos game in a single dispatch: each
   particle starts on a transform's fixed point (an exact attractor point) and
   applies a hash-driven sequence of transforms, so every rendered point lies
   on the attractor.
3. **Auto-fit** — the initial particle batch is reduced (min/max/sum on
   the GPU) to calculate a bounding box, and a "final transform" rescales and
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
