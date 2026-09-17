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
- **Sampling — Focus** (default) reallocates the existing particle and drawing
  budget toward the current view. It works at the default camera and ordinary
  centered zooms. The status below the control explicitly says **Focus active**,
  **Preparing focus**, or explains why the shape is unsupported.
- Pausing morphing freezes the current shape immediately. Resuming continues the
  transition; **Randomize** while paused selects a new static shape immediately.
- **Display — Detail** (default) adds small-scale depth shading and compresses
  bright highlights smoothly. **Exposure** controls its brightness. Bloom now
  starts at 0.25 instead of 2.2. **Classic** uses the original final composite;
  select Classic and set Bloom to 2.2 to compare the previous appearance.

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
  not a guarantee of 67M distinct visible pixels. Both Global and Focus draw
  N × transformCount vertices per batch. Both retain the same base cloud, so Fast
  refinement works in either sampling mode. Switching samplers with a one-batch
  budget preserves the exact base samples as well as the fit and lighting.
- **Lighting — Full** (default) retains all particles for voxel occupancy.
  **Balanced** caps lighting samples at about 2M and **Fast** at 524K, without
  lowering the number of rendered particles. Reduced lighting budgets may change
  shadows in sparse regions. Lighting stays fixed during one accumulation cycle.
- Positions retain float32 precision in packed 12-byte records (25% less storage
  than the previous 16-byte layout). Compute dispatches avoid mostly empty rows,
  batches use disjoint counter ranges, voxel occupancy uses atomic writes, and AO
  reuses a shared-memory neighborhood tile. Detail/Bloom/Kuwahara results are reused
  whenever the displayed image and relevant effect settings are unchanged.

### More detail per particle

The view sampler builds a prefix-free hierarchy of affine IFS branches and culls
branches outside the camera. It redistributes the same N × transformCount drawing
budget among the remaining branches. The previous version drew only N focused
points and silently disabled Focus in almost all normal views to compensate for
that loss; this restriction has been removed.

Conservative boxes alone do not capture overlapping procedural geometry. A bounded
pilot of 32 attractor samples per leaf estimates screen density and nearby depth.
This guides allocation while retaining a 15% natural-distribution component and
at least one point per surviving branch. Pilot visibility never removes a branch.
Each leaf samples all child maps, avoiding holes from fixing one child for a region.

The hierarchy is bounded to 8,192 visited nodes, 1,024 leaves, and 24 levels. After
80 ms without camera motion, its allocation updates. Focus keeps the original
12-byte base-particle buffer; a draw table uses at most 82.5 KiB. At most 1,055 draw
ranges keep every buffer index below N, including at 100M particles. Focus adds
one affine transform per drawn point and more draw calls; this is a quality tradeoff,
not a claim of faster rendering.

The original global batch still establishes fit and voxel lighting. A bounded
64-byte asynchronous readback exposes that fit to the hierarchy. Camera movement
and sampler switching do not regenerate the base cloud or refit the shape. Focus
is unavailable for maps without certified contractive bounds, and the UI reports
that fallback explicitly. Pausing now stops interpolation immediately so focusing
and accumulation can start on the selected shape.

Detail display shades existing pixels using nearby valid depth samples, at the
displayed scale rather than only the global 128³ lighting scale. Background pixels
and missing depth samples do not create shading halos. The extra full-resolution
RGBA16F target uses 8 bytes per pixel; the depth pass is cached at rest.

Focusing changes the finite sample distribution, not the IFS geometry. Views
whose pixels are already saturated with particles can show smaller differences.
Absolute float32 positions still limit extreme zoom; this change does not implement
arbitrary-precision fractals or make hidden surfaces visible. At 100M particles the
normal 67M accumulation target still gives one batch, but view focusing can now
redistribute that batch when the camera changes.

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
# Linux without a desktop display:
# xvfb-run -a npm run test:browser
node tests/helpers/sampling.mjs --benchmark
```

The browser suite runs the real app in headed Chromium with software WebGPU
(Xvfb supplies the display in CI). It checks
WGSL compilation, GPU errors, nonblank rendering, presets/palettes, accumulation,
camera controls, resizing, lighting, effects, cache invalidation, and profiling.
View tests invoke the production vertex-position function on the GPU and compare
it with independent CPU coordinates, check immediate pause in all morph modes,
and require Focus to activate in 30 ordinary centered CPU views. Browser quality
tests use the visible Sampling control, normal mouse-wheel zoom, a 1280×720 viewport,
and the standard 262K particle option on two classic and two frozen procedural
forms. Both modes use identical base samples, fit, particle count, and N×M drawing
budget. An eight-batch Global image supplies a denser geometry reference.

Geometry comparisons use white points and disable bloom to exclude glow from the
coverage metric; another pair uses a normal palette and Detail display.
`normal-focus-quality.json` records the measured results and camera settings;
`focus-coordinate-oracle.json` records the GPU coordinate errors. These fixtures
measure image quality, not 100M-particle hardware FPS.
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

The graphical pipeline builds on the Unity project:

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

Post-processing: the scene renders to an HDR offscreen target, then optional
depth detail shading, an
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
    view-sampling.js    conservative bounds, visible branch frontier, allocation
    ui.js               control panel wiring
  shaders/
    iterate.wgsl        attractor iteration
    reduce.wgsl         parallel min/max/sum reduction
    fit.wgsl            auto-fit final transform
    voxelize.wgsl       voxel grid + ambient occlusion
    render.wgsl         instanced point rendering with trilinear AO
    detail.wgsl         depth detail shading at display resolution
    post.wgsl           Kuwahara filter + bloom prefilter/blur
    present.wgsl        final composite (scene + bloom) to the swap chain
```

## Credits

Original tech and concept by **Acerola**. Iterated function systems and the
chaos game: <https://paulbourke.net/fractals/ifs/> ·
<https://en.wikipedia.org/wiki/Chaos_game>. Lerp smoothing:
[@acegikmo](https://www.youtube.com/@acegikmo).
