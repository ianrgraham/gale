//! GPU (wgpu) renderer for gale HDF5 trajectories. Reads the same files the Python viewer does
//! (single file or a split `<stem>.NNNN.h5` set), tessellates each high-order DG element on the CPU
//! (per-element ⇒ inter-element jumps show), colors it on the GPU with the Turbo colormap, and masks
//! rigid bodies / embedded boundaries as clean circles by discarding the field inside the true circle
//! in the fragment shader (resolution-independent — no element/node staircase) + a thin outline.
//!
//! Two modes:
//!   - **offscreen** (default): render a frame (or `--all`) to PNG — runs headless via Vulkan.
//!   - **`--window`**: interactive winit window (needs a display; intended for a workstation, e.g.
//!     viewing an SSHFS-mounted trajectory from a remote sim machine). Arrow keys scrub, Space
//!     play/pause, Home/End jump. `--watch` polls a split set for new files and appends them live,
//!     so you can watch a running sim (point it at the `<stem>.0000.h5` of a split run).
//!
//! Usage: gale-view <traj.h5> [--field u] [--comp 0|mag] [--frame N|--all] [--out P] [--height H]
//!                            [--window] [--watch] [--fps F]

use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use std::sync::Arc;
use std::time::{Duration, Instant};

const MAX_CIRCLES: usize = 16;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    pos: [f32; 2],   // NDC
    val: f32,        // normalized [0,1]
    world: [f32; 2], // physical coords (for the circle-discard test)
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LineVertex { pos: [f32; 2] }

/// Body circles (world center + radius) the field shader discards inside. std140-friendly layout.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CirclesU {
    n: u32,
    _pad: [u32; 3],
    c: [[f32; 4]; MAX_CIRCLES], // (cx, cy, r, _)
}

fn circles_uniform(poses: &[f64], radii: &[f64]) -> CirclesU {
    let nb = (poses.len() / 3).min(MAX_CIRCLES);
    let mut c = [[0f32; 4]; MAX_CIRCLES];
    for i in 0..nb {
        let r = radii.get(i).copied().unwrap_or(*radii.first().unwrap_or(&0.0));
        c[i] = [poses[i * 3] as f32, poses[i * 3 + 1] as f32, r as f32, 0.0];
    }
    CirclesU { n: nb as u32, _pad: [0; 3], c }
}

struct Args {
    path: String,
    field: String,
    comp: String,
    frame: Option<usize>,
    all: bool,
    out: Option<String>,
    height: u32,
    window: bool,
    watch: bool,
    fps: u32,
    vmin: Option<f32>,
    vmax: Option<f32>,
}

fn parse_args() -> Args {
    let mut a = Args { path: String::new(), field: "u".into(), comp: "0".into(), frame: None, all: false, out: None, height: 600, window: false, watch: false, fps: 20, vmin: None, vmax: None };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--field" => a.field = it.next().unwrap(),
            "--comp" => a.comp = it.next().unwrap(),
            "--frame" => a.frame = Some(it.next().unwrap().parse().unwrap()),
            "--all" => a.all = true,
            "--out" => a.out = Some(it.next().unwrap()),
            "--height" => a.height = it.next().unwrap().parse().unwrap(),
            "--window" => a.window = true,
            "--watch" => a.watch = true,
            "--fps" => a.fps = it.next().unwrap().parse().unwrap(),
            "--vmin" => a.vmin = Some(it.next().unwrap().parse().unwrap()),
            "--vmax" => a.vmax = Some(it.next().unwrap().parse().unwrap()),
            _ => a.path = arg,
        }
    }
    if a.path.is_empty() {
        eprintln!("usage: gale-view <traj.h5> [--field u] [--comp 0|mag] [--frame N|--all] [--out P] [--height H] [--vmin V] [--vmax V] [--window] [--watch] [--fps F]");
        std::process::exit(2);
    }
    a
}

// ---- trajectory file set (single file or split <stem>.NNNN.h5) ------------------------------------

fn member_base(fname: &str) -> Option<String> {
    let s = fname.strip_suffix(".h5")?;
    let (base, num) = s.rsplit_once('.')?;
    (num.len() == 4 && num.bytes().all(|c| c.is_ascii_digit())).then(|| base.to_string())
}

fn list_set(path: &str) -> (Vec<String>, String) {
    let p = std::path::Path::new(path);
    let dir = p.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(std::path::Path::new("."));
    let fname = p.file_name().and_then(|s| s.to_str()).unwrap_or(path);
    let glob_base = |base: &str| -> Vec<String> {
        let mut v: Vec<String> = std::fs::read_dir(dir).ok().into_iter().flatten()
            .filter_map(|e| e.ok()?.file_name().into_string().ok())
            .filter(|n| member_base(n).as_deref() == Some(base))
            .map(|n| dir.join(n).to_string_lossy().into_owned())
            .collect();
        v.sort();
        v
    };
    if let Some(base) = member_base(fname) {
        (glob_base(&base), dir.join(&base).to_string_lossy().into_owned())
    } else if p.exists() {
        (vec![path.to_string()], path.trim_end_matches(".h5").to_string())
    } else {
        let base = fname.strip_suffix(".h5").unwrap_or(fname);
        (glob_base(base), dir.join(base).to_string_lossy().into_owned())
    }
}

// ---- CPU tessellation ----------------------------------------------------------------------------

struct Tess {
    pos_ndc: Vec<[f32; 2]>,
    world: Vec<[f32; 2]>,
    raw: Vec<f32>,
    indices: Vec<u32>,
    width: u32,
    height: u32,
    bounds: [f32; 4], // x0, y0, dx, dy
}

fn tessellate(nodes: &[f32], field: &[f32], ne: usize, nn: usize, nc: usize, comp: &str, height: u32) -> Tess {
    let n1 = (nn as f64).sqrt().round() as usize;
    let (mut x0, mut x1, mut y0, mut y1) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
    for k in 0..ne * nn {
        let (x, y) = (nodes[k * 2], nodes[k * 2 + 1]);
        x0 = x0.min(x); x1 = x1.max(x); y0 = y0.min(y); y1 = y1.max(y);
    }
    let (dx, dy) = ((x1 - x0).max(1e-9), (y1 - y0).max(1e-9));
    let width = ((height as f32) * dx / dy).round().max(1.0) as u32;
    let value = |g: usize| -> f32 {
        if comp == "mag" {
            let mut s = 0.0f32;
            for c in 0..nc { let v = field[g * nc + c]; s += v * v; }
            s.sqrt()
        } else {
            field[g * nc + comp.parse::<usize>().unwrap_or(0)]
        }
    };
    let mut pos_ndc = Vec::with_capacity(ne * nn);
    let mut world = Vec::with_capacity(ne * nn);
    let mut raw = Vec::with_capacity(ne * nn);
    for k in 0..ne * nn {
        let (x, y) = (nodes[k * 2], nodes[k * 2 + 1]);
        pos_ndc.push([(x - x0) / dx * 2.0 - 1.0, (y - y0) / dy * 2.0 - 1.0]);
        world.push([x, y]);
        raw.push(value(k));
    }
    let mut indices = Vec::with_capacity(ne * (n1 - 1) * (n1 - 1) * 6);
    for e in 0..ne {
        let base = e * nn;
        for b in 0..n1 - 1 {
            for aa in 0..n1 - 1 {
                let (k00, k10, k01, k11) =
                    (base + b * n1 + aa, base + b * n1 + aa + 1, base + (b + 1) * n1 + aa, base + (b + 1) * n1 + aa + 1);
                if !(raw[k00].is_finite() && raw[k10].is_finite() && raw[k01].is_finite() && raw[k11].is_finite()) {
                    continue; // NaN-masked (rare; bodies are masked in-shader, not here)
                }
                for k in [k00, k10, k11, k00, k11, k01] { indices.push(k as u32); }
            }
        }
    }
    Tess { pos_ndc, world, raw, indices, width, height, bounds: [x0, y0, dx, dy] }
}

/// White circle-outline + orientation-tick line segments (LineList, NDC) per body.
fn body_lines(poses: &[f64], radii: &[f64], bounds: [f32; 4]) -> Vec<LineVertex> {
    let [x0, y0, dx, dy] = bounds;
    let to_ndc = |x: f64, y: f64| LineVertex { pos: [((x as f32 - x0) / dx) * 2.0 - 1.0, ((y as f32 - y0) / dy) * 2.0 - 1.0] };
    let nb = poses.len() / 3;
    const SEG: usize = 96;
    let mut v = Vec::new();
    for i in 0..nb {
        let (cx, cy, phi) = (poses[i * 3], poses[i * 3 + 1], poses[i * 3 + 2]);
        let r = radii.get(i).copied().unwrap_or(*radii.first().unwrap_or(&0.0));
        for s in 0..SEG {
            let (t0, t1) = (std::f64::consts::TAU * s as f64 / SEG as f64, std::f64::consts::TAU * (s + 1) as f64 / SEG as f64);
            v.push(to_ndc(cx + r * t0.cos(), cy + r * t0.sin()));
            v.push(to_ndc(cx + r * t1.cos(), cy + r * t1.sin()));
        }
        v.push(to_ndc(cx, cy));
        v.push(to_ndc(cx + r * phi.cos(), cy + r * phi.sin()));
    }
    v
}

// ---- HDF5 read ---------------------------------------------------------------------------------

fn read_radii(h: &hdf5::File) -> Vec<f64> {
    h.dataset("bodies/radius").and_then(|d| d.read_raw()).unwrap_or_default()
}

fn read_frame(h: &hdf5::File, key: &str, field: &str) -> (Vec<f32>, Vec<f32>, usize, usize, usize, Vec<f64>) {
    let g = h.group(&format!("frames/{key}")).unwrap();
    let tid: u64 = g.attr("topology").unwrap().read_scalar().unwrap();
    let tg = h.group(&format!("topology/{tid}")).unwrap();
    let ne: u64 = tg.attr("ne").unwrap().read_scalar().unwrap();
    let nn: u64 = tg.attr("nn").unwrap().read_scalar().unwrap();
    let nodes: Vec<f32> = tg.dataset("elem_nodes").unwrap().read_raw().unwrap();
    let ds = g.dataset(field).unwrap();
    let nc = ds.shape()[2];
    let fld: Vec<f32> = ds.read_raw().unwrap();
    let poses: Vec<f64> = g.dataset("body_pose").and_then(|d| d.read_raw()).unwrap_or_default();
    (nodes, fld, ne as usize, nn as usize, nc, poses)
}

fn open_frames(paths: &[String]) -> (Vec<hdf5::File>, Vec<(usize, String)>) {
    let mut handles = Vec::new();
    let mut frames = Vec::new();
    for p in paths {
        let Ok(h) = hdf5::File::open(p) else { continue };
        let Ok(g) = h.group("frames") else { continue };
        let Ok(mut keys) = g.member_names() else { continue };
        keys.sort();
        let hi = handles.len();
        for k in keys { frames.push((hi, k)); }
        handles.push(h);
    }
    (handles, frames)
}

// ---- GPU ----------------------------------------------------------------------------------------

const SHADER: &str = r#"
struct Circles { n: u32, p0: u32, p1: u32, p2: u32, c: array<vec4<f32>, 16> };
@group(0) @binding(0) var<uniform> circles: Circles;

struct VsOut { @builtin(position) clip: vec4<f32>, @location(0) val: f32, @location(1) world: vec2<f32> };
@vertex
fn vs_main(@location(0) pos: vec2<f32>, @location(1) val: f32, @location(2) world: vec2<f32>) -> VsOut {
    var o: VsOut; o.clip = vec4<f32>(pos, 0.0, 1.0); o.val = val; o.world = world; return o;
}
fn turbo(t: f32) -> vec3<f32> {
    let x = clamp(t, 0.0, 1.0);
    let v4 = vec4<f32>(1.0, x, x * x, x * x * x);
    let v2 = vec2<f32>(v4.z, v4.w) * v4.z;
    let r = dot(v4, vec4<f32>(0.13572138, 4.61539260, -42.66032258, 132.13108234)) + dot(v2, vec2<f32>(-152.94239396, 59.28637943));
    let g = dot(v4, vec4<f32>(0.09140261, 2.19418839, 4.84296658, -14.18503333)) + dot(v2, vec2<f32>(4.27729857, 2.82956604));
    let b = dot(v4, vec4<f32>(0.10667330, 12.64194608, -60.58204836, 110.36276771)) + dot(v2, vec2<f32>(-89.90310912, 27.34824973));
    return clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0));
}
@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    if (in.val != in.val) { discard; }
    // discard inside any body circle (resolution-independent embedded-boundary mask)
    for (var i: u32 = 0u; i < circles.n; i = i + 1u) {
        let d = in.world - circles.c[i].xy;
        if (dot(d, d) < circles.c[i].z * circles.c[i].z) { discard; }
    }
    return vec4<f32>(turbo(in.val), 1.0);
}
@vertex
fn vs_line(@location(0) pos: vec2<f32>) -> @builtin(position) vec4<f32> { return vec4<f32>(pos, 0.0, 1.0); }
@fragment
fn fs_outline() -> @location(0) vec4<f32> { return vec4<f32>(0.1, 0.1, 0.1, 1.0); }
"#;

/// Returns (field pipeline, body-outline pipeline, field bind-group layout for the Circles uniform).
fn make_pipelines(device: &wgpu::Device, format: wgpu::TextureFormat) -> (wgpu::RenderPipeline, wgpu::RenderPipeline, wgpu::BindGroupLayout) {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADER)) });
    let target = wgpu::ColorTargetState { format, blend: Some(wgpu::BlendState::REPLACE), write_mask: wgpu::ColorWrites::ALL };
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("circles"),
        entries: &[wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None }],
    });
    let field_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&bgl], push_constant_ranges: &[] });
    let field = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("field"), layout: Some(&field_layout),
        vertex: wgpu::VertexState { module: &shader, entry_point: "vs_main", buffers: &[wgpu::VertexBufferLayout {
            array_stride: 20, step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 0, shader_location: 0 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32, offset: 8, shader_location: 1 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 12, shader_location: 2 },
            ],
        }], compilation_options: Default::default() },
        fragment: Some(wgpu::FragmentState { module: &shader, entry_point: "fs_main", targets: &[Some(target.clone())], compilation_options: Default::default() }),
        primitive: wgpu::PrimitiveState::default(), depth_stencil: None, multisample: wgpu::MultisampleState::default(), multiview: None, cache: None,
    });
    let line_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[], push_constant_ranges: &[] });
    let outline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("body-outline"), layout: Some(&line_layout),
        vertex: wgpu::VertexState { module: &shader, entry_point: "vs_line", buffers: &[wgpu::VertexBufferLayout { array_stride: 8, step_mode: wgpu::VertexStepMode::Vertex, attributes: &[wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 0, shader_location: 0 }] }], compilation_options: Default::default() },
        fragment: Some(wgpu::FragmentState { module: &shader, entry_point: "fs_outline", targets: &[Some(target)], compilation_options: Default::default() }),
        primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::LineList, ..Default::default() },
        depth_stencil: None, multisample: wgpu::MultisampleState::default(), multiview: None, cache: None,
    });
    (field, outline, bgl)
}

fn fit_viewport(win_w: f32, win_h: f32, dx: f32, dy: f32) -> (f32, f32, f32, f32) {
    let (ad, aw) = (dx / dy, win_w / win_h);
    if aw > ad { let w = win_h * ad; ((win_w - w) * 0.5, 0.0, w, win_h) } else { let h = win_w / ad; (0.0, (win_h - h) * 0.5, win_w, h) }
}

fn vertices(t: &Tess, vmin: f32, vmax: f32) -> Vec<Vertex> {
    t.pos_ndc.iter().zip(&t.world).zip(&t.raw).map(|((&pos, &world), &v)| Vertex { pos, val: (v - vmin) / (vmax - vmin), world }).collect()
}

fn circles_bind_group(device: &wgpu::Device, bgl: &wgpu::BindGroupLayout, poses: &[f64], radii: &[f64]) -> wgpu::BindGroup {
    use wgpu::util::DeviceExt;
    let cu = circles_uniform(poses, radii);
    let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("circles"), contents: bytemuck::bytes_of(&cu), usage: wgpu::BufferUsages::UNIFORM });
    device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: bgl, entries: &[wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() }] })
}

// ---- offscreen (PNG) ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn render_png(device: &wgpu::Device, queue: &wgpu::Queue, field_pl: &wgpu::RenderPipeline, outline_pl: &wgpu::RenderPipeline, bind_group: &wgpu::BindGroup, verts: &[Vertex], indices: &[u32], lines: &[LineVertex], w: u32, h: u32, out: &str) {
    use wgpu::util::DeviceExt;
    let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(verts), usage: wgpu::BufferUsages::VERTEX });
    let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(indices), usage: wgpu::BufferUsages::INDEX });
    let lbuf = (!lines.is_empty()).then(|| device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(lines), usage: wgpu::BufferUsages::VERTEX }));
    let tex = device.create_texture(&wgpu::TextureDescriptor { label: None, size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 }, mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format: wgpu::TextureFormat::Rgba8Unorm, usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC, view_formats: &[] });
    let view = tex.create_view(&Default::default());
    let unpadded = w * 4;
    let padded = unpadded.div_ceil(256) * 256;
    let readback = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (padded * h) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor { label: None, color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &view, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::WHITE), store: wgpu::StoreOp::Store } })], depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None });
        rp.set_pipeline(field_pl);
        rp.set_bind_group(0, bind_group, &[]);
        rp.set_vertex_buffer(0, vbuf.slice(..));
        rp.set_index_buffer(ibuf.slice(..), wgpu::IndexFormat::Uint32);
        rp.draw_indexed(0..indices.len() as u32, 0, 0..1);
        if let Some(lbuf) = &lbuf { rp.set_pipeline(outline_pl); rp.set_vertex_buffer(0, lbuf.slice(..)); rp.draw(0..lines.len() as u32, 0..1); }
    }
    enc.copy_texture_to_buffer(wgpu::ImageCopyTexture { texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All }, wgpu::ImageCopyBuffer { buffer: &readback, layout: wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(padded), rows_per_image: Some(h) } }, wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 });
    queue.submit([enc.finish()]);
    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);
    let data = slice.get_mapped_range();
    let mut img = vec![0u8; (unpadded * h) as usize];
    for row in 0..h as usize { let (s, d) = (row * padded as usize, row * unpadded as usize); img[d..d + unpadded as usize].copy_from_slice(&data[s..s + unpadded as usize]); }
    drop(data); readback.unmap();
    image::save_buffer(out, &img, w, h, image::ColorType::Rgba8).expect("save png");
}

fn color_range(handles: &[hdf5::File], frames: &[(usize, String)], field: &str, comp: &str) -> (f32, f32) {
    let (mut vmin, mut vmax) = (f32::MAX, f32::MIN);
    for (hi, key) in frames {
        let (n, fld, ne, nn, nc, _p) = read_frame(&handles[*hi], key, field);
        let t = tessellate(&n, &fld, ne, nn, nc, comp, 2);
        for &v in &t.raw { if v.is_finite() { vmin = vmin.min(v); vmax = vmax.max(v); } }
    }
    if vmin < vmax { (vmin, vmax) } else { (0.0, 1.0) }
}

fn run_offscreen(args: &Args) {
    let (paths, stem) = list_set(&args.path);
    if paths.is_empty() { eprintln!("gale-view: no trajectory file found at {:?}", args.path); std::process::exit(1); }
    let (handles, frames) = open_frames(&paths);
    if frames.is_empty() { eprintln!("gale-view: {:?} has no frames yet (still being written?)", args.path); std::process::exit(1); }
    let radii = read_radii(&handles[0]);
    let nfr = frames.len();
    let prefix = args.out.clone().unwrap_or(stem);
    println!("{nfr} frames across {} file(s), field='{}' comp={}", handles.len(), args.field, args.comp);

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor { backends: wgpu::Backends::VULKAN | wgpu::Backends::GL, ..Default::default() });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: None, force_fallback_adapter: false })).expect("no wgpu adapter");
    println!("renderer: {} [{:?}]", adapter.get_info().name, adapter.get_info().backend);
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor { label: None, required_features: wgpu::Features::empty(), required_limits: adapter.limits(), memory_hints: Default::default() }, None)).unwrap();
    let (field_pl, outline_pl, bgl) = make_pipelines(&device, wgpu::TextureFormat::Rgba8Unorm);

    let to_render: Vec<usize> = if args.all { (0..nfr).collect() } else { vec![args.frame.unwrap_or(nfr - 1)] };
    let render_frames: Vec<(usize, String)> = to_render.iter().map(|&i| frames[i].clone()).collect();
    let (mut vmin, mut vmax) = color_range(&handles, &render_frames, &args.field, &args.comp);
    if let Some(v) = args.vmin { vmin = v; }
    if let Some(v) = args.vmax { vmax = v; }
    println!("value range [{vmin:.4}, {vmax:.4}]");
    for &i in &to_render {
        let (hi, key) = &frames[i];
        let (nodes, fld, ne, nn, nc, poses) = read_frame(&handles[*hi], key, &args.field);
        let t = tessellate(&nodes, &fld, ne, nn, nc, &args.comp, args.height);
        let has_bodies = !poses.is_empty() && !radii.is_empty();
        let lines = if has_bodies { body_lines(&poses, &radii, t.bounds) } else { Vec::new() };
        let bg = circles_bind_group(&device, &bgl, &poses, &radii);
        let verts = vertices(&t, vmin, vmax);
        let out = if args.all { format!("{prefix}_{:06}.png", i) } else { format!("{prefix}_wgpu.png") };
        render_png(&device, &queue, &field_pl, &outline_pl, &bg, &verts, &t.indices, &lines, t.width, t.height, &out);
        println!("wrote {out}  ({}×{})", t.width, t.height);
    }
}

// ---- interactive window ------------------------------------------------------------------------

struct FrameBuffers {
    vbuf: wgpu::Buffer,
    ibuf: wgpu::Buffer,
    nidx: u32,
    bind_group: wgpu::BindGroup,
    lbuf: Option<wgpu::Buffer>,
    nline: u32,
    bounds: [f32; 4],
}

#[allow(clippy::too_many_arguments)]
fn build_frame(device: &wgpu::Device, bgl: &wgpu::BindGroupLayout, handles: &[hdf5::File], frames: &[(usize, String)], i: usize, radii: &[f64], field: &str, comp: &str, vmin: &mut f32, vmax: &mut f32, ovmin: Option<f32>, ovmax: Option<f32>) -> FrameBuffers {
    use wgpu::util::DeviceExt;
    let (hi, key) = &frames[i];
    let (nodes, fld, ne, nn, nc, poses) = read_frame(&handles[*hi], key, field);
    let t = tessellate(&nodes, &fld, ne, nn, nc, comp, 2);
    for &v in &t.raw { if v.is_finite() { *vmin = vmin.min(v); *vmax = vmax.max(v); } }
    if !(*vmin < *vmax) { *vmin = 0.0; *vmax = 1.0; }
    // Manual --vmin/--vmax override the auto-range (fixed colour window across frames).
    if let Some(v) = ovmin { *vmin = v; }
    if let Some(v) = ovmax { *vmax = v; }
    let verts = vertices(&t, *vmin, *vmax);
    let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&verts), usage: wgpu::BufferUsages::VERTEX });
    let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&t.indices), usage: wgpu::BufferUsages::INDEX });
    let has_bodies = !poses.is_empty() && !radii.is_empty();
    let lines = if has_bodies { body_lines(&poses, radii, t.bounds) } else { Vec::new() };
    let lbuf = (!lines.is_empty()).then(|| device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&lines), usage: wgpu::BufferUsages::VERTEX }));
    let bind_group = circles_bind_group(device, bgl, &poses, radii);
    FrameBuffers { vbuf, ibuf, nidx: t.indices.len() as u32, bind_group, lbuf, nline: lines.len() as u32, bounds: t.bounds }
}

fn consume_files(paths: &[String], handles: &mut Vec<hdf5::File>, frames: &mut Vec<(usize, String)>, consumed: &mut usize, radii: &mut Vec<f64>) -> bool {
    let mut changed = false;
    while *consumed < paths.len() {
        let Ok(h) = hdf5::File::open(&paths[*consumed]) else { break };
        let Ok(g) = h.group("frames") else { break };
        let Ok(mut keys) = g.member_names() else { break };
        keys.sort();
        let hi = handles.len();
        for k in keys { frames.push((hi, k)); }
        if radii.is_empty() { *radii = read_radii(&h); }
        handles.push(h);
        *consumed += 1;
        changed = true;
    }
    changed
}

fn run_window(args: Args) {
    use winit::event::{ElementState, Event, WindowEvent};
    use winit::event_loop::{ControlFlow, EventLoop};
    use winit::keyboard::{KeyCode, PhysicalKey};
    use winit::window::WindowBuilder;

    let field = args.field;
    let comp = args.comp;
    let watch = args.watch;
    let path = args.path;
    let initial = args.frame;
    let (ovmin, ovmax) = (args.vmin, args.vmax);
    let frame_dt = Duration::from_secs_f64(1.0 / args.fps.max(1) as f64);

    let mut paths = list_set(&path).0;
    if paths.is_empty() && !watch {
        eprintln!("gale-view: no trajectory file found at {path:?}\n  (pass --watch to open a window and wait for it to appear)");
        std::process::exit(1);
    }
    let mut handles: Vec<hdf5::File> = Vec::new();
    let mut frames: Vec<(usize, String)> = Vec::new();
    let mut consumed = 0usize;
    let mut radii: Vec<f64> = Vec::new();
    consume_files(&paths, &mut handles, &mut frames, &mut consumed, &mut radii);
    if frames.is_empty() { println!("gale-view: waiting for frames at {path} …"); } else { println!("{} frame(s) loaded", frames.len()); }
    println!("controls: ←/→ scrub · Space play/pause · Home/End · Esc quit");

    let event_loop = EventLoop::new().unwrap();
    let window = Arc::new(WindowBuilder::new().with_title("gale-view").build(&event_loop).unwrap());
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor { backends: wgpu::Backends::VULKAN | wgpu::Backends::GL, ..Default::default() });
    let surface = instance.create_surface(window.clone()).unwrap();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: Some(&surface), force_fallback_adapter: false })).expect("no wgpu adapter");
    let limits = adapter.limits();
    let max_dim = limits.max_texture_dimension_2d;
    println!("renderer: {} [{:?}], max surface {max_dim}px", adapter.get_info().name, adapter.get_info().backend);
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor { label: None, required_features: wgpu::Features::empty(), required_limits: limits, memory_hints: Default::default() }, None)).unwrap();
    let caps = surface.get_capabilities(&adapter);
    let format = caps.formats.iter().copied().find(|f| !f.is_srgb()).unwrap_or(caps.formats[0]);
    let size = window.inner_size();
    let mut config = wgpu::SurfaceConfiguration { usage: wgpu::TextureUsages::RENDER_ATTACHMENT, format, width: size.width.clamp(1, max_dim), height: size.height.clamp(1, max_dim), present_mode: wgpu::PresentMode::Fifo, desired_maximum_frame_latency: 2, alpha_mode: caps.alpha_modes[0], view_formats: vec![] };
    surface.configure(&device, &config);
    let (field_pl, outline_pl, bgl) = make_pipelines(&device, format);

    let (mut vmin, mut vmax) = (f32::MAX, f32::MIN);
    let follow = watch;
    let mut cur: Option<usize> = (!frames.is_empty()).then(|| initial.unwrap_or(frames.len() - 1).min(frames.len() - 1));
    let mut fb: Option<FrameBuffers> = cur.map(|c| build_frame(&device, &bgl, &handles, &frames, c, &radii, &field, &comp, &mut vmin, &mut vmax, ovmin, ovmax));
    let mut playing = false;
    let mut last_advance = Instant::now();
    let mut last_scan = Instant::now();

    let _ = event_loop.run(move |event, elwt| match event {
        Event::WindowEvent { event, .. } => match event {
            WindowEvent::CloseRequested => elwt.exit(),
            WindowEvent::Resized(s) => {
                config.width = s.width.clamp(1, max_dim);
                config.height = s.height.clamp(1, max_dim);
                surface.configure(&device, &config);
                window.request_redraw();
            }
            WindowEvent::KeyboardInput { event: ke, .. } if ke.state == ElementState::Pressed => {
                let n = frames.len();
                if let PhysicalKey::Code(KeyCode::Escape | KeyCode::KeyQ) = ke.physical_key { elwt.exit(); return; }
                if n == 0 { return; }
                let mut c = cur.unwrap_or(0);
                match ke.physical_key {
                    PhysicalKey::Code(KeyCode::ArrowRight) => c = (c + 1).min(n - 1),
                    PhysicalKey::Code(KeyCode::ArrowLeft) => c = c.saturating_sub(1),
                    PhysicalKey::Code(KeyCode::Home) => c = 0,
                    PhysicalKey::Code(KeyCode::End) => c = n - 1,
                    PhysicalKey::Code(KeyCode::Space) => { playing = !playing; return; }
                    _ => return,
                }
                cur = Some(c);
                fb = Some(build_frame(&device, &bgl, &handles, &frames, c, &radii, &field, &comp, &mut vmin, &mut vmax, ovmin, ovmax));
                window.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                let frame = match surface.get_current_texture() { Ok(f) => f, Err(_) => { surface.configure(&device, &config); return; } };
                let view = frame.texture.create_view(&Default::default());
                let mut enc = device.create_command_encoder(&Default::default());
                {
                    let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor { label: None, color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &view, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::WHITE), store: wgpu::StoreOp::Store } })], depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None });
                    if let Some(fb) = &fb {
                        let (vx, vy, vw, vh) = fit_viewport(config.width as f32, config.height as f32, fb.bounds[2], fb.bounds[3]);
                        rp.set_viewport(vx, vy, vw, vh, 0.0, 1.0);
                        rp.set_pipeline(&field_pl);
                        rp.set_bind_group(0, &fb.bind_group, &[]);
                        rp.set_vertex_buffer(0, fb.vbuf.slice(..));
                        rp.set_index_buffer(fb.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                        rp.draw_indexed(0..fb.nidx, 0, 0..1);
                        if let Some(lbuf) = &fb.lbuf { rp.set_pipeline(&outline_pl); rp.set_vertex_buffer(0, lbuf.slice(..)); rp.draw(0..fb.nline, 0..1); }
                    }
                }
                queue.submit([enc.finish()]);
                frame.present();
            }
            _ => {}
        },
        Event::AboutToWait => {
            let mut dirty = false;
            if watch && last_scan.elapsed() > Duration::from_millis(500) {
                last_scan = Instant::now();
                let new_paths = list_set(&path).0;
                if new_paths.len() > paths.len() { paths = new_paths; }
                if consume_files(&paths, &mut handles, &mut frames, &mut consumed, &mut radii) {
                    if cur.is_none() { cur = Some(if follow { frames.len() - 1 } else { 0 }); dirty = true; }
                    else if follow { cur = Some(frames.len() - 1); dirty = true; }
                }
            }
            if playing && !frames.is_empty() && last_advance.elapsed() >= frame_dt {
                last_advance = Instant::now();
                let c = cur.unwrap_or(0);
                cur = Some(if c + 1 < frames.len() { c + 1 } else { 0 });
                dirty = true;
            }
            if dirty {
                if let Some(c) = cur { fb = Some(build_frame(&device, &bgl, &handles, &frames, c, &radii, &field, &comp, &mut vmin, &mut vmax, ovmin, ovmax)); }
                window.request_redraw();
            } else if cur.is_none() {
                window.request_redraw();
            }
            elwt.set_control_flow(ControlFlow::WaitUntil(Instant::now() + Duration::from_millis(33)));
        }
        _ => {}
    });
    std::process::exit(0);
}

fn main() {
    let args = parse_args();
    if args.window { run_window(args); } else { run_offscreen(&args); }
}
