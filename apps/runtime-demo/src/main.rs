//! Habanero runtime vertical-slice demo.
//!
//! Spawns a large, animated "galaxy" of instanced cubes driven entirely by the
//! ECS + physics simulation, renders it through the Vulkan instanced lighting
//! pipeline with an automatic orbiting camera, and reports live performance
//! telemetry (FPS, frame-time percentiles, draw counts, culling ratio).
//!
//! Flags:
//!   --stress=<N>     number of animated entities (default 12000, max 200000)
//!   --frames=<N>     run N frames then exit and print a perf report (headless CI)
//!   --no-vulkan      force the CPU/headless path (no GPU presentation)

use std::time::Instant;

use anyhow::Result;
use engine_core::Engine;
use engine_ecs::{Entity, Transform};
use engine_physics::{Orbit, Spin};
use tracing_subscriber::EnvFilter;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

struct DemoArgs {
    stress_entities: usize,
    max_frames: Option<u64>,
    use_vulkan: bool,
}

struct FrameStats {
    samples: Vec<f32>,
    capacity: usize,
    cursor: usize,
}

impl FrameStats {
    fn new(capacity: usize) -> Self {
        Self {
            samples: Vec::with_capacity(capacity),
            capacity,
            cursor: 0,
        }
    }

    fn push(&mut self, ms: f32) {
        if self.samples.len() < self.capacity {
            self.samples.push(ms);
        } else {
            self.samples[self.cursor] = ms;
            self.cursor = (self.cursor + 1) % self.capacity;
        }
    }

    fn percentile(&self, pct: f32) -> f32 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let mut sorted = self.samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let rank = ((pct / 100.0) * (sorted.len() as f32 - 1.0)).round() as usize;
        sorted[rank.min(sorted.len() - 1)]
    }
}

struct RuntimeApp {
    window: Option<Window>,
    window_id: Option<WindowId>,
    engine: Option<Engine>,
    last_tick: Instant,
    last_title_update: Instant,
    args: DemoArgs,
    frame_stats: FrameStats,
}

impl RuntimeApp {
    fn new(args: DemoArgs) -> Self {
        Self {
            window: None,
            window_id: None,
            engine: None,
            last_tick: Instant::now(),
            last_title_update: Instant::now(),
            args,
            frame_stats: FrameStats::new(240),
        }
    }

    /// Build an animated galaxy: concentric rings of orbiting, spinning cubes
    /// plus a central rising helix. Every entity carries `Transform` + `Spin` +
    /// `Orbit` components so the whole scene is driven by the ECS each frame.
    fn seed_scene(engine: &mut Engine, target_entities: usize) {
        let mut placed = 0usize;
        let mut ring = 0usize;

        while placed < target_entities {
            let radius = 3.0 + ring as f32 * 1.6;
            let circumference = std::f32::consts::TAU * radius;
            let count = ((circumference / 1.1) as usize).max(6);
            let orbit_dir = if ring.is_multiple_of(2) { 1.0 } else { -1.0 };
            let orbit_speed = orbit_dir * (0.45 / (1.0 + ring as f32 * 0.12));
            for i in 0..count {
                if placed >= target_entities {
                    break;
                }
                let theta = (i as f32 / count as f32) * std::f32::consts::TAU;
                let wobble = (ring as f32 * 0.6 + i as f32 * 0.25).sin() * 1.2;
                let scale = 0.35 + (ring % 3) as f32 * 0.12;
                let entity = engine.scene.spawn_named(
                    format!("Cube{placed}"),
                    Transform {
                        translation: [radius * theta.cos(), wobble, radius * theta.sin()],
                        scale: [scale, scale, scale],
                        ..Default::default()
                    },
                );
                Self::attach_motion(engine, entity, placed, orbit_speed);
                placed += 1;
            }
            ring += 1;
            if ring > 4096 {
                break;
            }
        }

        // Central rising helix as a visual focal point.
        let helix = (target_entities / 20).clamp(40, 600);
        for i in 0..helix {
            if placed >= target_entities {
                break;
            }
            let t = i as f32 * 0.35;
            let r = 1.2 + (t * 0.05).sin().abs() * 0.6;
            let entity = engine.scene.spawn_named(
                format!("Helix{i}"),
                Transform {
                    translation: [r * t.cos(), -6.0 + i as f32 * 0.18, r * t.sin()],
                    scale: [0.3, 0.3, 0.3],
                    ..Default::default()
                },
            );
            Self::attach_motion(engine, entity, placed, 0.8);
            placed += 1;
        }

        // Frame the camera near the galaxy rim: close enough that the frustum
        // cull does real work (off-screen / behind-camera instances), while the
        // far side of the disc fills the view for a dramatic flythrough.
        let extent = 3.0 + ring as f32 * 1.6;
        engine.camera.radius = (extent * 0.85).clamp(10.0, 150.0);
        engine.camera.height = (extent * 0.32).clamp(4.0, 70.0);
    }

    fn attach_motion(engine: &mut Engine, entity: Entity, seed: usize, orbit_speed: f32) {
        // Cheap deterministic pseudo-random spin axis/speed from the seed.
        let h = seed.wrapping_mul(2654435761) as u32;
        let ax = ((h & 0xFF) as f32 / 255.0) - 0.5;
        let ay = (((h >> 8) & 0xFF) as f32 / 255.0) - 0.2;
        let az = (((h >> 16) & 0xFF) as f32 / 255.0) - 0.5;
        let speed = 0.6 + ((h >> 24) & 0xFF) as f32 / 255.0 * 2.4;
        engine.scene.world.insert(
            entity,
            Spin {
                axis: [ax, ay + 0.3, az],
                speed,
            },
        );
        engine.scene.world.insert(entity, Orbit { speed: orbit_speed });
    }

    fn run_headless(&mut self) -> Result<()> {
        let frames = self.args.max_frames.unwrap_or(600);
        let mut engine = Engine::new()?;
        Self::seed_scene(&mut engine, self.args.stress_entities);
        tracing::info!(
            entities = engine.scene.names.len(),
            frames,
            "headless perf run starting (CPU extract + cull, no GPU present)"
        );
        let mut last = Instant::now();
        for _ in 0..frames {
            let now = Instant::now();
            let dt = now.duration_since(last);
            last = now;
            engine.tick(dt);
            self.frame_stats.push(dt.as_secs_f32() * 1000.0);
        }
        Self::report_perf(&self.frame_stats, &engine);
        Ok(())
    }

    fn report_perf(frame_stats: &FrameStats, engine: &Engine) {
        let p50 = frame_stats.percentile(50.0);
        let p95 = frame_stats.percentile(95.0);
        let p99 = frame_stats.percentile(99.0);
        tracing::info!(
            entities = engine.scene.names.len(),
            visible = engine.metrics.visible_draws,
            culled_pct = format!("{:.1}", engine.metrics.culled_ratio * 100.0),
            p50_ms = format!("{p50:.3}"),
            p95_ms = format!("{p95:.3}"),
            p99_ms = format!("{p99:.3}"),
            "perf report"
        );
        let csv = format!(
            "metric,value\nentities,{}\nvisible,{}\nculled_pct,{:.3}\np50_ms,{:.4}\np95_ms,{:.4}\np99_ms,{:.4}\n",
            engine.scene.names.len(),
            engine.metrics.visible_draws,
            engine.metrics.culled_ratio * 100.0,
            p50,
            p95,
            p99,
        );
        let _ = std::fs::create_dir_all("assets/processed");
        if let Err(error) = std::fs::write("assets/processed/perf_report.csv", csv) {
            tracing::warn!(?error, "failed to write perf report csv");
        } else {
            tracing::info!("perf report written to assets/processed/perf_report.csv");
        }
    }
}

impl ApplicationHandler for RuntimeApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = WindowAttributes::default()
            .with_title("Habanero Runtime")
            .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0));
        let window = match event_loop.create_window(attrs) {
            Ok(window) => window,
            Err(error) => {
                tracing::error!(?error, "failed to create runtime window");
                event_loop.exit();
                return;
            }
        };

        let mut engine = match if self.args.use_vulkan {
            Engine::new_with_window(&window).or_else(|_| Engine::new())
        } else {
            Engine::new()
        } {
            Ok(engine) => engine,
            Err(error) => {
                tracing::error!(?error, "failed to initialize engine");
                event_loop.exit();
                return;
            }
        };
        if engine.renderer.is_backend_active() {
            tracing::info!("Vulkan backend active");
        } else {
            tracing::warn!("Vulkan backend inactive; running fallback (no GPU presentation)");
        }
        let size = window.inner_size();
        engine.set_aspect(size.width, size.height);
        Self::seed_scene(&mut engine, self.args.stress_entities);
        tracing::info!(
            entities = engine.scene.names.len(),
            "runtime demo scene seeded"
        );

        self.window_id = Some(window.id());
        self.last_tick = Instant::now();
        self.last_title_update = Instant::now();
        self.engine = Some(engine);
        self.window = Some(window);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        if Some(window_id) != self.window_id {
            return;
        }
        match event {
            WindowEvent::CloseRequested => {
                if let Some(engine) = self.engine.as_ref() {
                    let _ = engine.scene.save_json("assets/processed/last_runtime_scene.json");
                }
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                if let Some(engine) = self.engine.as_mut() {
                    engine.resize(size.width, size.height);
                }
            }
            WindowEvent::RedrawRequested => {
                let Some(engine) = self.engine.as_mut() else {
                    return;
                };
                let now = Instant::now();
                let dt = now.duration_since(self.last_tick);
                self.last_tick = now;
                engine.tick(dt);
                self.frame_stats.push(dt.as_secs_f32() * 1000.0);

                if let Some(max) = self.args.max_frames {
                    if engine.frame_index >= max {
                        Self::report_perf(&self.frame_stats, engine);
                        event_loop.exit();
                        return;
                    }
                }

                if now.duration_since(self.last_title_update).as_millis() >= 250 {
                    let p50 = self.frame_stats.percentile(50.0);
                    let p99 = self.frame_stats.percentile(99.0);
                    let backend = if engine.renderer.is_backend_active() {
                        "Vulkan"
                    } else {
                        "Fallback"
                    };
                    if let Some(window) = self.window.as_ref() {
                        window.set_title(&format!(
                            "Habanero [{backend}] | {:.0} FPS | {:.2} ms (p50 {:.2} / p99 {:.2}) | entities {} | visible {} | culled {:.1}%",
                            engine.metrics.fps,
                            engine.metrics.dt_seconds * 1000.0,
                            p50,
                            p99,
                            engine.scene.names.len(),
                            engine.metrics.visible_draws,
                            engine.metrics.culled_ratio * 100.0,
                        ));
                    }
                    self.last_title_update = now;
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse()?))
        .init();

    let args = parse_args(std::env::args().skip(1));

    // Pure headless perf mode does not need a window/GPU surface.
    if args.max_frames.is_some() && !args.use_vulkan {
        let mut app = RuntimeApp::new(args);
        return app.run_headless();
    }

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = RuntimeApp::new(args);
    event_loop.run_app(&mut app)?;
    Ok(())
}

fn parse_args(args: impl Iterator<Item = String>) -> DemoArgs {
    let mut stress_entities = 12_000usize;
    let mut max_frames = None;
    let mut use_vulkan = true;
    let mut iter = args.peekable();
    while let Some(arg) = iter.next() {
        if let Some(raw) = arg.strip_prefix("--stress=") {
            stress_entities = raw.parse::<usize>().unwrap_or(stress_entities).min(200_000);
        } else if arg == "--stress" {
            if let Some(raw) = iter.next() {
                stress_entities = raw.parse::<usize>().unwrap_or(stress_entities).min(200_000);
            }
        } else if let Some(raw) = arg.strip_prefix("--frames=") {
            max_frames = raw.parse::<u64>().ok();
        } else if arg == "--frames" {
            if let Some(raw) = iter.next() {
                max_frames = raw.parse::<u64>().ok();
            }
        } else if arg == "--no-vulkan" {
            use_vulkan = false;
        } else if arg == "--vulkan" {
            use_vulkan = true;
        }
    }
    DemoArgs {
        stress_entities,
        max_frames,
        use_vulkan,
    }
}
