use std::time::Instant;

use anyhow::Result;
use engine_core::Engine;
use engine_ecs::Transform;
use engine_scene::Scene;
use tracing_subscriber::EnvFilter;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

struct RuntimeApp {
    window: Option<Window>,
    window_id: Option<WindowId>,
    engine: Option<Engine>,
    last_tick: Instant,
    last_title_update: Instant,
    stress_entities: usize,
}

impl RuntimeApp {
    fn new(stress_entities: usize) -> Self {
        Self {
            window: None,
            window_id: None,
            engine: None,
            last_tick: Instant::now(),
            last_title_update: Instant::now(),
            stress_entities,
        }
    }

    fn seed_scene(engine: &mut Engine, stress_entities: usize) {
        let enable_mesh_path = std::env::var("HBN_ENABLE_MESH_PATH")
            .ok()
            .as_deref()
            == Some("1");
        if enable_mesh_path {
            let mesh_assets = engine
                .assets
                .load_gltf_meshes("assets/src/sample_mesh.gltf")
                .unwrap_or_else(|error| {
                    tracing::warn!(?error, "failed to load sample gltf mesh, using fallback cube");
                    vec![engine.assets.fallback_cube_mesh()]
                });
            for mesh in mesh_assets {
                let mesh_id = engine.renderer.register_mesh(mesh.clone());
                tracing::info!(mesh = mesh.name, mesh_id = mesh_id.0, "registered render mesh");
            }
        } else {
            tracing::warn!(
                "mesh asset path disabled by default; set HBN_ENABLE_MESH_PATH=1 to enable experimental milestone #1"
            );
        }
        match Scene::load_json("assets/src/sample_scene.json") {
            Ok(scene) => engine.scene = scene,
            Err(_) => {
                engine.scene.spawn_named(
                    "CameraRig",
                    Transform {
                        translation: [0.0, 2.0, -8.0],
                        ..Default::default()
                    },
                );
                engine.scene.spawn_named("SpriteBatchTest", Transform::default());
                engine.scene.spawn_named(
                    "MeshPbrSphere",
                    Transform {
                        translation: [1.5, 0.5, 2.0],
                        ..Default::default()
                    },
                );
            }
        }
        if stress_entities > 0 {
            let grid = (stress_entities as f32).sqrt().ceil() as usize;
            let spacing = 0.35_f32;
            let half = grid as f32 * spacing * 0.5;
            let mut placed = 0usize;
            for x in 0..grid {
                for z in 0..grid {
                    if placed >= stress_entities {
                        return;
                    }
                    engine.scene.spawn_named(
                        format!("StressInstance{placed}"),
                        Transform {
                            translation: [
                                x as f32 * spacing - half,
                                (placed % 7) as f32 * 0.03,
                                z as f32 * spacing,
                            ],
                            ..Default::default()
                        },
                    );
                    placed += 1;
                }
            }
        }
    }
}

impl ApplicationHandler for RuntimeApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = WindowAttributes::default().with_title("Habanero Runtime");
        let window = match event_loop.create_window(attrs) {
            Ok(window) => window,
            Err(error) => {
                tracing::error!(?error, "failed to create runtime window");
                event_loop.exit();
                return;
            }
        };

        let use_vulkan = std::env::var("HBN_ENABLE_VULKAN")
            .ok()
            .as_deref()
            == Some("1");
        let mut engine = match if use_vulkan {
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
        if !use_vulkan {
            tracing::warn!("Vulkan disabled by default; set HBN_ENABLE_VULKAN=1 to test Vulkan backend");
        }
        if engine.renderer.is_backend_active() {
            window.set_title("Habanero Runtime [Vulkan]");
            tracing::info!("Vulkan backend active");
        } else {
            window.set_title("Habanero Runtime [Fallback Renderer]");
            tracing::warn!("Vulkan backend inactive; running fallback renderer");
        }
        Self::seed_scene(&mut engine, self.stress_entities);
        tracing::info!(stress_entities = self.stress_entities, "runtime demo scene seeded");

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
            WindowEvent::RedrawRequested => {
                if let Some(engine) = self.engine.as_mut() {
                    let now = Instant::now();
                    let dt = now.duration_since(self.last_tick);
                    self.last_tick = now;
                    engine.tick(dt);
                    if now.duration_since(self.last_title_update).as_millis() >= 250 {
                        if let Some(window) = self.window.as_ref() {
                            let stats = engine.renderer.stats;
                            let cull_pct = if stats.extracted_draws > 0 {
                                100.0
                                    * (1.0
                                        - (stats.visible_draws as f32 / stats.extracted_draws as f32))
                            } else {
                                0.0
                            };
                            let backend = if engine.renderer.is_backend_active() {
                                "Vulkan"
                            } else {
                                "Fallback"
                            };
                            window.set_title(&format!(
                                "Habanero Runtime [{backend}] | FPS {:.1} | {:.2} ms | entities {} | draws {} | culled {:.1}%",
                                engine.metrics.fps,
                                engine.metrics.dt_seconds * 1000.0,
                                engine.scene.names.len(),
                                engine.renderer.draw_calls_last_frame,
                                cull_pct
                            ));
                        }
                        self.last_title_update = now;
                    }
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
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let stress_entities = parse_stress_entities(std::env::args().skip(1));
    let mut app = RuntimeApp::new(stress_entities);
    event_loop.run_app(&mut app)?;
    Ok(())
}

fn parse_stress_entities(args: impl Iterator<Item = String>) -> usize {
    let mut value = 10_000usize;
    let mut iter = args.peekable();
    while let Some(arg) = iter.next() {
        if arg == "--stress" {
            if let Some(raw) = iter.next() {
                if let Ok(parsed) = raw.parse::<usize>() {
                    value = parsed.min(200_000);
                }
            }
        } else if let Some(raw) = arg.strip_prefix("--stress=") {
            if let Ok(parsed) = raw.parse::<usize>() {
                value = parsed.min(200_000);
            }
        }
    }
    value
}
