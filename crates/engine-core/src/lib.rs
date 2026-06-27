//! Engine lifecycle and scheduler.

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use crossbeam_channel::{unbounded, Receiver, Sender};
use engine_assets::AssetManager;
use engine_ecs::{EventBus, World};
use engine_physics::PhysicsSystem;
use engine_render::{Camera, VulkanRenderer};
use engine_scene::Scene;
use engine_scripting::ScriptHost;
use glam::Vec3;
use parking_lot::RwLock;
use rayon::{ThreadPool, ThreadPoolBuilder};
use winit::window::Window;

pub trait EnginePlugin: Send + Sync {
    fn name(&self) -> &'static str;
    fn build(&self, _engine: &mut Engine) -> Result<()> {
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub enum EngineEvent {
    ExitRequested,
}

pub struct JobSystem {
    pool: ThreadPool,
}

impl JobSystem {
    pub fn new(workers: usize) -> Result<Self> {
        let pool = ThreadPoolBuilder::new().num_threads(workers).build()?;
        Ok(Self { pool })
    }

    pub fn scope<'scope, F>(&'scope self, f: F)
    where
        F: FnOnce(&rayon::Scope<'scope>) + Send,
    {
        self.pool.scope(f);
    }
}

/// Orbiting demo camera configuration. The camera circles `target` at
/// `radius`/`height`, providing an automatic, hands-free flythrough.
#[derive(Debug, Clone, Copy)]
pub struct CameraController {
    pub target: [f32; 3],
    pub radius: f32,
    pub height: f32,
    pub orbit_speed: f32,
    pub fov_y_radians: f32,
    pub aspect: f32,
}

impl Default for CameraController {
    fn default() -> Self {
        Self {
            target: [0.0, 0.0, 0.0],
            radius: 26.0,
            height: 14.0,
            orbit_speed: 0.2,
            fov_y_radians: 60f32.to_radians(),
            aspect: 16.0 / 9.0,
        }
    }
}

pub struct Engine {
    pub scene: Scene,
    pub renderer: VulkanRenderer,
    pub assets: AssetManager,
    pub world: World,
    pub event_bus: EventBus,
    pub frame_index: u64,
    pub metrics: FrameMetrics,
    pub job_system: JobSystem,
    pub scripts: ScriptHost,
    pub camera: CameraController,
    pub elapsed_seconds: f32,
    events_tx: Sender<EngineEvent>,
    events_rx: Receiver<EngineEvent>,
    plugins: Vec<Arc<dyn EnginePlugin>>,
    running: bool,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct FrameMetrics {
    pub dt_seconds: f32,
    pub fps: f32,
    pub visible_draws: usize,
    pub culled_ratio: f32,
}

impl Engine {
    pub fn new() -> Result<Self> {
        Self::new_inner(VulkanRenderer::new()?)
    }

    pub fn new_with_window(window: &Window) -> Result<Self> {
        Self::new_inner(VulkanRenderer::new_with_window(window)?)
    }

    fn new_inner(renderer: VulkanRenderer) -> Result<Self> {
        let (events_tx, events_rx) = unbounded();
        Ok(Self {
            scene: Scene::new(),
            renderer,
            assets: AssetManager::new(),
            world: World::new(),
            event_bus: EventBus::default(),
            frame_index: 0,
            metrics: FrameMetrics::default(),
            job_system: JobSystem::new(num_cpus())?,
            scripts: ScriptHost::new(),
            camera: CameraController::default(),
            elapsed_seconds: 0.0,
            events_tx,
            events_rx,
            plugins: Vec::new(),
            running: true,
        })
    }

    /// Update the camera aspect ratio for the given surface size without
    /// touching GPU resources. Used for initial setup.
    pub fn set_aspect(&mut self, width: u32, height: u32) {
        if width != 0 && height != 0 {
            self.camera.aspect = width as f32 / height as f32;
        }
    }

    /// Update the camera aspect ratio and recreate the renderer surface after a
    /// window resize.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.camera.aspect = width as f32 / height as f32;
        self.renderer.resize();
    }

    /// Build the current frame's orbiting camera.
    pub fn current_camera(&self) -> Camera {
        let angle = self.elapsed_seconds * self.camera.orbit_speed;
        let target = Vec3::from_array(self.camera.target);
        let eye = target
            + Vec3::new(
                angle.cos() * self.camera.radius,
                self.camera.height,
                angle.sin() * self.camera.radius,
            );
        Camera::looking_at(eye, target, self.camera.aspect, self.camera.fov_y_radians)
    }

    pub fn register_plugin<P: EnginePlugin + 'static>(&mut self, plugin: P) -> Result<()> {
        plugin.build(self)?;
        self.plugins.push(Arc::new(plugin));
        Ok(())
    }

    pub fn event_sender(&self) -> Sender<EngineEvent> {
        self.events_tx.clone()
    }

    pub fn tick(&mut self, delta_time: Duration) {
        let dt_seconds = delta_time.as_secs_f32().min(0.1);
        self.metrics.dt_seconds = dt_seconds;
        if dt_seconds > 0.000_001 {
            self.metrics.fps = 1.0 / dt_seconds;
        }
        self.elapsed_seconds += dt_seconds;

        // Advance simulation on the job system to demonstrate off-main-thread
        // frame preparation, then read back the change set.
        self.job_system.scope(|scope| {
            scope.spawn(|_| {
                PhysicsSystem::step(&self.scene.world, dt_seconds);
            });
        });
        let _changed = self.scene.world.changed::<engine_ecs::Transform>();

        let camera = self.current_camera();
        let light_dir = Vec3::new(
            (self.elapsed_seconds * 0.35).cos() * 0.6,
            -1.0,
            (self.elapsed_seconds * 0.35).sin() * 0.6,
        )
        .normalize();

        let extracted = self.renderer.extract_scene(&self.scene);
        let visible = self.renderer.cull_visible(&extracted, &camera);
        let extracted_count = extracted.draw_packets.len();
        let visible_count = visible.draw_packets.len();
        self.renderer.update_stats(extracted_count, visible_count);
        self.metrics.visible_draws = visible_count;
        self.metrics.culled_ratio = if extracted_count > 0 {
            1.0 - (visible_count as f32 / extracted_count as f32)
        } else {
            0.0
        };
        self.renderer.submit(&visible, &camera, light_dir);
        self.frame_index = self.frame_index.saturating_add(1);
        self.event_bus.push(delta_time);
    }

    pub fn run_for(&mut self, frames: u64) {
        let mut last = Instant::now();
        while self.running && self.frame_index < frames {
            if let Ok(EngineEvent::ExitRequested) = self.events_rx.try_recv() {
                self.running = false;
                break;
            }
            let now = Instant::now();
            let dt = now.duration_since(last);
            last = now;
            self.tick(dt);
        }
    }
}

pub struct ScheduleGraph {
    pub stages: Arc<RwLock<Vec<&'static str>>>,
}

impl Default for ScheduleGraph {
    fn default() -> Self {
        Self {
            stages: Arc::new(RwLock::new(vec![
                "input",
                "gameplay",
                "physics",
                "animation",
                "render_extract",
                "render_submit",
                "editor",
            ])),
        }
    }
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(4)
}
