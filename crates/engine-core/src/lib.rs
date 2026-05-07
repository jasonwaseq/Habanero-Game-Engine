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
use parking_lot::RwLock;
use rayon::{ThreadPool, ThreadPoolBuilder};

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

pub struct Engine {
    pub scene: Scene,
    pub renderer: VulkanRenderer,
    pub assets: AssetManager,
    pub world: World,
    pub event_bus: EventBus,
    pub frame_index: u64,
    pub job_system: JobSystem,
    pub scripts: ScriptHost,
    events_tx: Sender<EngineEvent>,
    events_rx: Receiver<EngineEvent>,
    plugins: Vec<Arc<dyn EnginePlugin>>,
    running: bool,
}

impl Engine {
    pub fn new() -> Result<Self> {
        let (events_tx, events_rx) = unbounded();
        Ok(Self {
            scene: Scene::new(),
            renderer: VulkanRenderer::new()?,
            assets: AssetManager::new(),
            world: World::new(),
            event_bus: EventBus::default(),
            frame_index: 0,
            job_system: JobSystem::new(num_cpus())?,
            scripts: ScriptHost::new(),
            events_tx,
            events_rx,
            plugins: Vec::new(),
            running: true,
        })
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
        let dt_seconds = delta_time.as_secs_f32();
        self.job_system.scope(|scope| {
            scope.spawn(|_| {
                PhysicsSystem::step(&self.scene.world, dt_seconds);
            });
            scope.spawn(|_| {
                let _ = self.scene.world.changed::<engine_ecs::Transform>();
            });
        });
        let camera = Camera::perspective(16.0 / 9.0, 60f32.to_radians(), 0.1, 2000.0);
        let extracted = self.renderer.extract_scene(&self.scene);
        let visible = self.renderer.cull_visible(&extracted, &camera);
        self.renderer.submit(&visible);
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
