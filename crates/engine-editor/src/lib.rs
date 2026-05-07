//! Editor data model and panel state.

use engine_ecs::Entity;
use engine_scene::Scene;
use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct RuntimeStats {
    pub fps: f32,
    pub frame_time_ms: f32,
    pub draw_calls: usize,
    pub entities: usize,
}

#[derive(Default)]
pub struct EditorState {
    pub selected: Option<Entity>,
    pub stats: RuntimeStats,
    pub schedule_stages: Vec<&'static str>,
}

impl EditorState {
    pub fn hierarchy_items(&self, scene: &Scene) -> Vec<(Entity, String)> {
        scene.names.iter().map(|(e, n)| (*e, n.clone())).collect()
    }

    pub fn update_schedule_view(&mut self, stages: Vec<&'static str>) {
        self.schedule_stages = stages;
    }
}
