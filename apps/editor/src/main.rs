use anyhow::Result;
use engine_core::{Engine, ScheduleGraph};
use engine_editor::EditorState;
use engine_ecs::Transform;
use tracing_subscriber::EnvFilter;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let mut engine = Engine::new()?;
    engine.scene.spawn_named("EditorCamera", Transform::default());
    engine.scene.spawn_named("DirectionalLight", Transform::default());

    let mut editor = EditorState::default();
    let schedule = ScheduleGraph::default();
    editor.update_schedule_view(schedule.stages.read().clone());
    let hierarchy = editor.hierarchy_items(&engine.scene);
    editor.stats.entities = hierarchy.len();
    editor.stats.draw_calls = engine.renderer.draw_calls_last_frame;
    editor.stats.frame_time_ms = engine.renderer.frame_time_ms;
    editor.stats.fps = if editor.stats.frame_time_ms > 0.0 {
        1000.0 / editor.stats.frame_time_ms
    } else {
        0.0
    };
    tracing::info!(entities = editor.stats.entities, "editor shell initialized");
    Ok(())
}
