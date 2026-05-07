use anyhow::Result;
use engine_core::Engine;
use engine_ecs::Transform;

fn main() -> Result<()> {
    let mut engine = Engine::new()?;
    engine.scene.spawn_named(
        "MainCamera",
        Transform {
            translation: [0.0, 2.0, -7.0],
            ..Default::default()
        },
    );
    engine.scene.spawn_named(
        "SunLight",
        Transform {
            translation: [4.0, 6.0, 2.0],
            ..Default::default()
        },
    );
    for i in 0..100 {
        engine.scene.spawn_named(
            format!("MeshInstance{i}"),
            Transform {
                translation: [i as f32 * 0.2, 0.0, 1.0 + (i % 10) as f32 * 0.5],
                ..Default::default()
            },
        );
    }
    engine.run_for(120);
    tracing::info!(
        entities = engine.scene.names.len(),
        draw_calls = engine.renderer.draw_calls_last_frame,
        "3d demo complete"
    );
    Ok(())
}
