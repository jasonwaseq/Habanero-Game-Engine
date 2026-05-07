use anyhow::Result;
use engine_core::Engine;
use engine_ecs::Transform;

fn main() -> Result<()> {
    let mut engine = Engine::new()?;
    for x in -256..256 {
        let entity = engine.scene.world.spawn();
        engine.scene.world.insert(
            entity,
            Transform {
                translation: [x as f32 * 0.2, 0.0, 0.0],
                ..Default::default()
            },
        );
        engine
            .scene
            .names
            .insert(entity, format!("Sprite{x}"));
    }
    engine.run_for(60);
    tracing::info!(
        draw_calls = engine.renderer.draw_calls_last_frame,
        entities = engine.scene.names.len(),
        "sprite demo complete"
    );
    Ok(())
}
