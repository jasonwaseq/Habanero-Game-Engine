use std::time::Duration;

use anyhow::Result;
use engine_core::Engine;
use engine_ecs::Transform;
use engine_scene::Scene;
use tracing_subscriber::EnvFilter;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let mut engine = Engine::new()?;
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

    engine.tick(Duration::from_millis(16));
    engine.run_for(180);
    let _ = engine.scene.save_json("assets/processed/last_runtime_scene.json");
    tracing::info!(
        frames = engine.frame_index,
        draw_calls = engine.renderer.draw_calls_last_frame,
        "runtime demo complete"
    );
    Ok(())
}
