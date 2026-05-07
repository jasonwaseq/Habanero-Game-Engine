use anyhow::Result;
use engine_ecs::{Transform, World};

fn main() -> Result<()> {
    let world = World::new();
    let count = 250_000u64;
    for i in 0..count {
        let entity = world.spawn();
        world.insert(
            entity,
            Transform {
                translation: [i as f32, 0.0, 0.0],
                ..Default::default()
            },
        );
    }

    let mut sum = 0.0;
    for (_, t) in world.query::<Transform>() {
        sum += t.translation[0];
    }
    tracing::info!(entities = count, sum, "ecs stress finished");
    Ok(())
}
