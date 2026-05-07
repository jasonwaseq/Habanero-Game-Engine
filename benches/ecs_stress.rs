use std::time::Instant;

use engine_ecs::{Transform, World};

#[test]
fn ecs_iteration_stress() {
    let world = World::new();
    let count = 100_000u64;
    for i in 0..count {
        let e = world.spawn();
        world.insert(
            e,
            Transform {
                translation: [i as f32, 0.0, 0.0],
                ..Default::default()
            },
        );
    }

    let start = Instant::now();
    let mut accum = 0.0f32;
    for (_, t) in world.query::<Transform>() {
        accum += t.translation[0];
    }
    let elapsed = start.elapsed();
    assert!(accum > 0.0);
    println!("iterated {count} transforms in {:?}", elapsed);
}
