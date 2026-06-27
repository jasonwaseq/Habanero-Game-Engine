//! ECS throughput smoke test.
//!
//! Exercises bulk spawn/insert, sequential `query`, parallel `par_query`, and
//! the in-place `for_each_mut` write path at a non-trivial entity count so the
//! hot paths are covered by `cargo test`.

use std::time::Instant;

use engine_ecs::{Transform, World};

#[test]
fn ecs_iteration_and_parallel_throughput() {
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
    assert_eq!(world.count::<Transform>(), count as usize);

    let start = Instant::now();
    let mut accum = 0.0f32;
    for (_, t) in world.query::<Transform>() {
        accum += t.translation[0];
    }
    let elapsed = start.elapsed();
    assert!(accum > 0.0);
    println!("sequential iterated {count} transforms in {elapsed:?}");

    // Parallel reduction over the same data must agree with the sequential sum.
    let par_sum: f32 = world
        .par_query::<Transform, f32, _>(|(_, t)| t.translation[0])
        .into_iter()
        .sum();
    assert!((par_sum - accum).abs() < 1.0);

    // In-place mutation under a single write lock.
    world.for_each_mut::<Transform, _>(|_, t| {
        t.translation[1] = 1.0;
    });
    let lifted = world
        .query::<Transform>()
        .into_iter()
        .filter(|(_, t)| (t.translation[1] - 1.0).abs() < f32::EPSILON)
        .count();
    assert_eq!(lifted, count as usize);
}
