//! End-to-end engine tick test (headless, no GPU backend).

use std::time::Duration;

use engine_core::Engine;
use engine_ecs::Transform;
use engine_physics::{Orbit, Spin};

#[test]
fn tick_advances_time_and_simulates_motion() {
    let mut engine = Engine::new().expect("engine init");

    let orbiting = engine.scene.spawn_named(
        "Orbiter",
        Transform {
            translation: [5.0, 0.0, 0.0],
            ..Default::default()
        },
    );
    engine.scene.world.insert(orbiting, Orbit { speed: 1.0 });
    engine.scene.world.insert(
        orbiting,
        Spin {
            axis: [0.0, 1.0, 0.0],
            speed: 2.0,
        },
    );

    for _ in 0..30 {
        engine.tick(Duration::from_millis(16));
    }

    assert!(engine.elapsed_seconds > 0.0);
    assert_eq!(engine.frame_index, 30);
    assert!(engine.metrics.fps > 0.0);

    // The orbiting entity must have moved off the +X axis but kept its radius.
    let t = engine
        .scene
        .world
        .get::<Transform>(orbiting)
        .expect("transform");
    let radius = (t.translation[0].powi(2) + t.translation[2].powi(2)).sqrt();
    assert!((radius - 5.0).abs() < 0.05);
    assert!(t.translation[2].abs() > 0.01);
}

#[test]
fn camera_orbits_over_time() {
    let mut engine = Engine::new().expect("engine init");
    let first = engine.current_camera().view;
    engine.tick(Duration::from_millis(500));
    let later = engine.current_camera().view;
    assert_ne!(first, later);
}

#[test]
fn metrics_report_culling_on_empty_scene() {
    let mut engine = Engine::new().expect("engine init");
    engine.tick(Duration::from_millis(16));
    // No entities -> nothing visible, culled ratio defined and finite.
    assert_eq!(engine.metrics.visible_draws, 0);
    assert!(engine.metrics.culled_ratio.is_finite());
}
