use engine_ecs::{Entity, Transform};
use engine_scene::Scene;

#[test]
fn scene_json_roundtrip() {
    let mut scene = Scene::new();
    scene.spawn_named("A", Transform::default());
    scene
        .save_json("scene_roundtrip_test.json")
        .expect("save scene");
    let loaded = Scene::load_json("scene_roundtrip_test.json").expect("load scene");
    assert_eq!(loaded.names.len(), 1);
    std::fs::remove_file("scene_roundtrip_test.json").expect("cleanup");
}

#[test]
fn child_world_matrix_includes_parent_translation() {
    let mut scene = Scene::new();
    let parent = scene.spawn_named(
        "Parent",
        Transform {
            translation: [10.0, 0.0, 0.0],
            ..Default::default()
        },
    );
    scene.spawn_named(
        "Child",
        Transform {
            translation: [1.0, 0.0, 0.0],
            parent: Some(parent),
            ..Default::default()
        },
    );

    let matrices: std::collections::HashMap<Entity, glam::Mat4> =
        scene.world_matrices().into_iter().collect();
    // Child should be offset by parent (10) + local (1) = 11 on X.
    let child_entity = scene
        .names
        .iter()
        .find(|(_, name)| name.as_str() == "Child")
        .map(|(e, _)| *e)
        .expect("child entity");
    let child_world = matrices[&child_entity];
    let translation = child_world.w_axis;
    assert!((translation.x - 11.0).abs() < 1e-4);
}
