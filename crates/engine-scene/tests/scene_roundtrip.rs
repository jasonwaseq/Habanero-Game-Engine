use engine_ecs::Transform;
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
