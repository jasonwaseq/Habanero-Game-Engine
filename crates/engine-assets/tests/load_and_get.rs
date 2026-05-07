use engine_assets::AssetManager;

#[test]
fn load_and_get_bytes() {
    std::fs::write("asset_test.bin", [1u8, 2, 3, 4]).expect("write fixture");
    let manager = AssetManager::new();
    let handle = manager
        .load_bytes("binary", "asset_test.bin")
        .expect("load bytes");
    let bytes = manager.get(&handle).expect("cached bytes");
    assert_eq!(bytes.as_slice(), &[1, 2, 3, 4]);
    std::fs::remove_file("asset_test.bin").expect("cleanup");
}
