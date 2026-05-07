use anyhow::Result;
use engine_assets::AssetManager;

fn main() -> Result<()> {
    let mut assets = AssetManager::new();
    assets.watch("assets/src")?;
    let handle = assets.load_bytes("scene", "assets/src/sample_scene.json")?;
    let before = assets.get(&handle).map(|b| b.len()).unwrap_or_default();
    assets.reload(handle.id)?;
    let after = assets.get(&handle).map(|b| b.len()).unwrap_or_default();
    tracing::info!(before, after, "hot reload demo complete");
    Ok(())
}
