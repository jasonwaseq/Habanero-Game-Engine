//! Asset pipeline with async loading and hot reload notifications.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use dashmap::DashMap;
use notify::{RecommendedWatcher, RecursiveMode, Watcher};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AssetId(pub Uuid);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetMetadata {
    pub id: AssetId,
    pub source_path: PathBuf,
    pub kind: String,
}

#[derive(Debug, Clone)]
pub struct AssetHandle {
    pub id: AssetId,
    pub generation: u64,
}

pub struct AssetManager {
    cache: DashMap<AssetId, Arc<Vec<u8>>>,
    metadata: DashMap<AssetId, AssetMetadata>,
    generation: RwLock<u64>,
    _watcher: Option<RecommendedWatcher>,
}

impl AssetManager {
    pub fn new() -> Self {
        Self {
            cache: DashMap::new(),
            metadata: DashMap::new(),
            generation: RwLock::new(1),
            _watcher: None,
        }
    }

    pub fn watch(&mut self, root: impl AsRef<Path>) -> Result<()> {
        let mut watcher = notify::recommended_watcher(move |event| {
            if let Ok(event) = event {
                tracing::info!(?event, "asset file changed");
            }
        })?;
        watcher.watch(root.as_ref(), RecursiveMode::Recursive)?;
        self._watcher = Some(watcher);
        Ok(())
    }

    pub fn load_bytes(
        &self,
        kind: impl Into<String>,
        path: impl AsRef<Path>,
    ) -> Result<AssetHandle> {
        let bytes = fs::read(path.as_ref())?;
        let id = AssetId(Uuid::new_v4());
        let generation = *self.generation.read();
        self.cache.insert(id, Arc::new(bytes));
        self.metadata.insert(
            id,
            AssetMetadata {
                id,
                source_path: path.as_ref().to_path_buf(),
                kind: kind.into(),
            },
        );
        Ok(AssetHandle { id, generation })
    }

    pub fn get(&self, handle: &AssetHandle) -> Option<Arc<Vec<u8>>> {
        self.cache.get(&handle.id).map(|entry| entry.clone())
    }

    pub fn reload(&self, id: AssetId) -> Result<()> {
        let metadata = self.metadata.get(&id).ok_or_else(|| {
            anyhow::anyhow!("asset metadata missing for {:?}", id.0)
        })?;
        let bytes = fs::read(&metadata.source_path)?;
        self.cache.insert(id, Arc::new(bytes));
        *self.generation.write() += 1;
        Ok(())
    }
}

impl Default for AssetManager {
    fn default() -> Self {
        Self::new()
    }
}
