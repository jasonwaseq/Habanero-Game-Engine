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

#[repr(C)]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MeshVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshAsset {
    pub name: String,
    pub vertices: Vec<MeshVertex>,
    pub indices: Vec<u32>,
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

    pub fn load_gltf_meshes(&self, path: impl AsRef<Path>) -> Result<Vec<MeshAsset>> {
        let (document, buffers, _) = gltf::import(path.as_ref())?;
        let mut meshes = Vec::new();
        for mesh in document.meshes() {
            for primitive in mesh.primitives() {
                let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
                let positions = reader
                    .read_positions()
                    .ok_or_else(|| anyhow::anyhow!("gltf primitive missing positions"))?
                    .collect::<Vec<_>>();
                let normals = reader
                    .read_normals()
                    .map(|iter| iter.collect::<Vec<_>>())
                    .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; positions.len()]);
                let uvs = reader
                    .read_tex_coords(0)
                    .map(|coords| coords.into_f32().collect::<Vec<_>>())
                    .unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);
                let vertices = positions
                    .iter()
                    .enumerate()
                    .map(|(idx, pos)| MeshVertex {
                        position: *pos,
                        normal: normals.get(idx).copied().unwrap_or([0.0, 1.0, 0.0]),
                        uv: uvs.get(idx).copied().unwrap_or([0.0, 0.0]),
                    })
                    .collect::<Vec<_>>();
                let indices = reader
                    .read_indices()
                    .map(|indices| indices.into_u32().collect::<Vec<_>>())
                    .unwrap_or_else(|| (0..vertices.len() as u32).collect::<Vec<_>>());
                meshes.push(MeshAsset {
                    name: mesh
                        .name()
                        .map(ToString::to_string)
                        .unwrap_or_else(|| format!("mesh_{}", mesh.index())),
                    vertices,
                    indices,
                });
            }
        }
        if meshes.is_empty() {
            return Err(anyhow::anyhow!("no meshes found in gltf"));
        }
        Ok(meshes)
    }

    pub fn fallback_cube_mesh(&self) -> MeshAsset {
        // Minimal indexed cube to keep demo rendering even without external assets.
        let verts = vec![
            MeshVertex { position: [-0.5, -0.5, -0.5], normal: [0.0, 0.0, -1.0], uv: [0.0, 0.0] },
            MeshVertex { position: [0.5, -0.5, -0.5], normal: [0.0, 0.0, -1.0], uv: [1.0, 0.0] },
            MeshVertex { position: [0.5, 0.5, -0.5], normal: [0.0, 0.0, -1.0], uv: [1.0, 1.0] },
            MeshVertex { position: [-0.5, 0.5, -0.5], normal: [0.0, 0.0, -1.0], uv: [0.0, 1.0] },
            MeshVertex { position: [-0.5, -0.5, 0.5], normal: [0.0, 0.0, 1.0], uv: [0.0, 0.0] },
            MeshVertex { position: [0.5, -0.5, 0.5], normal: [0.0, 0.0, 1.0], uv: [1.0, 0.0] },
            MeshVertex { position: [0.5, 0.5, 0.5], normal: [0.0, 0.0, 1.0], uv: [1.0, 1.0] },
            MeshVertex { position: [-0.5, 0.5, 0.5], normal: [0.0, 0.0, 1.0], uv: [0.0, 1.0] },
        ];
        let idx = vec![
            0, 1, 2, 2, 3, 0, // back
            4, 6, 5, 6, 4, 7, // front
            0, 4, 5, 5, 1, 0, // bottom
            3, 2, 6, 6, 7, 3, // top
            1, 5, 6, 6, 2, 1, // right
            0, 3, 7, 7, 4, 0, // left
        ];
        MeshAsset {
            name: "fallback_cube".to_string(),
            vertices: verts,
            indices: idx,
        }
    }
}

impl Default for AssetManager {
    fn default() -> Self {
        Self::new()
    }
}
