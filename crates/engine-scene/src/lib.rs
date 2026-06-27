//! Scene graph and serialization.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use engine_ecs::{Entity, Transform, World};
use glam::{Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SceneError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneEntity {
    pub name: String,
    pub transform: Transform,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SceneDocument {
    pub entities: Vec<SceneEntity>,
}

pub struct Scene {
    pub world: World,
    pub names: HashMap<Entity, String>,
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}

impl Scene {
    pub fn new() -> Self {
        Self {
            world: World::new(),
            names: HashMap::new(),
        }
    }

    pub fn spawn_named(&mut self, name: impl Into<String>, transform: Transform) -> Entity {
        let entity = self.world.spawn();
        self.world.insert(entity, transform);
        self.names.insert(entity, name.into());
        entity
    }

    pub fn world_matrices(&self) -> Vec<(Entity, Mat4)> {
        let transforms = self.world.query::<Transform>();
        let has_hierarchy = transforms.iter().any(|(_, t)| t.parent.is_some());
        if !has_hierarchy {
            // Fast path: no parenting, so local == world. Avoids building the
            // per-frame lookup map for flat scenes (the common case).
            return transforms
                .into_iter()
                .map(|(entity, transform)| (entity, transform_to_mat4(&transform)))
                .collect();
        }
        let lookup: HashMap<Entity, Transform> = transforms.iter().cloned().collect();
        transforms
            .into_iter()
            .map(|(entity, transform)| {
                let local = transform_to_mat4(&transform);
                let world = transform
                    .parent
                    .and_then(|parent| lookup.get(&parent))
                    .map_or(local, |parent_t| transform_to_mat4(parent_t) * local);
                (entity, world)
            })
            .collect()
    }

    pub fn to_document(&self) -> SceneDocument {
        let entities = self
            .world
            .query::<Transform>()
            .into_iter()
            .map(|(entity, transform)| SceneEntity {
                name: self
                    .names
                    .get(&entity)
                    .cloned()
                    .unwrap_or_else(|| format!("Entity{}", entity.0)),
                transform,
            })
            .collect();
        SceneDocument { entities }
    }

    pub fn from_document(document: SceneDocument) -> Self {
        let mut scene = Scene::new();
        for entry in document.entities {
            scene.spawn_named(entry.name, entry.transform);
        }
        scene
    }

    pub fn save_json(&self, path: impl AsRef<Path>) -> Result<(), SceneError> {
        let encoded = serde_json::to_string_pretty(&self.to_document())?;
        fs::write(path, encoded)?;
        Ok(())
    }

    pub fn load_json(path: impl AsRef<Path>) -> Result<Self, SceneError> {
        let data = fs::read_to_string(path)?;
        let document: SceneDocument = serde_json::from_str(&data)?;
        Ok(Self::from_document(document))
    }
}

fn transform_to_mat4(transform: &Transform) -> Mat4 {
    let translation = Vec3::from_array(transform.translation);
    let rotation = Quat::from_array(transform.rotation);
    let scale = Vec3::from_array(transform.scale);
    Mat4::from_scale_rotation_translation(scale, rotation, translation)
}
