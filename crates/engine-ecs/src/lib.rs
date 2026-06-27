//! High-performance ECS core for the Habanero engine.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::RwLock;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Entity identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Entity(pub u64);

/// Marker trait for components.
pub trait Component: Send + Sync + 'static {}

impl<T> Component for T where T: Send + Sync + 'static {}

#[derive(Debug, Error)]
pub enum EcsError {
    #[error("component storage missing for type")]
    MissingStorage,
}

/// Sparse-set component storage.
#[derive(Default)]
struct SparseSet<T: Component> {
    dense: Vec<T>,
    entities: Vec<Entity>,
    sparse: HashMap<Entity, usize>,
    changed: Vec<bool>,
}

impl<T: Component> SparseSet<T> {
    fn insert(&mut self, entity: Entity, value: T) {
        if let Some(index) = self.sparse.get(&entity).copied() {
            self.dense[index] = value;
            self.changed[index] = true;
            return;
        }
        let idx = self.dense.len();
        self.dense.push(value);
        self.entities.push(entity);
        self.changed.push(true);
        self.sparse.insert(entity, idx);
    }

    fn remove(&mut self, entity: Entity) {
        let Some(removed) = self.sparse.remove(&entity) else {
            return;
        };
        let last = self.dense.len() - 1;
        if removed != last {
            self.dense.swap(removed, last);
            self.entities.swap(removed, last);
            self.changed.swap(removed, last);
            let moved = self.entities[removed];
            self.sparse.insert(moved, removed);
        }
        self.dense.pop();
        self.entities.pop();
        self.changed.pop();
    }
}

trait ErasedStorage: Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn remove_entity(&mut self, entity: Entity);
}

impl<T: Component> ErasedStorage for SparseSet<T> {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn remove_entity(&mut self, entity: Entity) {
        self.remove(entity);
    }
}

/// ECS world containing all entities and component storages.
#[derive(Default)]
pub struct World {
    next_entity: AtomicU64,
    entities: RwLock<Vec<Entity>>,
    storages: RwLock<HashMap<TypeId, Box<dyn ErasedStorage>>>,
}

impl World {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn spawn(&self) -> Entity {
        let id = self.next_entity.fetch_add(1, Ordering::Relaxed);
        let entity = Entity(id);
        self.entities.write().push(entity);
        entity
    }

    pub fn despawn(&self, entity: Entity) {
        self.entities.write().retain(|e| *e != entity);
        for storage in self.storages.write().values_mut() {
            storage.remove_entity(entity);
        }
    }

    pub fn insert<T: Component>(&self, entity: Entity, value: T) {
        let mut storages = self.storages.write();
        let storage = storages
            .entry(TypeId::of::<T>())
            .or_insert_with(|| {
                Box::new(SparseSet::<T> {
                    dense: Vec::new(),
                    entities: Vec::new(),
                    sparse: HashMap::new(),
                    changed: Vec::new(),
                })
            });
        let set = storage
            .as_any_mut()
            .downcast_mut::<SparseSet<T>>()
            .expect("storage type mismatch");
        set.insert(entity, value);
    }

    pub fn get<T: Component + Clone>(&self, entity: Entity) -> Option<T> {
        let storages = self.storages.read();
        let storage = storages.get(&TypeId::of::<T>())?;
        let set = storage.as_any().downcast_ref::<SparseSet<T>>()?;
        let idx = *set.sparse.get(&entity)?;
        Some(set.dense[idx].clone())
    }

    pub fn query<T: Component + Clone>(&self) -> Vec<(Entity, T)> {
        let storages = self.storages.read();
        let Some(storage) = storages.get(&TypeId::of::<T>()) else {
            return Vec::new();
        };
        let set = storage
            .as_any()
            .downcast_ref::<SparseSet<T>>()
            .expect("storage type mismatch");
        set.entities
            .iter()
            .enumerate()
            .map(|(idx, entity)| (*entity, set.dense[idx].clone()))
            .collect()
    }

    /// Parallel query execution for embarrassingly parallel jobs.
    pub fn par_query<T, R, F>(&self, f: F) -> Vec<R>
    where
        T: Component + Clone,
        R: Send,
        F: Fn((Entity, T)) -> R + Send + Sync,
    {
        self.query::<T>().into_par_iter().map(f).collect()
    }

    /// Mutate every component of type `T` in place under a single write lock.
    ///
    /// This is the high-throughput write path: it avoids the per-entity
    /// clone+reinsert round-trip of `query` + `insert` and marks every touched
    /// component as changed for downstream change-detection. The callback must
    /// not call back into the same world (the storage map is locked).
    pub fn for_each_mut<T, F>(&self, mut f: F)
    where
        T: Component,
        F: FnMut(Entity, &mut T),
    {
        let mut storages = self.storages.write();
        let Some(storage) = storages.get_mut(&TypeId::of::<T>()) else {
            return;
        };
        let set = storage
            .as_any_mut()
            .downcast_mut::<SparseSet<T>>()
            .expect("storage type mismatch");
        for idx in 0..set.dense.len() {
            f(set.entities[idx], &mut set.dense[idx]);
            set.changed[idx] = true;
        }
    }

    /// Number of components currently stored for type `T`.
    pub fn count<T: Component>(&self) -> usize {
        let storages = self.storages.read();
        storages
            .get(&TypeId::of::<T>())
            .and_then(|storage| storage.as_any().downcast_ref::<SparseSet<T>>())
            .map(|set| set.dense.len())
            .unwrap_or(0)
    }

    pub fn clear_change_flags<T: Component>(&self) -> Result<(), EcsError> {
        let mut storages = self.storages.write();
        let Some(storage) = storages.get_mut(&TypeId::of::<T>()) else {
            return Err(EcsError::MissingStorage);
        };
        let set = storage
            .as_any_mut()
            .downcast_mut::<SparseSet<T>>()
            .expect("storage type mismatch");
        set.changed.fill(false);
        Ok(())
    }

    pub fn changed<T: Component + Clone>(&self) -> Vec<(Entity, T)> {
        let storages = self.storages.read();
        let Some(storage) = storages.get(&TypeId::of::<T>()) else {
            return Vec::new();
        };
        let set = storage
            .as_any()
            .downcast_ref::<SparseSet<T>>()
            .expect("storage type mismatch");
        set.entities
            .iter()
            .enumerate()
            .filter(|(idx, _)| set.changed[*idx])
            .map(|(idx, entity)| (*entity, set.dense[idx].clone()))
            .collect()
    }
}

/// Transform component supporting hierarchy references.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transform {
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
    pub parent: Option<Entity>,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            translation: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0],
            parent: None,
        }
    }
}

/// Lightweight event bus used to decouple systems.
#[derive(Default)]
pub struct EventBus {
    events: RwLock<Vec<Box<dyn Any + Send + Sync>>>,
}

impl EventBus {
    pub fn push<E: Send + Sync + 'static>(&self, event: E) {
        self.events.write().push(Box::new(event));
    }

    pub fn drain<E: Clone + Send + Sync + 'static>(&self) -> Vec<E> {
        let mut events = self.events.write();
        let mut out = Vec::new();
        events.retain(|event| {
            if let Some(data) = event.downcast_ref::<E>() {
                out.push(data.clone());
                false
            } else {
                true
            }
        });
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_get_remove_component() {
        let world = World::new();
        let entity = world.spawn();
        world.insert(entity, Transform::default());
        assert!(world.get::<Transform>(entity).is_some());
        world.despawn(entity);
        assert!(world.get::<Transform>(entity).is_none());
    }

    #[test]
    fn change_detection_marks_written_components() {
        let world = World::new();
        let entity = world.spawn();
        world.insert(entity, Transform::default());
        let changed = world.changed::<Transform>();
        assert_eq!(changed.len(), 1);
        world.clear_change_flags::<Transform>().expect("clear flags");
        assert!(world.changed::<Transform>().is_empty());
    }

    #[test]
    fn for_each_mut_updates_in_place_and_marks_changed() {
        let world = World::new();
        let e = world.spawn();
        world.insert(e, Transform::default());
        world.clear_change_flags::<Transform>().expect("clear flags");
        world.for_each_mut::<Transform, _>(|_, t| {
            t.translation[0] = 5.0;
        });
        let t = world.get::<Transform>(e).expect("transform");
        assert_eq!(t.translation[0], 5.0);
        assert_eq!(world.changed::<Transform>().len(), 1);
    }

    #[test]
    fn count_reflects_storage_size() {
        let world = World::new();
        assert_eq!(world.count::<Transform>(), 0);
        for _ in 0..10 {
            let e = world.spawn();
            world.insert(e, Transform::default());
        }
        assert_eq!(world.count::<Transform>(), 10);
    }

    #[test]
    fn swap_remove_keeps_sparse_indices_consistent() {
        let world = World::new();
        let a = world.spawn();
        let b = world.spawn();
        let c = world.spawn();
        world.insert(a, Transform { translation: [1.0, 0.0, 0.0], ..Default::default() });
        world.insert(b, Transform { translation: [2.0, 0.0, 0.0], ..Default::default() });
        world.insert(c, Transform { translation: [3.0, 0.0, 0.0], ..Default::default() });
        world.despawn(b);
        assert!(world.get::<Transform>(b).is_none());
        assert_eq!(world.get::<Transform>(a).expect("a").translation[0], 1.0);
        assert_eq!(world.get::<Transform>(c).expect("c").translation[0], 3.0);
        assert_eq!(world.count::<Transform>(), 2);
    }

    #[test]
    fn event_bus_drains_matching_type_only() {
        let bus = EventBus::default();
        bus.push(7u32);
        bus.push(String::from("hello"));
        let ints = bus.drain::<u32>();
        assert_eq!(ints, vec![7]);
        // The string event remains for its own consumer.
        assert_eq!(bus.drain::<String>(), vec!["hello".to_string()]);
    }
}
