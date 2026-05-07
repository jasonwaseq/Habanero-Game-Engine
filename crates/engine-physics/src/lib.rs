//! Physics integration layer.

use engine_ecs::{Entity, Transform, World};
use glam::Vec3;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Velocity {
    pub linear: [f32; 3],
}

impl Default for Velocity {
    fn default() -> Self {
        Self {
            linear: [0.0, 0.0, 0.0],
        }
    }
}

pub struct PhysicsSystem;

impl PhysicsSystem {
    pub fn step(world: &World, dt_seconds: f32) {
        let transforms = world.query::<Transform>();
        let velocities = world.query::<Velocity>();
        let velocity_map: std::collections::HashMap<Entity, Velocity> = velocities.into_iter().collect();
        for (entity, mut transform) in transforms {
            if let Some(velocity) = velocity_map.get(&entity) {
                let current = Vec3::from_array(transform.translation);
                let vel = Vec3::from_array(velocity.linear);
                transform.translation = (current + vel * dt_seconds).to_array();
                world.insert(entity, transform);
            }
        }
    }
}
