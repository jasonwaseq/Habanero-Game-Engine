//! Physics integration layer.
//!
//! Provides a small, dependency-light motion model used by the runtime demo to
//! exercise the ECS write path every frame: linear velocity integration,
//! local-space spin, and orbital motion about the world origin. All motion is
//! purely `dt`-driven so the simulation is deterministic regardless of frame
//! pacing.

use engine_ecs::{Entity, Transform, World};
use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

/// Linear velocity in world units per second.
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

/// Local-space angular velocity (radians/second) about a normalized axis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Spin {
    pub axis: [f32; 3],
    pub speed: f32,
}

impl Default for Spin {
    fn default() -> Self {
        Self {
            axis: [0.0, 1.0, 0.0],
            speed: 1.0,
        }
    }
}

/// Orbital motion about the world Y axis at the entity's current radius.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Orbit {
    /// Angular velocity in radians/second.
    pub speed: f32,
}

impl Default for Orbit {
    fn default() -> Self {
        Self { speed: 0.5 }
    }
}

pub struct PhysicsSystem;

impl PhysicsSystem {
    /// Advance all motion components by `dt_seconds`.
    ///
    /// Writes back only the transforms that actually have a motion component,
    /// keeping the ECS change set tight for downstream change-detection systems.
    pub fn step(world: &World, dt_seconds: f32) {
        if dt_seconds <= 0.0 {
            return;
        }
        // Snapshot the (typically sparse) motion components once, then apply them
        // to transforms under a single write lock via `for_each_mut`. This keeps
        // the hot path to one lock acquisition instead of one per entity.
        let velocities: std::collections::HashMap<Entity, Velocity> =
            world.query::<Velocity>().into_iter().collect();
        let spins: std::collections::HashMap<Entity, Spin> =
            world.query::<Spin>().into_iter().collect();
        let orbits: std::collections::HashMap<Entity, Orbit> =
            world.query::<Orbit>().into_iter().collect();

        if velocities.is_empty() && spins.is_empty() && orbits.is_empty() {
            return;
        }

        world.for_each_mut::<Transform, _>(|entity, transform| {
            if let Some(velocity) = velocities.get(&entity) {
                let current = Vec3::from_array(transform.translation);
                let vel = Vec3::from_array(velocity.linear);
                transform.translation = (current + vel * dt_seconds).to_array();
            }

            if let Some(orbit) = orbits.get(&entity) {
                let angle = orbit.speed * dt_seconds;
                let pos = Vec3::from_array(transform.translation);
                let rotated = Quat::from_rotation_y(angle) * pos;
                transform.translation = rotated.to_array();
            }

            if let Some(spin) = spins.get(&entity) {
                let axis = Vec3::from_array(spin.axis).normalize_or_zero();
                if axis.length_squared() > 0.0 {
                    let delta = Quat::from_axis_angle(axis, spin.speed * dt_seconds);
                    let current = Quat::from_array(transform.rotation).normalize();
                    let next = (delta * current).normalize();
                    transform.rotation = next.to_array();
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn velocity_integrates_position() {
        let world = World::new();
        let e = world.spawn();
        world.insert(e, Transform::default());
        world.insert(
            e,
            Velocity {
                linear: [1.0, 0.0, 0.0],
            },
        );
        PhysicsSystem::step(&world, 0.5);
        let t = world.get::<Transform>(e).expect("transform");
        assert!((t.translation[0] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn spin_changes_orientation() {
        let world = World::new();
        let e = world.spawn();
        world.insert(e, Transform::default());
        world.insert(
            e,
            Spin {
                axis: [0.0, 1.0, 0.0],
                speed: std::f32::consts::PI,
            },
        );
        PhysicsSystem::step(&world, 1.0);
        let t = world.get::<Transform>(e).expect("transform");
        // 180 degrees about Y -> quaternion y component ~1.0
        assert!(t.rotation[1].abs() > 0.9);
    }

    #[test]
    fn orbit_preserves_radius() {
        let world = World::new();
        let e = world.spawn();
        world.insert(
            e,
            Transform {
                translation: [2.0, 0.0, 0.0],
                ..Default::default()
            },
        );
        world.insert(e, Orbit { speed: 1.0 });
        PhysicsSystem::step(&world, 0.25);
        let t = world.get::<Transform>(e).expect("transform");
        let radius = (t.translation[0].powi(2) + t.translation[2].powi(2)).sqrt();
        assert!((radius - 2.0).abs() < 1e-4);
        // It actually moved off the axis.
        assert!(t.translation[2].abs() > 1e-3);
    }

    #[test]
    fn zero_dt_is_noop() {
        let world = World::new();
        let e = world.spawn();
        world.insert(
            e,
            Transform {
                translation: [1.0, 2.0, 3.0],
                ..Default::default()
            },
        );
        world.insert(e, Velocity { linear: [9.0, 9.0, 9.0] });
        PhysicsSystem::step(&world, 0.0);
        let t = world.get::<Transform>(e).expect("transform");
        assert_eq!(t.translation, [1.0, 2.0, 3.0]);
    }
}
