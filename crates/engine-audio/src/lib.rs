//! Audio abstraction layer.

use anyhow::Result;
use glam::Vec3;
use kira::manager::{AudioManager, AudioManagerSettings};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialVoice {
    pub emitter: [f32; 3],
}

pub struct AudioSystem {
    _manager: AudioManager,
    listener_position: Vec3,
}

impl AudioSystem {
    pub fn new() -> Result<Self> {
        let manager = AudioManager::new(AudioManagerSettings::default())?;
        Ok(Self {
            _manager: manager,
            listener_position: Vec3::ZERO,
        })
    }

    pub fn set_listener_position(&mut self, position: [f32; 3]) {
        self.listener_position = Vec3::from_array(position);
    }
}
