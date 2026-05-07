//! Vulkan renderer and render graph abstractions.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use ash::vk;
use engine_scene::Scene;
use glam::{Mat4, Vec3A};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use winit::window::Window;

mod vulkan;
use vulkan::VulkanBackend;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Camera {
    pub view: Mat4,
    pub projection: Mat4,
}

impl Camera {
    pub fn perspective(aspect: f32, fov_y_radians: f32, near: f32, far: f32) -> Self {
        Self {
            view: Mat4::IDENTITY,
            projection: Mat4::perspective_rh(fov_y_radians, aspect, near, far),
        }
    }

    pub fn orthographic(width: f32, height: f32, near: f32, far: f32) -> Self {
        Self {
            view: Mat4::IDENTITY,
            projection: Mat4::orthographic_rh(
                -width * 0.5,
                width * 0.5,
                -height * 0.5,
                height * 0.5,
                near,
                far,
            ),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DrawPacket {
    pub model: Mat4,
    pub material: MaterialId,
    pub mesh: MeshId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MaterialId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MeshId(pub u32);

#[derive(Debug, Clone)]
pub struct Material {
    pub albedo: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
}

#[derive(Default)]
pub struct RenderWorld {
    pub draw_packets: Vec<DrawPacket>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RenderStats {
    pub extracted_draws: usize,
    pub visible_draws: usize,
}

#[derive(Default)]
pub struct FrameGraph {
    passes: Vec<RenderPassNode>,
}

#[derive(Debug, Clone)]
pub struct RenderPassNode {
    pub name: &'static str,
    pub kind: PassKind,
}

#[derive(Debug, Clone, Copy)]
pub enum PassKind {
    DepthPrepass,
    GBuffer,
    Lighting,
    ShadowMap,
    Ssao,
    Bloom,
    ForwardTransparent,
    PostProcess,
}

impl FrameGraph {
    pub fn deferred_default() -> Self {
        Self {
            passes: vec![
                RenderPassNode {
                    name: "depth_prepass",
                    kind: PassKind::DepthPrepass,
                },
                RenderPassNode {
                    name: "gbuffer",
                    kind: PassKind::GBuffer,
                },
                RenderPassNode {
                    name: "lighting",
                    kind: PassKind::Lighting,
                },
                RenderPassNode {
                    name: "shadow_map",
                    kind: PassKind::ShadowMap,
                },
                RenderPassNode {
                    name: "ssao",
                    kind: PassKind::Ssao,
                },
                RenderPassNode {
                    name: "bloom",
                    kind: PassKind::Bloom,
                },
                RenderPassNode {
                    name: "forward_transparent",
                    kind: PassKind::ForwardTransparent,
                },
                RenderPassNode {
                    name: "post_process",
                    kind: PassKind::PostProcess,
                },
            ],
        }
    }

    pub fn pass_count(&self) -> usize {
        self.passes.len()
    }
}

pub struct VulkanRenderer {
    pub materials: Arc<RwLock<HashMap<MaterialId, Material>>>,
    pub frame_graph: FrameGraph,
    pub draw_calls_last_frame: usize,
    pub frame_time_ms: f32,
    pub gpu_memory_budget_mb: u32,
    pub stats: RenderStats,
    backend: Option<VulkanBackend>,
    surface_format: vk::Format,
}

impl VulkanRenderer {
    pub fn new() -> Result<Self> {
        Ok(Self {
            materials: Arc::new(RwLock::new(HashMap::new())),
            frame_graph: FrameGraph::deferred_default(),
            draw_calls_last_frame: 0,
            frame_time_ms: 0.0,
            gpu_memory_budget_mb: 512,
            stats: RenderStats::default(),
            backend: None,
            surface_format: vk::Format::B8G8R8A8_UNORM,
        })
    }

    pub fn new_with_window(window: &Window) -> Result<Self> {
        let backend = match VulkanBackend::new(window) {
            Ok(backend) => Some(backend),
            Err(error) => {
                tracing::warn!(?error, "Vulkan backend initialization failed; using stub backend");
                None
            }
        };
        Ok(Self {
            materials: Arc::new(RwLock::new(HashMap::new())),
            frame_graph: FrameGraph::deferred_default(),
            draw_calls_last_frame: 0,
            frame_time_ms: 0.0,
            gpu_memory_budget_mb: 512,
            stats: RenderStats::default(),
            backend,
            surface_format: vk::Format::B8G8R8A8_UNORM,
        })
    }

    pub fn register_material(&mut self, id: MaterialId, material: Material) {
        self.materials.write().insert(id, material);
    }

    pub fn extract_scene(&self, scene: &Scene) -> RenderWorld {
        let draw_packets = scene
            .world_matrices()
            .into_iter()
            .map(|(_entity, model)| DrawPacket {
                model,
                material: MaterialId(0),
                mesh: MeshId(0),
            })
            .collect();
        RenderWorld { draw_packets }
    }

    pub fn cull_visible(&self, render_world: &RenderWorld, camera: &Camera) -> RenderWorld {
        let view_proj = camera.projection * camera.view;
        let mut visible = Vec::with_capacity(render_world.draw_packets.len());
        for packet in &render_world.draw_packets {
            let world_pos = packet.model.transform_point3a(Vec3A::ZERO);
            let clip = view_proj * world_pos.extend(1.0);
            if clip.w <= 0.0001 {
                continue;
            }
            let ndc = clip.truncate() / clip.w;
            let in_frustum = ndc.x.abs() <= 1.15 && ndc.y.abs() <= 1.15 && ndc.z >= -0.05 && ndc.z <= 1.05;
            if in_frustum {
                visible.push(packet.clone());
            }
        }
        RenderWorld { draw_packets: visible }
    }

    pub fn submit(&mut self, render_world: &RenderWorld) {
        self.draw_calls_last_frame = render_world.draw_packets.len();
        let _active_passes = self.frame_graph.pass_count();
        let _format = self.surface_format;
        if let Some(backend) = self.backend.as_ref() {
            let stats = backend.stats();
            self.gpu_memory_budget_mb = (stats.extent.width.saturating_mul(stats.extent.height) / 4096)
                .max(256);
            let _swapchain_info = (stats.swapchain_images, stats.swapchain_format);
            if let Err(error) = backend.render_gbuffer_frame() {
                tracing::warn!(?error, "gbuffer pass execution failed");
            }
        }
        self.frame_time_ms = (self.draw_calls_last_frame as f32 * 0.01).max(0.1);
    }

    pub fn update_stats(&mut self, extracted_draws: usize, visible_draws: usize) {
        self.stats = RenderStats {
            extracted_draws,
            visible_draws,
        };
    }

    pub fn is_backend_active(&self) -> bool {
        self.backend.is_some()
    }
}
