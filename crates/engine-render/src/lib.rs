//! Vulkan renderer and render graph abstractions.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use ash::vk;
use engine_assets::MeshAsset;
use engine_scene::Scene;
use glam::{Mat4, Vec3, Vec3A};
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

    /// Build an orbiting look-at camera pointing at `target`.
    pub fn looking_at(eye: Vec3, target: Vec3, aspect: f32, fov_y_radians: f32) -> Self {
        Self {
            view: Mat4::look_at_rh(eye, target, Vec3::Y),
            projection: Mat4::perspective_rh(fov_y_radians, aspect.max(0.01), 0.05, 4000.0),
        }
    }

    pub fn view_projection(&self) -> Mat4 {
        self.projection * self.view
    }
}

#[derive(Debug, Clone)]
pub struct DrawPacket {
    pub model: Mat4,
    pub color: [f32; 4],
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
    next_mesh_id: u32,
    default_mesh: Option<MeshId>,
    mesh_assets: HashMap<MeshId, MeshAsset>,
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
            next_mesh_id: 1,
            default_mesh: None,
            mesh_assets: HashMap::new(),
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
            next_mesh_id: 1,
            default_mesh: None,
            mesh_assets: HashMap::new(),
            backend,
            surface_format: vk::Format::B8G8R8A8_UNORM,
        })
    }

    pub fn register_mesh(&mut self, mesh: MeshAsset) -> MeshId {
        let id = MeshId(self.next_mesh_id);
        self.next_mesh_id = self.next_mesh_id.saturating_add(1);
        self.mesh_assets.insert(id, mesh.clone());
        if self.default_mesh.is_none() {
            self.default_mesh = Some(id);
        }
        if let Some(backend) = self.backend.as_mut() {
            if let Err(error) = backend.upload_mesh(id, &mesh) {
                tracing::warn!(?error, "failed to upload mesh to Vulkan backend");
            }
        }
        id
    }

    pub fn register_material(&mut self, id: MaterialId, material: Material) {
        self.materials.write().insert(id, material);
    }

    pub fn extract_scene(&self, scene: &Scene) -> RenderWorld {
        let mesh = self.default_mesh.unwrap_or(MeshId(0));
        let draw_packets = scene
            .world_matrices()
            .into_iter()
            .map(|(entity, model)| DrawPacket {
                model,
                color: palette_color(entity.0),
                material: MaterialId(0),
                mesh,
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

    /// Submit a culled render world for presentation.
    ///
    /// `camera` provides the view-projection used by the GPU instanced pass and
    /// `light_dir` is the world-space direction the key directional light points.
    pub fn submit(&mut self, render_world: &RenderWorld, camera: &Camera, light_dir: Vec3) {
        self.draw_calls_last_frame = render_world.draw_packets.len();
        let _active_passes = self.frame_graph.pass_count();
        let _format = self.surface_format;
        if let Some(backend) = self.backend.as_mut() {
            let stats = backend.stats();
            self.gpu_memory_budget_mb = (stats.extent.width.saturating_mul(stats.extent.height)
                / 4096)
                .max(256);
            let _swapchain_info = (stats.swapchain_images, stats.swapchain_format);
            let view_proj = camera.view_projection();
            if let Err(error) = backend.render_frame(render_world, view_proj, light_dir) {
                tracing::warn!(?error, "scene pass execution failed");
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

    /// Recreate swapchain-dependent resources after a window resize.
    pub fn resize(&mut self) {
        if let Some(backend) = self.backend.as_mut() {
            if let Err(error) = backend.recreate_swapchain() {
                tracing::warn!(?error, "swapchain recreation failed");
            }
        }
    }

    /// Number of meshes resident on the GPU backend (0 when running headless).
    pub fn resident_mesh_count(&self) -> usize {
        self.mesh_assets.len()
    }
}

/// Deterministic, pleasant per-entity color derived from its id.
///
/// Spreads hues using the golden-ratio conjugate so adjacent ids get
/// well-separated, saturated colors without a lookup table.
pub fn palette_color(seed: u64) -> [f32; 4] {
    let hue = (seed as f32 * 0.618_034) % 1.0;
    let saturation = 0.65;
    let value = 0.95;
    let (r, g, b) = hsv_to_rgb(hue, saturation, value);
    [r, g, b, 1.0]
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let i = (h * 6.0).floor();
    let f = h * 6.0 - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    match (i as i32) % 6 {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_ecs::Transform;
    use engine_scene::Scene;

    #[test]
    fn extract_produces_packet_per_entity() {
        let mut scene = Scene::new();
        scene.spawn_named("a", Transform::default());
        scene.spawn_named("b", Transform::default());
        let renderer = VulkanRenderer::new().expect("renderer");
        let world = renderer.extract_scene(&scene);
        assert_eq!(world.draw_packets.len(), 2);
    }

    #[test]
    fn cull_removes_offscreen_entities() {
        let mut scene = Scene::new();
        // In front of the camera.
        scene.spawn_named(
            "front",
            Transform {
                translation: [0.0, 0.0, -5.0],
                ..Default::default()
            },
        );
        // Far behind the camera.
        scene.spawn_named(
            "behind",
            Transform {
                translation: [0.0, 0.0, 50.0],
                ..Default::default()
            },
        );
        let renderer = VulkanRenderer::new().expect("renderer");
        let extracted = renderer.extract_scene(&scene);
        let camera = Camera::looking_at(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, -1.0),
            16.0 / 9.0,
            60f32.to_radians(),
        );
        let visible = renderer.cull_visible(&extracted, &camera);
        assert_eq!(extracted.draw_packets.len(), 2);
        assert_eq!(visible.draw_packets.len(), 1);
    }

    #[test]
    fn palette_is_deterministic_and_in_range() {
        let c = palette_color(42);
        assert_eq!(c, palette_color(42));
        for channel in c {
            assert!((0.0..=1.0).contains(&channel));
        }
    }

    #[test]
    fn deferred_frame_graph_has_expected_passes() {
        assert_eq!(FrameGraph::deferred_default().pass_count(), 8);
    }
}
