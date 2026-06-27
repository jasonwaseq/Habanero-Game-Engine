use std::ffi::CString;
use std::mem::size_of;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{anyhow, Context, Result};
use ash::{vk, Entry};
use engine_assets::{MeshAsset, MeshVertex};
use glam::Mat4;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::window::Window;
use crate::{DrawPacket, RenderWorld};

/// Per-instance data uploaded to the GPU each frame for instanced drawing.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct InstanceData {
    /// Column-major model matrix.
    model: [f32; 16],
    color: [f32; 4],
}

/// Push constant block shared by the scene pipeline's vertex + fragment stages.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct ScenePush {
    view_proj: [f32; 16],
    light_dir: [f32; 4],
}

const MAX_INSTANCES: usize = 200_000;

/// Swapchain plus its derived image views and chosen format/extent.
type SwapchainBundle = (
    vk::SwapchainKHR,
    Vec<vk::Image>,
    Vec<vk::ImageView>,
    vk::Format,
    vk::Extent2D,
);

pub struct DescriptorAllocator {
    device: Arc<ash::Device>,
    pool: vk::DescriptorPool,
}

impl DescriptorAllocator {
    pub fn new(device: Arc<ash::Device>) -> Result<Self> {
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 512,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 1024,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 512,
            },
        ];
        let create_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(2048)
            .pool_sizes(&pool_sizes)
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);
        let pool = unsafe { device.create_descriptor_pool(&create_info, None) }?;
        Ok(Self { device, pool })
    }

    pub fn allocate(&self, layout: vk::DescriptorSetLayout) -> Result<vk::DescriptorSet> {
        let layouts = [layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.pool)
            .set_layouts(&layouts);
        let sets = unsafe { self.device.allocate_descriptor_sets(&alloc_info) }?;
        sets.first()
            .copied()
            .ok_or_else(|| anyhow!("descriptor allocator returned no sets"))
    }

    pub fn free(&self, set: vk::DescriptorSet) -> Result<()> {
        unsafe {
            self.device.free_descriptor_sets(self.pool, &[set])?;
        }
        Ok(())
    }
}

impl Drop for DescriptorAllocator {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_descriptor_pool(self.pool, None);
        }
    }
}

struct ImageResource {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
}

struct GBuffer {
    albedo: ImageResource,
    normals: ImageResource,
    material: ImageResource,
    depth: ImageResource,
}

struct FullscreenPipeline {
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

struct ScenePipeline {
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

struct BufferResource {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
}

/// A host-visible buffer that stays mapped for the lifetime of the backend.
struct MappedBuffer {
    resource: BufferResource,
    ptr: *mut u8,
    /// Total mapped size in bytes; used to bound per-frame instance writes.
    size: usize,
}

struct GpuMesh {
    vertex: BufferResource,
    index: BufferResource,
    #[allow(dead_code)]
    vertex_count: u32,
    index_count: u32,
}

pub struct VulkanBackend {
    // Keep the loader library (`vulkan-1.dll`) alive for the backend's lifetime.
    // Dropping the `Entry` unloads it, which invalidates loader-dispatched
    // instance-level functions such as the WSI surface queries.
    _entry: Entry,
    instance: ash::Instance,
    debug_utils_instance: Option<ash::ext::debug_utils::Instance>,
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
    device: Arc<ash::Device>,
    debug_utils_device: Option<ash::ext::debug_utils::Device>,
    surface_loader: ash::khr::surface::Instance,
    swapchain_loader: ash::khr::swapchain::Device,
    surface: vk::SurfaceKHR,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_views: Vec<vk::ImageView>,
    swapchain_format: vk::Format,
    extent: vk::Extent2D,
    physical_device: vk::PhysicalDevice,
    graphics_queue: vk::Queue,
    graphics_queue_family: u32,
    // `Option` so we can drop it (destroying its pool) while the device is still
    // alive, before `Drop` tears down the device.
    descriptor_allocator: Option<DescriptorAllocator>,
    gbuffer: GBuffer,
    gbuffer_render_pass: vk::RenderPass,
    fullscreen_pipeline: FullscreenPipeline,
    scene_pipeline: ScenePipeline,
    cube: GpuMesh,
    instance_buffer: MappedBuffer,
    framebuffers: Vec<vk::Framebuffer>,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    image_available: vk::Semaphore,
    render_finished: vk::Semaphore,
    in_flight: vk::Fence,
    frame_counter: AtomicU64,
    debug_labels_enabled: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct BackendStats {
    pub swapchain_images: usize,
    pub swapchain_format: vk::Format,
    pub extent: vk::Extent2D,
}

impl VulkanBackend {
    pub fn new(window: &Window) -> Result<Self> {
        let entry = unsafe { Entry::load() }.context("failed to load Vulkan loader")?;
        let app_name = CString::new("HabaneroEngine").expect("valid app name");
        let engine_name = CString::new("Habanero").expect("valid engine name");
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(vk::make_api_version(0, 0, 2, 0))
            .engine_name(&engine_name)
            .engine_version(vk::make_api_version(0, 0, 2, 0))
            .api_version(vk::API_VERSION_1_3);

        let display_handle = window
            .display_handle()
            .map_err(|e| anyhow!("display handle error: {e}"))?;
        let mut ext_names = ash_window::enumerate_required_extensions(display_handle.as_raw())?
            .to_vec();

        let instance_extensions =
            unsafe { entry.enumerate_instance_extension_properties(None) }
                .context("failed to enumerate instance extensions")?;
        let has_debug_utils_ext = instance_extensions.iter().any(|ext| {
            let name = unsafe { std::ffi::CStr::from_ptr(ext.extension_name.as_ptr()) };
            name == ash::ext::debug_utils::NAME
        });
        if has_debug_utils_ext {
            ext_names.push(ash::ext::debug_utils::NAME.as_ptr());
        }

        let validation_layer = CString::new("VK_LAYER_KHRONOS_validation").expect("layer CString");
        let available_layers =
            unsafe { entry.enumerate_instance_layer_properties() }
                .context("failed to enumerate instance layers")?;
        let has_validation_layer = available_layers.iter().any(|layer| {
            let name = unsafe { std::ffi::CStr::from_ptr(layer.layer_name.as_ptr()) };
            name.to_bytes() == validation_layer.as_bytes()
        });
        let layer_names = if cfg!(debug_assertions) && has_validation_layer {
            vec![validation_layer.as_ptr()]
        } else {
            Vec::new()
        };
        let instance_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&ext_names)
            .enabled_layer_names(&layer_names);
        let instance = unsafe { entry.create_instance(&instance_info, None) }?;
        let debug_utils_instance = if has_debug_utils_ext {
            Some(ash::ext::debug_utils::Instance::new(&entry, &instance))
        } else {
            None
        };
        let enable_validation_callback = std::env::var("HBN_ENABLE_VALIDATION_CALLBACK")
            .ok()
            .as_deref()
            == Some("1");
        let debug_messenger = if cfg!(debug_assertions) && has_validation_layer && enable_validation_callback {
            if let Some(debug_utils) = debug_utils_instance.as_ref() {
                let messenger_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                    .message_severity(
                        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                            | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
                    )
                    .message_type(
                        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                            | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                            | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                    )
                    .pfn_user_callback(Some(vulkan_debug_callback));
                Some(unsafe { debug_utils.create_debug_utils_messenger(&messenger_info, None) }?)
            } else {
                None
            }
        } else {
            None
        };

        let window_handle = window
            .window_handle()
            .map_err(|e| anyhow!("window handle error: {e}"))?;
        let surface = unsafe {
            ash_window::create_surface(
                &entry,
                &instance,
                display_handle.as_raw(),
                window_handle.as_raw(),
                None,
            )
        }?;
        let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);

        let (physical_device, graphics_queue_family) =
            pick_physical_device(&instance, &surface_loader, surface)?;
        let queue_priorities = [1.0_f32];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(graphics_queue_family)
            .queue_priorities(&queue_priorities)];
        let device_extensions = [ash::khr::swapchain::NAME.as_ptr()];
        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_info)
            .enabled_extension_names(&device_extensions);
        let device = unsafe { instance.create_device(physical_device, &device_info, None) }?;
        let device = Arc::new(device);
        let debug_labels_enabled = std::env::var("HBN_ENABLE_DEBUG_LABELS")
            .ok()
            .as_deref()
            == Some("1");
        let debug_utils_device = if has_debug_utils_ext && debug_labels_enabled {
            Some(ash::ext::debug_utils::Device::new(&instance, &device))
        } else {
            None
        };
        let graphics_queue = unsafe { device.get_device_queue(graphics_queue_family, 0) };

        let swapchain_loader = ash::khr::swapchain::Device::new(&instance, &device);
        let (swapchain, swapchain_images, swapchain_views, swapchain_format, extent) = create_swapchain(
            &device,
            &surface_loader,
            &swapchain_loader,
            physical_device,
            surface,
            graphics_queue_family,
            window.inner_size().width.max(1),
            window.inner_size().height.max(1),
        )?;

        let descriptor_allocator = DescriptorAllocator::new(device.clone())?;
        warmup_descriptor_allocator(&device, &descriptor_allocator)?;
        let gbuffer = create_gbuffer(&instance, &device, physical_device, extent)?;
        let gbuffer_render_pass = create_gbuffer_render_pass(&device, swapchain_format)?;
        let fullscreen_pipeline = create_fullscreen_pipeline(&device, gbuffer_render_pass)?;
        let scene_pipeline = create_scene_pipeline(&device, gbuffer_render_pass)?;

        let cube_asset = unit_cube_mesh();
        let cube = upload_mesh(&instance, &device, physical_device, &cube_asset)?;

        let instance_buffer = create_mapped_buffer(
            &instance,
            &device,
            physical_device,
            MAX_INSTANCES * size_of::<InstanceData>(),
            vk::BufferUsageFlags::VERTEX_BUFFER,
        )?;

        let framebuffers = create_framebuffers(
            &device,
            gbuffer_render_pass,
            &swapchain_views,
            &gbuffer,
            extent,
        )?;

        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(graphics_queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe { device.create_command_pool(&pool_info, None) }?;
        let cmd_alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .command_buffer_count(framebuffers.len() as u32)
            .level(vk::CommandBufferLevel::PRIMARY);
        let command_buffers = unsafe { device.allocate_command_buffers(&cmd_alloc) }?;

        let sem_info = vk::SemaphoreCreateInfo::default();
        let image_available = unsafe { device.create_semaphore(&sem_info, None) }?;
        let render_finished = unsafe { device.create_semaphore(&sem_info, None) }?;
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let in_flight = unsafe { device.create_fence(&fence_info, None) }?;

        Ok(Self {
            _entry: entry,
            instance,
            debug_utils_instance,
            debug_messenger,
            device,
            debug_utils_device,
            surface_loader,
            swapchain_loader,
            surface,
            swapchain,
            swapchain_images,
            swapchain_views,
            swapchain_format,
            extent,
            physical_device,
            graphics_queue,
            graphics_queue_family,
            descriptor_allocator: Some(descriptor_allocator),
            gbuffer,
            gbuffer_render_pass,
            fullscreen_pipeline,
            scene_pipeline,
            cube,
            instance_buffer,
            framebuffers,
            command_pool,
            command_buffers,
            image_available,
            render_finished,
            in_flight,
            frame_counter: AtomicU64::new(0),
            debug_labels_enabled,
        })
    }

    /// External mesh uploads are accepted but the demo renders the resident cube.
    pub fn upload_mesh(&mut self, _mesh_id: crate::MeshId, _mesh: &MeshAsset) -> Result<()> {
        Ok(())
    }

    fn write_instances(&mut self, packets: &[DrawPacket]) -> u32 {
        let capacity = self.instance_buffer.size / size_of::<InstanceData>();
        let count = packets.len().min(capacity);
        // SAFETY: `instance_buffer.ptr` is a persistently-mapped, host-coherent
        // allocation of `instance_buffer.size` bytes; we never write past it.
        unsafe {
            let dst = self.instance_buffer.ptr.cast::<InstanceData>();
            for (i, packet) in packets.iter().take(count).enumerate() {
                let data = InstanceData {
                    model: packet.model.to_cols_array(),
                    color: packet.color,
                };
                std::ptr::write(dst.add(i), data);
            }
        }
        count as u32
    }

    pub fn render_frame(
        &mut self,
        render_world: &RenderWorld,
        view_proj: Mat4,
        light_dir: glam::Vec3,
    ) -> Result<()> {
        if self.extent.width == 0 || self.extent.height == 0 {
            return Ok(());
        }
        // Descriptor pool is warmed at init and reserved for upcoming
        // per-frame uniform/material sets.
        let _ = self.descriptor_allocator.as_ref();
        let instance_count = self.write_instances(&render_world.draw_packets);
        let frame = self.frame_counter.fetch_add(1, Ordering::Relaxed) as f32;
        let t = frame * 0.02;
        let bg_r = (0.02 + 0.015 * (t * 0.5).sin().abs()).clamp(0.0, 1.0);
        let bg_g = (0.02 + 0.02 * (t * 0.37).sin().abs()).clamp(0.0, 1.0);
        let bg_b = (0.05 + 0.05 * (t * 0.23).sin().abs()).clamp(0.0, 1.0);

        unsafe {
            self.device
                .wait_for_fences(&[self.in_flight], true, u64::MAX)?;
        }
        let acquire = unsafe {
            self.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                self.image_available,
                vk::Fence::null(),
            )
        };
        let image_index = match acquire {
            Ok((index, _suboptimal)) => index,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.recreate_swapchain()?;
                return Ok(());
            }
            Err(err) => return Err(err.into()),
        };

        unsafe {
            self.device.reset_fences(&[self.in_flight])?;
        }

        let push = ScenePush {
            view_proj: view_proj.to_cols_array(),
            light_dir: [light_dir.x, light_dir.y, light_dir.z, 0.0],
        };

        let command_buffer = self.command_buffers[image_index as usize];
        unsafe {
            self.device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())?;
            let begin_info = vk::CommandBufferBeginInfo::default();
            self.device.begin_command_buffer(command_buffer, &begin_info)?;
            if self.debug_labels_enabled {
                begin_label(self.debug_utils_device.as_ref(), command_buffer, c"scene_frame");
            }
            let clear_values = [
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0],
                    },
                },
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.5, 0.5, 1.0, 1.0],
                    },
                },
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.5, 0.0, 1.0],
                    },
                },
                vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                },
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [bg_r, bg_g, bg_b, 1.0],
                    },
                },
            ];
            let render_pass_info = vk::RenderPassBeginInfo::default()
                .render_pass(self.gbuffer_render_pass)
                .framebuffer(self.framebuffers[image_index as usize])
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: self.extent,
                })
                .clear_values(&clear_values);
            self.device.cmd_begin_render_pass(
                command_buffer,
                &render_pass_info,
                vk::SubpassContents::INLINE,
            );

            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: self.extent.width as f32,
                height: self.extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            let scissor = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.extent,
            };
            self.device.cmd_set_viewport(command_buffer, 0, &[viewport]);
            self.device.cmd_set_scissor(command_buffer, 0, &[scissor]);

            if instance_count > 0 {
                if self.debug_labels_enabled {
                    begin_label(
                        self.debug_utils_device.as_ref(),
                        command_buffer,
                        c"instanced_scene",
                    );
                }
                self.device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.scene_pipeline.pipeline,
                );
                let vertex_buffers = [self.cube.vertex.buffer, self.instance_buffer.resource.buffer];
                let offsets = [0_u64, 0_u64];
                self.device
                    .cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &offsets);
                self.device.cmd_bind_index_buffer(
                    command_buffer,
                    self.cube.index.buffer,
                    0,
                    vk::IndexType::UINT32,
                );
                let push_bytes = std::slice::from_raw_parts(
                    (&push as *const ScenePush).cast::<u8>(),
                    size_of::<ScenePush>(),
                );
                self.device.cmd_push_constants(
                    command_buffer,
                    self.scene_pipeline.layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    push_bytes,
                );
                self.device.cmd_draw_indexed(
                    command_buffer,
                    self.cube.index_count,
                    instance_count,
                    0,
                    0,
                    0,
                );
                if self.debug_labels_enabled {
                    end_label(self.debug_utils_device.as_ref(), command_buffer);
                }
            } else {
                self.device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.fullscreen_pipeline.pipeline,
                );
                self.device.cmd_draw(command_buffer, 3, 1, 0, 0);
            }

            self.device.cmd_end_render_pass(command_buffer);
            if self.debug_labels_enabled {
                end_label(self.debug_utils_device.as_ref(), command_buffer);
            }
            self.device.end_command_buffer(command_buffer)?;

            let wait_semaphores = [self.image_available];
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let signal_semaphores = [self.render_finished];
            let cmd_bufs = [command_buffer];
            let submit_info = vk::SubmitInfo::default()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&cmd_bufs)
                .signal_semaphores(&signal_semaphores);
            self.device
                .queue_submit(self.graphics_queue, &[submit_info], self.in_flight)?;

            let present_wait = [self.render_finished];
            let present_swaps = [self.swapchain];
            let present_indices = [image_index];
            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(&present_wait)
                .swapchains(&present_swaps)
                .image_indices(&present_indices);
            match self
                .swapchain_loader
                .queue_present(self.graphics_queue, &present_info)
            {
                Ok(_) => {}
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => {
                    self.recreate_swapchain()?;
                }
                Err(err) => return Err(err.into()),
            }
        }
        Ok(())
    }

    /// Recreate the swapchain and all size-dependent attachments after a resize
    /// or an out-of-date surface. Pipelines use dynamic viewport/scissor, so they
    /// survive resizes untouched.
    pub fn recreate_swapchain(&mut self) -> Result<()> {
        let capabilities = unsafe {
            self.surface_loader
                .get_physical_device_surface_capabilities(self.physical_device, self.surface)?
        };
        let new_extent = if capabilities.current_extent.width != u32::MAX {
            capabilities.current_extent
        } else {
            self.extent
        };
        if new_extent.width == 0 || new_extent.height == 0 {
            // Minimized: keep resources, skip rendering until restored.
            self.extent = new_extent;
            return Ok(());
        }

        unsafe {
            self.device.device_wait_idle()?;
            for framebuffer in self.framebuffers.drain(..) {
                self.device.destroy_framebuffer(framebuffer, None);
            }
            destroy_image_resource(&self.device, &self.gbuffer.albedo);
            destroy_image_resource(&self.device, &self.gbuffer.normals);
            destroy_image_resource(&self.device, &self.gbuffer.material);
            destroy_image_resource(&self.device, &self.gbuffer.depth);
            for view in self.swapchain_views.drain(..) {
                self.device.destroy_image_view(view, None);
            }
            self.swapchain_loader.destroy_swapchain(self.swapchain, None);
        }

        let (swapchain, images, views, format, extent) = create_swapchain(
            &self.device,
            &self.surface_loader,
            &self.swapchain_loader,
            self.physical_device,
            self.surface,
            self.graphics_queue_family,
            new_extent.width,
            new_extent.height,
        )?;
        self.swapchain = swapchain;
        self.swapchain_images = images;
        self.swapchain_views = views;
        self.swapchain_format = format;
        self.extent = extent;
        self.gbuffer = create_gbuffer(&self.instance, &self.device, self.physical_device, extent)?;
        self.framebuffers = create_framebuffers(
            &self.device,
            self.gbuffer_render_pass,
            &self.swapchain_views,
            &self.gbuffer,
            extent,
        )?;
        // Keep one command buffer per swapchain image if the count changed.
        if self.framebuffers.len() != self.command_buffers.len() {
            unsafe {
                self.device
                    .free_command_buffers(self.command_pool, &self.command_buffers);
            }
            let cmd_alloc = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.command_pool)
                .command_buffer_count(self.framebuffers.len() as u32)
                .level(vk::CommandBufferLevel::PRIMARY);
            self.command_buffers = unsafe { self.device.allocate_command_buffers(&cmd_alloc) }?;
        }
        Ok(())
    }

    pub fn stats(&self) -> BackendStats {
        BackendStats {
            swapchain_images: self.swapchain_images.len(),
            swapchain_format: self.swapchain_format,
            extent: self.extent,
        }
    }
}

impl Drop for VulkanBackend {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            // Destroy the descriptor pool (via its own Drop) before we tear down
            // the device it references.
            self.descriptor_allocator.take();
            self.device.destroy_fence(self.in_flight, None);
            self.device.destroy_semaphore(self.render_finished, None);
            self.device.destroy_semaphore(self.image_available, None);
            self.device.destroy_command_pool(self.command_pool, None);
            for framebuffer in &self.framebuffers {
                self.device.destroy_framebuffer(*framebuffer, None);
            }
            self.device.destroy_render_pass(self.gbuffer_render_pass, None);
            self.device
                .destroy_pipeline(self.fullscreen_pipeline.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.fullscreen_pipeline.layout, None);
            self.device.destroy_pipeline(self.scene_pipeline.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.scene_pipeline.layout, None);

            destroy_buffer_resource(&self.device, &self.cube.vertex);
            destroy_buffer_resource(&self.device, &self.cube.index);
            self.device
                .unmap_memory(self.instance_buffer.resource.memory);
            destroy_buffer_resource(&self.device, &self.instance_buffer.resource);

            destroy_image_resource(&self.device, &self.gbuffer.albedo);
            destroy_image_resource(&self.device, &self.gbuffer.normals);
            destroy_image_resource(&self.device, &self.gbuffer.material);
            destroy_image_resource(&self.device, &self.gbuffer.depth);

            for view in &self.swapchain_views {
                self.device.destroy_image_view(*view, None);
            }
            self.swapchain_loader.destroy_swapchain(self.swapchain, None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.device.destroy_device(None);
            if let (Some(debug_utils), Some(messenger)) =
                (self.debug_utils_instance.as_ref(), self.debug_messenger)
            {
                debug_utils.destroy_debug_utils_messenger(messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

fn pick_physical_device(
    instance: &ash::Instance,
    surface_loader: &ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
) -> Result<(vk::PhysicalDevice, u32)> {
    let devices = unsafe { instance.enumerate_physical_devices() }?;
    // Pick the first device/queue family that supports both graphics and present
    // for this surface. On hybrid-graphics systems this is the GPU the surface
    // was created against, which avoids cross-ICD surface query faults.
    for physical_device in devices {
        let queue_props =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        for (index, family) in queue_props.iter().enumerate() {
            let supports_graphics = family.queue_flags.contains(vk::QueueFlags::GRAPHICS);
            let supports_present = unsafe {
                surface_loader.get_physical_device_surface_support(
                    physical_device,
                    index as u32,
                    surface,
                )?
            };
            if supports_graphics && supports_present {
                return Ok((physical_device, index as u32));
            }
        }
    }
    Err(anyhow!("no suitable Vulkan physical device with present support"))
}

#[allow(clippy::too_many_arguments)]
fn create_swapchain(
    device: &ash::Device,
    surface_loader: &ash::khr::surface::Instance,
    swapchain_loader: &ash::khr::swapchain::Device,
    physical_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    queue_family: u32,
    width: u32,
    height: u32,
) -> Result<SwapchainBundle> {
    let capabilities =
        unsafe { surface_loader.get_physical_device_surface_capabilities(physical_device, surface) }?;
    let formats =
        unsafe { surface_loader.get_physical_device_surface_formats(physical_device, surface) }?;
    let present_modes = unsafe {
        surface_loader.get_physical_device_surface_present_modes(physical_device, surface)
    }?;

    let chosen_format = formats
        .iter()
        .find(|format| {
            format.format == vk::Format::B8G8R8A8_UNORM
                && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .copied()
        .unwrap_or_else(|| formats[0]);

    let present_mode = present_modes
        .iter()
        .copied()
        .find(|mode| *mode == vk::PresentModeKHR::MAILBOX)
        .unwrap_or(vk::PresentModeKHR::FIFO);

    let mut image_count = capabilities.min_image_count.saturating_add(1);
    if capabilities.max_image_count > 0 {
        image_count = image_count.min(capabilities.max_image_count);
    }

    let extent = if capabilities.current_extent.width != u32::MAX {
        capabilities.current_extent
    } else {
        vk::Extent2D {
            width: width.clamp(
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,
            ),
            height: height.clamp(
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
            ),
        }
    };

    let queue_families = [queue_family];
    let create_info = vk::SwapchainCreateInfoKHR::default()
        .surface(surface)
        .min_image_count(image_count)
        .image_format(chosen_format.format)
        .image_color_space(chosen_format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .queue_family_indices(&queue_families)
        .pre_transform(capabilities.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_mode)
        .clipped(true);
    let swapchain = unsafe { swapchain_loader.create_swapchain(&create_info, None) }?;
    let images = unsafe { swapchain_loader.get_swapchain_images(swapchain) }?;

    let mut views = Vec::with_capacity(images.len());
    for image in &images {
        let subresource = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);
        let view_info = vk::ImageViewCreateInfo::default()
            .image(*image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(chosen_format.format)
            .subresource_range(subresource);
        let view = unsafe { device.create_image_view(&view_info, None) }?;
        views.push(view);
    }

    Ok((swapchain, images, views, chosen_format.format, extent))
}

fn create_gbuffer(
    instance: &ash::Instance,
    device: &ash::Device,
    physical_device: vk::PhysicalDevice,
    extent: vk::Extent2D,
) -> Result<GBuffer> {
    Ok(GBuffer {
        albedo: create_image_resource(
            instance,
            device,
            physical_device,
            extent,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::INPUT_ATTACHMENT,
            vk::ImageAspectFlags::COLOR,
        )?,
        normals: create_image_resource(
            instance,
            device,
            physical_device,
            extent,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::INPUT_ATTACHMENT,
            vk::ImageAspectFlags::COLOR,
        )?,
        material: create_image_resource(
            instance,
            device,
            physical_device,
            extent,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::INPUT_ATTACHMENT,
            vk::ImageAspectFlags::COLOR,
        )?,
        depth: create_image_resource(
            instance,
            device,
            physical_device,
            extent,
            vk::Format::D32_SFLOAT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::ImageAspectFlags::DEPTH,
        )?,
    })
}

fn create_image_resource(
    instance: &ash::Instance,
    device: &ash::Device,
    physical_device: vk::PhysicalDevice,
    extent: vk::Extent2D,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    aspect: vk::ImageAspectFlags,
) -> Result<ImageResource> {
    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D {
            width: extent.width,
            height: extent.height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let image = unsafe { device.create_image(&image_info, None) }?;
    let mem_requirements = unsafe { device.get_image_memory_requirements(image) };
    let memory_type = find_memory_type(
        instance,
        physical_device,
        mem_requirements.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )
    .ok_or_else(|| anyhow!("unable to find device-local memory type"))?;
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(mem_requirements.size)
        .memory_type_index(memory_type);
    let memory = unsafe { device.allocate_memory(&alloc_info, None) }?;
    unsafe {
        device.bind_image_memory(image, memory, 0)?;
    }
    let subresource = vk::ImageSubresourceRange::default()
        .aspect_mask(aspect)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1);
    let view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(subresource);
    let view = unsafe { device.create_image_view(&view_info, None) }?;
    Ok(ImageResource {
        image,
        memory,
        view,
    })
}

fn create_gbuffer_render_pass(device: &ash::Device, swapchain_format: vk::Format) -> Result<vk::RenderPass> {
    let attachments = [
        vk::AttachmentDescription::default()
            .format(vk::Format::R16G16B16A16_SFLOAT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
        vk::AttachmentDescription::default()
            .format(vk::Format::R16G16B16A16_SFLOAT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
        vk::AttachmentDescription::default()
            .format(vk::Format::R8G8B8A8_UNORM)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
        vk::AttachmentDescription::default()
            .format(vk::Format::D32_SFLOAT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
        vk::AttachmentDescription::default()
            .format(swapchain_format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR),
    ];

    let color_refs = [
        vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        },
        vk::AttachmentReference {
            attachment: 1,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        },
        vk::AttachmentReference {
            attachment: 2,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        },
        vk::AttachmentReference {
            attachment: 4,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        },
    ];
    let depth_ref = vk::AttachmentReference {
        attachment: 3,
        layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    };
    let subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_refs)
        .depth_stencil_attachment(&depth_ref);
    let dependency = vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        )
        .dst_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        )
        .dst_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        );
    let subpasses = [subpass];
    let dependencies = [dependency];
    let pass_info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(&subpasses)
        .dependencies(&dependencies);
    let render_pass = unsafe { device.create_render_pass(&pass_info, None) }?;
    Ok(render_pass)
}

fn create_framebuffers(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    swapchain_views: &[vk::ImageView],
    gbuffer: &GBuffer,
    extent: vk::Extent2D,
) -> Result<Vec<vk::Framebuffer>> {
    let mut out = Vec::with_capacity(swapchain_views.len());
    for view in swapchain_views {
        let attachments = [
            gbuffer.albedo.view,
            gbuffer.normals.view,
            gbuffer.material.view,
            gbuffer.depth.view,
            *view,
        ];
        let info = vk::FramebufferCreateInfo::default()
            .render_pass(render_pass)
            .attachments(&attachments)
            .width(extent.width)
            .height(extent.height)
            .layers(1);
        out.push(unsafe { device.create_framebuffer(&info, None) }?);
    }
    Ok(out)
}

fn find_memory_type(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    required_bits: u32,
    properties: vk::MemoryPropertyFlags,
) -> Option<u32> {
    let memory_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
    for index in 0..memory_props.memory_type_count {
        let mem_type = memory_props.memory_types[index as usize];
        let supported = required_bits & (1 << index) != 0;
        if supported && mem_type.property_flags.contains(properties) {
            return Some(index);
        }
    }
    None
}

fn destroy_image_resource(device: &ash::Device, resource: &ImageResource) {
    unsafe {
        device.destroy_image_view(resource.view, None);
        device.destroy_image(resource.image, None);
        device.free_memory(resource.memory, None);
    }
}

fn destroy_buffer_resource(device: &ash::Device, resource: &BufferResource) {
    unsafe {
        device.destroy_buffer(resource.buffer, None);
        device.free_memory(resource.memory, None);
    }
}

fn create_buffer_with_data(
    instance: &ash::Instance,
    device: &ash::Device,
    physical_device: vk::PhysicalDevice,
    data: &[u8],
    usage: vk::BufferUsageFlags,
) -> Result<BufferResource> {
    let buffer_info = vk::BufferCreateInfo::default()
        .size(data.len().max(1) as u64)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buffer = unsafe { device.create_buffer(&buffer_info, None) }?;
    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
    let mem_type = find_memory_type(
        instance,
        physical_device,
        requirements.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )
    .ok_or_else(|| anyhow!("unable to find host visible memory for buffer"))?;
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(mem_type);
    let memory = unsafe { device.allocate_memory(&alloc_info, None) }?;
    unsafe {
        device.bind_buffer_memory(buffer, memory, 0)?;
        let mapped = device.map_memory(memory, 0, data.len() as u64, vk::MemoryMapFlags::empty())?;
        std::ptr::copy_nonoverlapping(data.as_ptr(), mapped.cast::<u8>(), data.len());
        device.unmap_memory(memory);
    }
    Ok(BufferResource { buffer, memory })
}

fn create_mapped_buffer(
    instance: &ash::Instance,
    device: &ash::Device,
    physical_device: vk::PhysicalDevice,
    size: usize,
    usage: vk::BufferUsageFlags,
) -> Result<MappedBuffer> {
    let buffer_info = vk::BufferCreateInfo::default()
        .size(size as u64)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buffer = unsafe { device.create_buffer(&buffer_info, None) }?;
    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
    let mem_type = find_memory_type(
        instance,
        physical_device,
        requirements.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )
    .ok_or_else(|| anyhow!("unable to find host visible memory for mapped buffer"))?;
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(mem_type);
    let memory = unsafe { device.allocate_memory(&alloc_info, None) }?;
    let ptr = unsafe {
        device.bind_buffer_memory(buffer, memory, 0)?;
        device.map_memory(memory, 0, size as u64, vk::MemoryMapFlags::empty())?
    };
    Ok(MappedBuffer {
        resource: BufferResource { buffer, memory },
        ptr: ptr.cast::<u8>(),
        size,
    })
}

fn upload_mesh(
    instance: &ash::Instance,
    device: &ash::Device,
    physical_device: vk::PhysicalDevice,
    mesh: &MeshAsset,
) -> Result<GpuMesh> {
    let vertex = create_buffer_with_data(
        instance,
        device,
        physical_device,
        as_u8_slice(&mesh.vertices),
        vk::BufferUsageFlags::VERTEX_BUFFER,
    )?;
    let index = create_buffer_with_data(
        instance,
        device,
        physical_device,
        as_u8_slice(&mesh.indices),
        vk::BufferUsageFlags::INDEX_BUFFER,
    )?;
    Ok(GpuMesh {
        vertex,
        index,
        vertex_count: mesh.vertices.len() as u32,
        index_count: mesh.indices.len() as u32,
    })
}

fn unit_cube_mesh() -> MeshAsset {
    // (normal, v0, v1, v2, v3) per cube face.
    type Face = ([f32; 3], [f32; 3], [f32; 3], [f32; 3], [f32; 3]);
    // 24 vertices (4 per face) with per-face normals so lighting reads cleanly.
    let faces: [Face; 6] = [
        // (normal, v0, v1, v2, v3) wound CCW when viewed from outside
        ([0.0, 0.0, 1.0], [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]),
        ([0.0, 0.0, -1.0], [0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5]),
        ([1.0, 0.0, 0.0], [0.5, -0.5, 0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5]),
        ([-1.0, 0.0, 0.0], [-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, 0.5, -0.5]),
        ([0.0, 1.0, 0.0], [-0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5]),
        ([0.0, -1.0, 0.0], [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [-0.5, -0.5, 0.5]),
    ];
    let uvs = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let mut vertices = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);
    for (face_idx, (normal, v0, v1, v2, v3)) in faces.iter().enumerate() {
        let base = (face_idx * 4) as u32;
        for (corner, pos) in [v0, v1, v2, v3].into_iter().enumerate() {
            vertices.push(MeshVertex {
                position: *pos,
                normal: *normal,
                uv: uvs[corner],
            });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 3, base]);
    }
    MeshAsset {
        name: "unit_cube".to_string(),
        vertices,
        indices,
    }
}

fn warmup_descriptor_allocator(device: &ash::Device, allocator: &DescriptorAllocator) -> Result<()> {
    let binding = vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::VERTEX);
    let bindings = [binding];
    let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    let layout = unsafe { device.create_descriptor_set_layout(&layout_info, None) }?;
    let set = allocator.allocate(layout)?;
    allocator.free(set)?;
    unsafe {
        device.destroy_descriptor_set_layout(layout, None);
    }
    Ok(())
}

fn dynamic_viewport_state() -> [vk::DynamicState; 2] {
    [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]
}

fn create_fullscreen_pipeline(
    device: &ash::Device,
    render_pass: vk::RenderPass,
) -> Result<FullscreenPipeline> {
    let vert_spv = compile_shader(
        r#"
            struct VsOut {
                @builtin(position) pos: vec4<f32>,
                @location(0) uv: vec2<f32>,
            };

            @vertex
            fn main(@builtin(vertex_index) vertex_index: u32) -> VsOut {
                var positions = array<vec2<f32>, 3>(
                    vec2<f32>(-1.0, -1.0),
                    vec2<f32>( 3.0, -1.0),
                    vec2<f32>(-1.0,  3.0)
                );
                var out: VsOut;
                let p = positions[vertex_index];
                out.pos = vec4<f32>(p, 0.0, 1.0);
                out.uv = p * 0.5 + vec2<f32>(0.5, 0.5);
                return out;
            }
        "#,
        ShaderStage::Vertex,
        "fullscreen.vert",
    )?;
    let frag_spv = compile_shader(
        r#"
            struct FsOut {
                @location(0) outAlbedo: vec4<f32>,
                @location(1) outNormal: vec4<f32>,
                @location(2) outMaterial: vec4<f32>,
                @location(3) outPresent: vec4<f32>,
            };

            @fragment
            fn main(@location(0) uv: vec2<f32>) -> FsOut {
                var out: FsOut;
                let g = mix(vec3<f32>(0.02, 0.03, 0.08), vec3<f32>(0.06, 0.02, 0.12), uv.y);
                out.outAlbedo = vec4<f32>(g, 1.0);
                out.outNormal = vec4<f32>(0.5, 0.5, 1.0, 1.0);
                out.outMaterial = vec4<f32>(0.04, 0.7, 0.0, 1.0);
                out.outPresent = vec4<f32>(g, 1.0);
                return out;
            }
        "#,
        ShaderStage::Fragment,
        "fullscreen.frag",
    )?;

    let vert_module = create_shader_module(device, &vert_spv)?;
    let frag_module = create_shader_module(device, &frag_spv)?;
    let entry_name = CString::new("main").expect("valid shader entry");
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_module)
            .name(&entry_name),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_module)
            .name(&entry_name),
    ];

    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let dynamic_states = dynamic_viewport_state();
    let dynamic_state =
        vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(false)
        .depth_write_enable(false)
        .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
        .stencil_test_enable(false);
    let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(false)
        .color_write_mask(
            vk::ColorComponentFlags::R
                | vk::ColorComponentFlags::G
                | vk::ColorComponentFlags::B
                | vk::ColorComponentFlags::A,
        );
    let color_blend_attachments = [
        color_blend_attachment,
        color_blend_attachment,
        color_blend_attachment,
        color_blend_attachment,
    ];
    let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
        .logic_op_enable(false)
        .attachments(&color_blend_attachments);
    let layout_info = vk::PipelineLayoutCreateInfo::default();
    let layout = unsafe { device.create_pipeline_layout(&layout_info, None) }?;

    let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .dynamic_state(&dynamic_state)
        .rasterization_state(&raster)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&color_blend)
        .layout(layout)
        .render_pass(render_pass)
        .subpass(0);
    let pipelines = unsafe {
        device.create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
    }
    .map_err(|(_, err)| anyhow!("failed to create graphics pipeline: {err:?}"))?;
    unsafe {
        device.destroy_shader_module(vert_module, None);
        device.destroy_shader_module(frag_module, None);
    }

    Ok(FullscreenPipeline {
        layout,
        pipeline: pipelines[0],
    })
}

fn create_scene_pipeline(
    device: &ash::Device,
    render_pass: vk::RenderPass,
) -> Result<ScenePipeline> {
    let vert_spv = compile_shader(
        r#"
            struct PushConstants {
                view_proj: mat4x4<f32>,
                light_dir: vec4<f32>,
            };
            var<push_constant> pc: PushConstants;

            struct VsOut {
                @builtin(position) pos: vec4<f32>,
                @location(0) world_normal: vec3<f32>,
                @location(1) color: vec4<f32>,
            };

            @vertex
            fn main(
                @location(0) in_pos: vec3<f32>,
                @location(1) in_normal: vec3<f32>,
                @location(2) in_uv: vec2<f32>,
                @location(3) m0: vec4<f32>,
                @location(4) m1: vec4<f32>,
                @location(5) m2: vec4<f32>,
                @location(6) m3: vec4<f32>,
                @location(7) in_color: vec4<f32>
            ) -> VsOut {
                let model = mat4x4<f32>(m0, m1, m2, m3);
                let world = model * vec4<f32>(in_pos, 1.0);
                var out: VsOut;
                out.pos = pc.view_proj * world;
                let n = model * vec4<f32>(in_normal, 0.0);
                out.world_normal = normalize(n.xyz);
                out.color = in_color;
                return out;
            }
        "#,
        ShaderStage::Vertex,
        "scene.vert",
    )?;
    let frag_spv = compile_shader(
        r#"
            struct PushConstants {
                view_proj: mat4x4<f32>,
                light_dir: vec4<f32>,
            };
            var<push_constant> pc: PushConstants;

            struct FsOut {
                @location(0) outAlbedo: vec4<f32>,
                @location(1) outNormal: vec4<f32>,
                @location(2) outMaterial: vec4<f32>,
                @location(3) outPresent: vec4<f32>,
            };

            @fragment
            fn main(
                @location(0) world_normal: vec3<f32>,
                @location(1) color: vec4<f32>
            ) -> FsOut {
                let n = normalize(world_normal);
                let l = normalize(pc.light_dir.xyz);
                let ndotl = max(dot(n, -l), 0.0);
                let ambient = 0.18;
                let rim = pow(1.0 - max(n.z, 0.0), 2.0) * 0.15;
                let lighting = ambient + ndotl * 0.9 + rim;
                let lit = color.rgb * lighting;
                var out: FsOut;
                out.outAlbedo = vec4<f32>(color.rgb, 1.0);
                out.outNormal = vec4<f32>(n * 0.5 + vec3<f32>(0.5, 0.5, 0.5), 1.0);
                out.outMaterial = vec4<f32>(0.04, 0.6, 0.0, 1.0);
                out.outPresent = vec4<f32>(lit, 1.0);
                return out;
            }
        "#,
        ShaderStage::Fragment,
        "scene.frag",
    )?;
    let vert_module = create_shader_module(device, &vert_spv)?;
    let frag_module = create_shader_module(device, &frag_spv)?;
    let entry_name = CString::new("main").expect("valid shader entry");
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_module)
            .name(&entry_name),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_module)
            .name(&entry_name),
    ];

    let bindings = [
        vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(size_of::<MeshVertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX),
        vk::VertexInputBindingDescription::default()
            .binding(1)
            .stride(size_of::<InstanceData>() as u32)
            .input_rate(vk::VertexInputRate::INSTANCE),
    ];
    let attrs = [
        // per-vertex
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(0),
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(12),
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(2)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(24),
        // per-instance model matrix columns
        vk::VertexInputAttributeDescription::default()
            .binding(1)
            .location(3)
            .format(vk::Format::R32G32B32A32_SFLOAT)
            .offset(0),
        vk::VertexInputAttributeDescription::default()
            .binding(1)
            .location(4)
            .format(vk::Format::R32G32B32A32_SFLOAT)
            .offset(16),
        vk::VertexInputAttributeDescription::default()
            .binding(1)
            .location(5)
            .format(vk::Format::R32G32B32A32_SFLOAT)
            .offset(32),
        vk::VertexInputAttributeDescription::default()
            .binding(1)
            .location(6)
            .format(vk::Format::R32G32B32A32_SFLOAT)
            .offset(48),
        // per-instance color
        vk::VertexInputAttributeDescription::default()
            .binding(1)
            .location(7)
            .format(vk::Format::R32G32B32A32_SFLOAT)
            .offset(64),
    ];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&bindings)
        .vertex_attribute_descriptions(&attrs);
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let dynamic_states = dynamic_viewport_state();
    let dynamic_state =
        vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .cull_mode(vk::CullModeFlags::BACK)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL);
    let attachment = vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(false)
        .color_write_mask(
            vk::ColorComponentFlags::R
                | vk::ColorComponentFlags::G
                | vk::ColorComponentFlags::B
                | vk::ColorComponentFlags::A,
        );
    let color_blend_attachments = [attachment, attachment, attachment, attachment];
    let color_blend =
        vk::PipelineColorBlendStateCreateInfo::default().attachments(&color_blend_attachments);
    let push_ranges = [vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
        .offset(0)
        .size(size_of::<ScenePush>() as u32)];
    let layout_info = vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push_ranges);
    let layout = unsafe { device.create_pipeline_layout(&layout_info, None) }?;
    let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .dynamic_state(&dynamic_state)
        .rasterization_state(&raster)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&color_blend)
        .layout(layout)
        .render_pass(render_pass)
        .subpass(0);
    let pipelines = unsafe {
        device.create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
    }
    .map_err(|(_, err)| anyhow!("failed to create scene pipeline: {err:?}"))?;
    unsafe {
        device.destroy_shader_module(vert_module, None);
        device.destroy_shader_module(frag_module, None);
    }
    Ok(ScenePipeline {
        layout,
        pipeline: pipelines[0],
    })
}

fn create_shader_module(device: &ash::Device, spirv_words: &[u32]) -> Result<vk::ShaderModule> {
    let info = vk::ShaderModuleCreateInfo::default().code(spirv_words);
    let module = unsafe { device.create_shader_module(&info, None) }?;
    Ok(module)
}

enum ShaderStage {
    Vertex,
    Fragment,
}

fn compile_shader(source: &str, stage: ShaderStage, name: &str) -> Result<Vec<u32>> {
    let module =
        naga::front::wgsl::parse_str(source).map_err(|e| anyhow!("WGSL parse error in {name}: {e}"))?;
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    let module_info = validator
        .validate(&module)
        .map_err(|e| anyhow!("WGSL validation error in {name}: {e}"))?;
    let pipeline_options = naga::back::spv::PipelineOptions {
        shader_stage: match stage {
            ShaderStage::Vertex => naga::ShaderStage::Vertex,
            ShaderStage::Fragment => naga::ShaderStage::Fragment,
        },
        entry_point: "main".to_string(),
    };
    let options = naga::back::spv::Options::default();
    let words =
        naga::back::spv::write_vec(&module, &module_info, &options, Some(&pipeline_options))
            .map_err(|e| anyhow!("SPIR-V generation failed in {name}: {e}"))?;
    Ok(words)
}

fn as_u8_slice<T>(slice: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(slice.as_ptr().cast::<u8>(), std::mem::size_of_val(slice)) }
}

fn begin_label(
    debug_utils: Option<&ash::ext::debug_utils::Device>,
    command_buffer: vk::CommandBuffer,
    name: &std::ffi::CStr,
) {
    if let Some(debug_utils) = debug_utils {
        let label = vk::DebugUtilsLabelEXT::default().label_name(name);
        unsafe {
            debug_utils.cmd_begin_debug_utils_label(command_buffer, &label);
        }
    }
}

fn end_label(debug_utils: Option<&ash::ext::debug_utils::Device>, command_buffer: vk::CommandBuffer) {
    if let Some(debug_utils) = debug_utils {
        unsafe {
            debug_utils.cmd_end_debug_utils_label(command_buffer);
        }
    }
}

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_types: vk::DebugUtilsMessageTypeFlagsEXT,
    callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    let message = if callback_data.is_null() {
        "<null callback data>"
    } else {
        let message_ptr = (*callback_data).p_message;
        if message_ptr.is_null() {
            "<null validation message>"
        } else {
            std::ffi::CStr::from_ptr(message_ptr)
                .to_str()
                .unwrap_or("<invalid utf8 validation message>")
        }
    };
    tracing::warn!(?message_severity, ?message_types, "{message}");
    vk::FALSE
}
