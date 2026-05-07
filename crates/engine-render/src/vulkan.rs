use std::ffi::CString;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{anyhow, Context, Result};
use ash::{vk, Entry};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::window::Window;

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

pub struct VulkanBackend {
    instance: ash::Instance,
    device: Arc<ash::Device>,
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
    descriptor_allocator: DescriptorAllocator,
    gbuffer: GBuffer,
    gbuffer_render_pass: vk::RenderPass,
    fullscreen_pipeline: FullscreenPipeline,
    framebuffers: Vec<vk::Framebuffer>,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    image_available: vk::Semaphore,
    render_finished: vk::Semaphore,
    in_flight: vk::Fence,
    frame_counter: AtomicU64,
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
        let fullscreen_pipeline =
            create_fullscreen_pipeline(&device, gbuffer_render_pass, extent)?;
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
            instance,
            device,
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
            descriptor_allocator,
            gbuffer,
            gbuffer_render_pass,
            fullscreen_pipeline,
            framebuffers,
            command_pool,
            command_buffers,
            image_available,
            render_finished,
            in_flight,
            frame_counter: AtomicU64::new(0),
        })
    }

    pub fn render_gbuffer_frame(&self) -> Result<()> {
        let _ = self.swapchain_images.len();
        let _ = self.swapchain_format;
        let _ = self.physical_device;
        let _ = self.graphics_queue_family;
        let _ = &self.descriptor_allocator;
        let frame = self.frame_counter.fetch_add(1, Ordering::Relaxed) as f32;
        let t = frame * 0.02;
        let present_r = (t.sin() * 0.5 + 0.5).clamp(0.0, 1.0);
        let present_g = ((t + 2.094).sin() * 0.5 + 0.5).clamp(0.0, 1.0);
        let present_b = ((t + 4.188).sin() * 0.5 + 0.5).clamp(0.0, 1.0);

        unsafe {
            self.device
                .wait_for_fences(&[self.in_flight], true, u64::MAX)?;
            self.device.reset_fences(&[self.in_flight])?;
        }
        let (image_index, _) = unsafe {
            self.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                self.image_available,
                vk::Fence::null(),
            )
        }?;

        let command_buffer = self.command_buffers[image_index as usize];
        unsafe {
            self.device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())?;
            let begin_info = vk::CommandBufferBeginInfo::default();
            self.device.begin_command_buffer(command_buffer, &begin_info)?;
            let clear_values = [
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.03, 0.03, 0.04, 1.0],
                    },
                },
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.5, 0.5, 1.0, 1.0],
                    },
                },
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.5, 0.5, 0.0, 1.0],
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
                        float32: [present_r, present_g, present_b, 1.0],
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
            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.fullscreen_pipeline.pipeline,
            );
            self.device.cmd_draw(command_buffer, 3, 1, 0, 0);
            self.device.cmd_end_render_pass(command_buffer);
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
            self.swapchain_loader
                .queue_present(self.graphics_queue, &present_info)?;
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
    for physical_device in devices {
        let queue_props =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        for (index, props) in queue_props.iter().enumerate() {
            let supports_graphics = props.queue_flags.contains(vk::QueueFlags::GRAPHICS);
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

fn create_swapchain(
    device: &ash::Device,
    surface_loader: &ash::khr::surface::Instance,
    swapchain_loader: &ash::khr::swapchain::Device,
    physical_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    queue_family: u32,
    width: u32,
    height: u32,
) -> Result<(
    vk::SwapchainKHR,
    Vec<vk::Image>,
    Vec<vk::ImageView>,
    vk::Format,
    vk::Extent2D,
)> {
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

    let extent = vk::Extent2D {
        width: width.clamp(
            capabilities.min_image_extent.width,
            capabilities.max_image_extent.width,
        ),
        height: height.clamp(
            capabilities.min_image_extent.height,
            capabilities.max_image_extent.height,
        ),
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
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);
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

fn create_fullscreen_pipeline(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    extent: vk::Extent2D,
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
                let wave = 0.5 + 0.5 * sin(uv.x * 9.0);
                let base = vec3<f32>(uv.x, uv.y, wave);
                out.outAlbedo = vec4<f32>(base, 1.0);
                out.outNormal = vec4<f32>(0.5, 0.5, 1.0, 1.0);
                out.outMaterial = vec4<f32>(0.04, 0.7, 0.0, 1.0);
                out.outPresent = vec4<f32>(base.z, base.y, base.x, 1.0);
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
    let viewport = vk::Viewport {
        x: 0.0,
        y: 0.0,
        width: extent.width as f32,
        height: extent.height as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    };
    let scissor = vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent,
    };
    let viewports = [viewport];
    let scissors = [scissor];
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewports(&viewports)
        .scissors(&scissors);
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::CLOCKWISE);
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
