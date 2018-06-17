#[macro_use] extern crate failure;
#[macro_use] extern crate log;
#[macro_use] extern crate vulkano;
#[macro_use] extern crate vulkano_shader_derive;

extern crate env_logger;
extern crate image;

use image::{
    ImageBuffer,
    Rgba,
};

use log::Level;

use std::{
    cmp::Ordering,
    fmt::Write,
    mem,
    sync::Arc,
};

use vulkano::{
    buffer::{
        BufferUsage,
        CpuAccessibleBuffer,
    },
    command_buffer::{
        AutoCommandBufferBuilder,
        CommandBuffer,
        DynamicState,
    },
    descriptor::descriptor_set::PersistentDescriptorSet,
    device::{
        Device,
        Queue,
    },
    format::{
        ClearValue,
        Format,
    },
    framebuffer::{
        Framebuffer,
        Subpass,
    },
    image::{
        AttachmentImage,
        Dimensions,
        ImageUsage,
        StorageImage,
    },
    instance::{
        self,
        debug::{
            DebugCallback,
            MessageTypes,
        },
        ApplicationInfo,
        DeviceExtensions,
        Features,
        Instance,
        InstanceExtensions,
        PhysicalDevice,
        PhysicalDeviceType,
        Version,
    },
    pipeline::{
        ComputePipeline,
        GraphicsPipeline,
        viewport::Viewport,
    },
    sync::GpuFuture,
};


// === GENERAL-PURPOSE LOGIC (+ DEBUG PRINTOUT) ===

// TODO: Manage errors more carefully
// TODO: Review device selection in light of new program requirements
// TODO: Split this code up in multiple modules

/// We use failure's type-erased error handling
type Result<T> = std::result::Result<T, failure::Error>;

/// Create an instance of a Vulkan context
///
/// This is basically a thin wrapper around vulkano's Instance::new() which
/// optionally logs extra diagnosis information about the host.
///
fn create_instance(application_info: Option<&ApplicationInfo>,
                   extensions: &InstanceExtensions,
                   layers: &[&str]) -> Result<Arc<Instance>> {
    // Display detailed diagnostics information, if requested.
    if log_enabled!(Level::Info) {
        // Display the application info
        info!("Application info: {:?}", application_info);

        // Display available instance extensions
        let supported_exts = InstanceExtensions::supported_by_core()?;
        info!("Supported instance extensions: {:?}", supported_exts);

        // Display available instance layers
        info!("Available instance layers:");
        for layer in instance::layers_list()? {
            info!("    - {} ({}) [Version {}, targeting Vulkan v{}]",
                  layer.name(),
                  layer.description(),
                  layer.implementation_version(),
                  layer.vulkan_version());
        }
    }

    // Create our instance
    Ok(Instance::new(application_info, extensions, layers)?)
}

// Select a single physical device, with lots of debug printout
fn select_physical_device(
    instance: &Arc<Instance>,
    filter: impl Fn(PhysicalDevice) -> bool,
    preference: impl Fn(PhysicalDevice, PhysicalDevice) -> Ordering
) -> Result<PhysicalDevice> {
        // Enumerate the physical devices
    info!("---- BEGINNING OF PHYSICAL DEVICE LIST ----");
    let mut favorite_device = None;
    for device in PhysicalDevice::enumerate(instance) {
        // Low-level device and driver information
        info!("\nDevice #{}: {}", device.index(), device.name());
        info!("Type: {:?}", device.ty());
        info!("Driver version: {}", device.driver_version());
        info!("PCI vendor/device id: 0x{:x}/0x{:x}",
              device.pci_vendor_id(),
              device.pci_device_id());
        if log_enabled!(Level::Debug) {
            let uuid = device.uuid();
            let mut uuid_str = String::with_capacity(2 * uuid.len());
            for byte in uuid {
                write!(&mut uuid_str, "{:02x}", byte)?;
            }
            info!("UUID: 0x{}", uuid_str);
        }

        // Supported Vulkan API version and extensions
        info!("Vulkan API version: {}", device.api_version());
        info!("Supported device extensions: {:?}",
              DeviceExtensions::supported_by_device(device));

        // Supported Vulkan features
        let supported_features = device.supported_features();
        info!("{:#?}", supported_features);
        ensure!(supported_features.robust_buffer_access,
                "Robust buffer access support is mandated by the Vulkan spec");

        // Queue families
        if log_enabled!(Level::Debug) {
            info!("Queue familie(s):");
            let mut family_str = String::new();
            for family in device.queue_families() {
                family_str.clear();
                write!(&mut family_str,
                       "    {}: {} queue(s) for ",
                       family.id(),
                       family.queues_count())?;
                if family.supports_graphics() {
                    write!(&mut family_str, "graphics, ")?;
                }
                if family.supports_compute() {
                    write!(&mut family_str, "compute, " )?;
                }
                if family.supports_transfers() {
                    write!(&mut family_str, "transfers, ")?;
                }
                if family.supports_sparse_binding() {
                    write!(&mut family_str, "sparse resource bindings, ")?;
                }
                info!("{}", family_str);
            }
        }

        // Memory types
        if log_enabled!(Level::Debug) {
            info!("Memory type(s):");
            let mut type_str = String::new();
            for memory_type in device.memory_types() {
                type_str.clear();
                write!(&mut type_str,
                       "    {}: from heap #{}, ",
                       memory_type.id(),
                       memory_type.heap().id())?;
                if memory_type.is_device_local() {
                    write!(&mut type_str, "on device, ")?;
                } else {
                    write!(&mut type_str, "on host, ")?;
                }
                if memory_type.is_host_visible() {
                    write!(&mut type_str, "host-visible, ")?;
                } else {
                    write!(&mut type_str, "only accessible by device, ")?;
                }
                if memory_type.is_host_coherent() {
                    write!(&mut type_str, "host-coherent, ")?;
                }
                if memory_type.is_host_cached() {
                    write!(&mut type_str, "host-cached, ")?;
                }
                if memory_type.is_lazily_allocated() {
                    write!(&mut type_str, "lazily allocated, ")?;
                }
                info!("{}", type_str);
            }
        }

        // Memory heaps
        if log_enabled!(Level::Debug) {
            info!("Memory heap(s):");
            let mut heap_str = String::new();
            for heap in device.memory_heaps() {
                heap_str.clear();
                write!(&mut heap_str,
                       "    {}: {} bytes, ",
                       heap.id(),
                       heap.size())?;
                if heap.is_device_local() {
                    write!(&mut heap_str, "on device, ")?;
                } else {
                    write!(&mut heap_str, "on host, ")?;
                }
                info!("{}", heap_str);
            }
        }

        // Device limits
        info!("Device limits:");
        let limits = device.limits();
        info!("    - Max image dimension:");
        info!("        * 1D: {}", limits.max_image_dimension_1d());
        info!("        * 2D: {}", limits.max_image_dimension_2d());
        info!("        * 3D: {}", limits.max_image_dimension_3d());
        info!("        * Cube: {}", limits.max_image_dimension_cube());
        info!("    - Max image array layers: {}", limits.max_image_array_layers());
        info!("    - Max texel buffer elements: {}", limits.max_texel_buffer_elements());
        info!("    - Max uniform buffer range: {}", limits.max_uniform_buffer_range());
        info!("    - Max storage buffer range: {}", limits.max_storage_buffer_range());
        info!("    - Max push constants size: {} bytes", limits.max_push_constants_size());
        info!("    - Max memory allocation count: {}", limits.max_memory_allocation_count());
        info!("    - Max sampler allocation count: {}", limits.max_sampler_allocation_count());
        info!("    - Buffer image granularity: {} bytes", limits.buffer_image_granularity());
        info!("    - Sparse address space size: {} bytes", limits.sparse_address_space_size());
        info!("    - Max bound descriptor sets: {}", limits.max_bound_descriptor_sets());
        info!("    - Max per-stage descriptors:");
        info!("        * Samplers: {}", limits.max_per_stage_descriptor_samplers());
        info!("        * Uniform buffers: {}", limits.max_per_stage_descriptor_uniform_buffers());
        info!("        * Storage buffers: {}", limits.max_per_stage_descriptor_storage_buffers());
        info!("        * Sampled images: {}", limits.max_per_stage_descriptor_sampled_images());
        info!("        * Storage images: {}", limits.max_per_stage_descriptor_storage_images());
        info!("        * Input attachments: {}", limits.max_per_stage_descriptor_input_attachments());
        info!("    - Max per-stage resources: {}", limits.max_per_stage_resources());
        info!("    - Max descriptor set:");
        info!("        * Samplers: {}", limits.max_descriptor_set_samplers());
        info!("        * Uniform buffers: {}", limits.max_descriptor_set_uniform_buffers());
        info!("        * Dynamic uniform buffers: {}", limits.max_descriptor_set_uniform_buffers_dynamic());
        info!("        * Storage buffers: {}", limits.max_descriptor_set_storage_buffers());
        info!("        * Dynamic storage buffers: {}", limits.max_descriptor_set_storage_buffers_dynamic());
        info!("        * Sampled images: {}", limits.max_descriptor_set_sampled_images());
        info!("        * Storage images: {}", limits.max_descriptor_set_storage_images());
        info!("        * Input attachments: {}", limits.max_descriptor_set_input_attachments());
        info!("    - Vertex input limits:");
        info!("        * Max attributes: {}", limits.max_vertex_input_attributes());
        info!("        * Max bindings: {}", limits.max_vertex_input_bindings());
        info!("        * Max attribute offset: {}", limits.max_vertex_input_attribute_offset());
        info!("        * Max binding stride: {}", limits.max_vertex_input_binding_stride());
        info!("    - Max vertex output components: {}", limits.max_vertex_output_components());
        info!("    - Max tesselation generation level: {}", limits.max_tessellation_generation_level());
        info!("    - Max tesselation patch size: {} vertices", limits.max_tessellation_patch_size());
        info!("    - Tesselation control shader limits:");
        info!("        * Inputs per vertex: {}", limits.max_tessellation_control_per_vertex_input_components());
        info!("        * Outputs per vertex: {}", limits.max_tessellation_control_per_vertex_output_components());
        info!("        * Outputs per patch: {}", limits.max_tessellation_control_per_patch_output_components());
        info!("        * Total outputs: {}", limits.max_tessellation_control_total_output_components());
        info!("    - Tesselation evaluation shader limits:");
        info!("        * Inputs: {}", limits.max_tessellation_evaluation_input_components());
        info!("        * Outputs: {}", limits.max_tessellation_evaluation_output_components());
        info!("    - Geometry shader limits:");
        info!("        * Invocations: {}", limits.max_geometry_shader_invocations());
        info!("        * Inputs per vertex: {}", limits.max_geometry_input_components());
        info!("        * Outputs per vertex: {}", limits.max_geometry_output_components());
        info!("        * Emitted vertices: {}", limits.max_geometry_output_vertices());
        info!("        * Total outputs: {}", limits.max_geometry_total_output_components());
        info!("    - Fragment shader limits:");
        info!("        * Inputs: {}", limits.max_fragment_input_components());
        info!("        * Output attachmnents: {}", limits.max_fragment_output_attachments());
        info!("        * Dual-source output attachments: {}", limits.max_fragment_dual_src_attachments());
        info!("        * Combined output resources: {}", limits.max_fragment_combined_output_resources());
        info!("    - Compute shader limits:");
        info!("        * Shared memory: {} bytes", limits.max_compute_shared_memory_size());
        info!("        * Work group count: {:?}", limits.max_compute_work_group_count());
        info!("        * Work group invocations: {}", limits.max_compute_work_group_invocations());
        info!("        * Work group size: {:?}", limits.max_compute_work_group_size());
        info!("    - Sub-pixel precision: {} bits", limits.sub_pixel_precision_bits());
        info!("    - Sub-texel precision: {} bits", limits.sub_texel_precision_bits());
        info!("    - Mipmap precision: {} bits", limits.mipmap_precision_bits());
        info!("    - Max draw index: {}", limits.max_draw_indexed_index_value());
        info!("    - Max draws per indirect call: {}", limits.max_draw_indirect_count());
        info!("    - Max sampler LOD bias: {}", limits.max_sampler_lod_bias());
        info!("    - Max anisotropy: {}", limits.max_sampler_anisotropy());
        info!("    - Max viewports: {}", limits.max_viewports());
        info!("    - Max viewport dimensions: {:?}", limits.max_viewport_dimensions());
        info!("    - Viewport bounds range: {:?}", limits.viewport_bounds_range());
        info!("    - Viewport subpixel precision: {} bits", limits.viewport_sub_pixel_bits());
        info!("    - Minimal alignments:");
        info!("        * Host allocations: {} bytes", limits.min_memory_map_alignment());
        info!("        * Texel buffer offset: {} bytes", limits.min_texel_buffer_offset_alignment());
        info!("        * Uniform buffer offset: {} bytes", limits.min_uniform_buffer_offset_alignment());
        info!("        * Storage buffer offset: {} bytes", limits.min_storage_buffer_offset_alignment());
        info!("    - Offset ranges:");
        info!("        * Texel fetch: [{}, {}]", limits.min_texel_offset(), limits.max_texel_offset());
        info!("        * Texel gather: [{}, {}]", limits.min_texel_gather_offset(), limits.max_texel_gather_offset());
        info!("        * Interpolation: [{}, {}]", limits.min_interpolation_offset(), limits.max_interpolation_offset());
        info!("    - Sub-pixel interpolation rounding: {} bits", limits.sub_pixel_interpolation_offset_bits());
        info!("    - Framebuffer limits:");
        info!("        * Max size: [{}, {}]", limits.max_framebuffer_width(), limits.max_framebuffer_height());
        info!("        * Max layers: {}", limits.max_framebuffer_layers());
        info!("        * Supported color sample counts: 0b{:b}", limits.framebuffer_color_sample_counts());
        info!("        * Supported depth sample counts: 0b{:b}", limits.framebuffer_depth_sample_counts());
        info!("        * Supported stencil sample counts: 0b{:b}", limits.framebuffer_stencil_sample_counts());
        info!("        * Supported detached sample counts: 0b{:b}", limits.framebuffer_no_attachments_sample_counts());
        info!("    - Max subpass color attachments: {}", limits.max_color_attachments());
        info!("    - Supported sample counts for sampled images:");
        info!("        * Non-integer color: 0b{:b}", limits.sampled_image_color_sample_counts());
        info!("        * Integer color: 0b{:b}", limits.sampled_image_integer_sample_counts());
        info!("        * Depth: 0b{:b}", limits.sampled_image_depth_sample_counts());
        info!("        * Stencil: 0b{:b}", limits.sampled_image_stencil_sample_counts());
        info!("    - Supported storage image sample counts: 0b{:b}", limits.storage_image_sample_counts());
        info!("    - Max SampleMask words: {}", limits.max_sample_mask_words());
        info!("    - Timestamp support on compute and graphics queues: {}", limits.timestamp_compute_and_graphics() != 0);
        info!("    - Timestamp period: {} ns", limits.timestamp_period());
        info!("    - Max clip distances: {}", limits.max_clip_distances());
        info!("    - Max cull distances: {}", limits.max_cull_distances());
        info!("    - Max clip and cull distances: {}", limits.max_combined_clip_and_cull_distances());
        info!("    - Discrete queue priorities: {}", limits.discrete_queue_priorities());
        info!("    - Point size range: {:?}", limits.point_size_range());
        info!("    - Line width range: {:?}", limits.line_width_range());
        info!("    - Point size granularity: {}", limits.point_size_granularity());
        info!("    - Line width granularity: {}", limits.line_width_granularity());
        info!("    - Strict line rasterization: {}", limits.strict_lines() != 0);
        info!("    - Standard sample locations: {}", limits.standard_sample_locations() != 0);
        info!("    - Optimal buffer copy offset alignment: {} bytes", limits.optimal_buffer_copy_offset_alignment());
        info!("    - Optimal buffer copy row pitch alignment: {} bytes", limits.optimal_buffer_copy_row_pitch_alignment());
        info!("    - Non-coherent atom size: {} bytes", limits.non_coherent_atom_size());

        // Does it fit our selection criteria?
        let is_selected = filter(device);
        info!("Selected: {}", is_selected);

        // If so, do we consider it better than devices seen before (if any)?
        if is_selected {
            let is_better = if let Some(best_so_far) = favorite_device {
                preference(device, best_so_far) == Ordering::Greater
            } else {
                true
            };
            if is_better { favorite_device = Some(device); }
            info!("Preferred: {}", is_better);
        }
    }
    info!("\n---- END OF PHYSICAL DEVICE LIST ----");

    // Return our physical device of choice (hopefully there is one)
    favorite_device.ok_or(failure::err_msg("No suitable physical device found"))
}


// === APPLICATION-SPECIFIC LOGIC ===

// Creates a debug callback to see output from validation layers
fn new_debug_callback(instance: &Arc<Instance>) -> Result<DebugCallback> {
    let max_log_level = log::max_level();
    Ok(DebugCallback::new(
        instance,
        MessageTypes {
            error: (max_log_level >= log::LevelFilter::Error),
            warning: (max_log_level >= log::LevelFilter::Warn),
            performance_warning: (max_log_level >= log::LevelFilter::Warn),
            information: (max_log_level >= log::LevelFilter::Info),
            debug: (max_log_level >= log::LevelFilter::Debug),
        },
        |msg| {
            let log_level = match msg.ty {
                MessageTypes { error: true, .. } => Level::Error,
                MessageTypes { performance_warning: true, .. }
                | MessageTypes { warning: true, .. } => Level::Warn,
                MessageTypes { information: true, .. } => Level::Info,
                MessageTypes { debug: true, .. } => Level::Debug,
                _ => unimplemented!()
            };
            log!(log_level,
                 "VULKAN{}{}{}{}{} @ {} \t=> {}",
                 if msg.ty.error { " ERRO" } else { "" },
                 if msg.ty.warning { " WARN" } else { "" },
                 if msg.ty.performance_warning { " PERF" } else { "" },
                 if msg.ty.information { " INFO" } else { "" },
                 if msg.ty.debug { " DEBG" } else { "" },
                 msg.layer_prefix, msg.description);
        }
    )?)
}

// Tells whether we can use a certain physical device or not
fn device_filter(dev: PhysicalDevice,
                 features: &Features,
                 extensions: &DeviceExtensions) -> bool {
    // This code was written against Vulkan v1.0.76. We tolerate older
    // patch releases and new minor versions but not new major versions.
    let min_ver = Version { major: 1, minor: 0, patch: 0 };
    let max_ver = Version { major: 2, minor: 0, patch: 0 };
    if (dev.api_version() < min_ver) || (dev.api_version() >= max_ver) {
        return false;
    }

    // For this learning exercise, we want at least a hybrid graphics + compute
    // queue (this implies data transfer support)
    if dev.queue_families()
          .find(|f| f.supports_graphics() && f.supports_compute())
          .is_none()
    {
        return false;
    }

    // Some features may be requested by the user, we need to look at them
    if !dev.supported_features().superset_of(features) {
        return false;
    }

    // Same goes for extensions
    let unsupported_exts =
        extensions.difference(&DeviceExtensions::supported_by_device(dev));
    if unsupported_exts != DeviceExtensions::none() {
        return false;
    }

    // TODO: May end up looking at device limits and memory heap sizes as well.

    // If control reaches this point, we can use this device
    true
}

// Tells how acceptable device "dev1" compares to alternative "dev2"
fn device_preference(dev1: PhysicalDevice, dev2: PhysicalDevice) -> Ordering {
    // Device type preference should suffice most of the time
    use PhysicalDeviceType::*;
    let type_pref = match (dev1.ty(), dev2.ty()) {
        // If both devices have the same type, this doesn't play a role
        (same1, same2) if same1 == same2 => Ordering::Equal,

        // Discrete GPU goes first
        (DiscreteGpu, _) => Ordering::Greater,
        (_, DiscreteGpu) => Ordering::Less,

        // Then comes integrated GPU
        (IntegratedGpu, _) => Ordering::Greater,
        (_, IntegratedGpu) => Ordering::Less,

        // Then comes virtual GPU
        (VirtualGpu, _) => Ordering::Greater,
        (_, VirtualGpu) => Ordering::Less,

        // Then comes "other" (can't be worse than CPU?)
        (Other, _) => Ordering::Greater,
        (_, Other) => Ordering::Less,

        // We have actually covered all cases, but Rust can't see it :(
        _ => unreachable!(),
    };
    if type_pref != Ordering::Equal { return type_pref; }

    // Dedicated data transfer queues are also useful. The more of them
    // we have the better, I think.
    let dedic_transfers = |dev: PhysicalDevice| -> usize {
        dev.queue_families()
           .filter(|q| q.supports_transfers() &&
                       !(q.supports_graphics() || q.supports_compute()))
           .map(|q| q.queues_count())
           .sum()

    };
    let queue_pref = dedic_transfers(dev1).cmp(&dedic_transfers(dev2));
    if queue_pref != Ordering::Equal { return queue_pref; }

    // TODO: Could also look at dedicated compute queues if we ever end
    //       up interested in asynchronous compute.

    // If control reaches this point, we like both devices equally
    Ordering::Equal
}

// Try reading from and writing to vulkano's simplest buffer type
fn test_buffer_read_write(device: &Arc<Device>) {
    // This is a basic buffer type that can contain anything
    let buffer = CpuAccessibleBuffer::from_data(device.clone(),
                                                BufferUsage::all(),
                                                42usize)
                                     .expect("Failed to create buffer");

    // We can access it with RWLock-like semantics
    let mut writer = buffer.write().expect("Buffer should be unlocked");
    assert_eq!(*writer, 42usize);
    *writer = 43;
    mem::drop(writer);
    let reader = buffer.read().expect("Buffer should be unlocked");
    assert_eq!(*reader, 43);
}

// Try it again, but this time make the GPU perform the memory copy
fn test_buffer_copy(device: &Arc<Device>, queue: &Arc<Queue>) {
    // Build a source and destination buffer
    let source = CpuAccessibleBuffer::from_iter(device.clone(),
                                                BufferUsage::transfer_source(),
                                                1..42)
                                     .expect("Failed to create source buffer");
    let dest =
        CpuAccessibleBuffer::from_iter(device.clone(),
                                       BufferUsage::transfer_destination(),
                                       (1..42).map(|_| 0))
                            .expect("Failed to create destination buffer");

    // Tell the GPU to eagerly copy from the source to the destination
    let command_buffer =
        AutoCommandBufferBuilder::new(device.clone(), queue.family())
                                 .expect("Failed to start a command buffer")
                                 .copy_buffer(source.clone(), dest.clone())
                                 .expect("Failed to enqueue a buffer copy")
                                 .build()
                                 .expect("Failed to build the command buffer");
    let future = command_buffer.execute(queue.clone())
                               .expect("Failed to submit command to driver");
    future.then_signal_fence_and_flush().expect("Failed to flush the future")
          .wait(None).expect("Failed to await the copy");

    // Check that the copy was performed correctly
    let source_data = source.read().expect("Source buffer should be unlocked");
    let dest_data = dest.read().expect("Dest buffer should be unlocked");
    assert_eq!(*source_data, *dest_data, "Source and dest should now be equal");
}

// Do a computation on a buffer using a compute pipeline
fn test_buffer_compute(device: &Arc<Device>, queue: &Arc<Queue>) {
    // Here is some data
    let data_buffer =
        CpuAccessibleBuffer::from_iter(device.clone(),
                                       BufferUsage {
                                           storage_buffer: true,
                                           .. BufferUsage::none()
                                       },
                                       0..65536)
                            .expect("Failed to create buffer");

    // We will process this data using the following compute shader
    #[allow(unused)]
    mod cs {
        #[derive(VulkanoShader)]
        #[ty = "compute"]
        #[src = "
#version 460

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer Data {
    uint data[];
} buf;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    buf.data[idx] *= 42;
}
        "]
        struct Dummy;
    }

    // Load the shader into the Vulkan implementation
    let shader = cs::Shader::load(device.clone())
                            .expect("Failed to load shader module");

    // Set up a compute pipeline containing that shader
    let pipeline =
        Arc::new(ComputePipeline::new(device.clone(),
                                      &shader.main_entry_point(),
                                      &())
                                 .expect("Failed to create compute pipeline"));

    // Build a descriptor set to attach that pipeline to our buffer
    let descriptor_set = Arc::new(
        PersistentDescriptorSet::start(pipeline.clone(), 0)
                                .add_buffer(data_buffer.clone())
                                .expect("Failed to add buffer to descriptors")
                                .build()
                                .expect("Failed to build descriptor set")
    );

    // Build a command buffer that runs the computation
    let command_buffer =
        AutoCommandBufferBuilder::new(device.clone(), queue.family())
                                 .expect("Failed to start a command buffer")
                                 .dispatch([1024, 1, 1],
                                           pipeline.clone(),
                                           descriptor_set.clone(),
                                           ())
                                 .expect("Failed to add dispatch command")
                                 .build()
                                 .expect("Failed to build command buffer");

    // Schedule the computation and wait for it
    command_buffer.execute(queue.clone())
                  .expect("Failed to submit command buffer to driver")
                  .then_signal_fence_and_flush()
                  .expect("Failed to flush the future")
                  .wait(None)
                  .expect("Failed to await the computation");

    // Check that it worked
    let content = data_buffer.read().expect("Failed to access buffer");
    for (n, val) in content.iter().enumerate() {
        assert_eq!(*val, (n as u32) * 42);
    }
}

// Let's perform some basic operations with an image
fn test_image_basics(device: &Arc<Device>, queue: &Arc<Queue>) {
    // Create an image
    let (image, init) =
        StorageImage::uninitialized(device.clone(),
                                    Dimensions::Dim2d { width: 1024,
                                                        height: 1024 },
                                    Format::R8G8B8A8Unorm,
                                    ImageUsage {
                                       transfer_source: true,
                                       transfer_destination: true,
                                       storage: true,
                                       .. ImageUsage::none()
                                    },
                                    Some(queue.family()))
                     .expect("Failed to create image");

    // Create a buffer to copy the final image contents in
    let buf = CpuAccessibleBuffer::from_iter(device.clone(),
                                             BufferUsage {
                                                transfer_destination: true,
                                                .. BufferUsage::none()
                                             },
                                             (0 .. 1024 * 1024 *4).map(|_| 0u8))
                                  .expect("Failed to create dest buffer");

    // Ask the GPU to fill the image with purple
    let command_buffer =
        AutoCommandBufferBuilder::new(device.clone(), queue.family())
                                 .expect("Failed to start a command buffer")
                                 .clear_color_image(
                                    init,
                                    ClearValue::Float([0.5, 0.0, 0.5, 1.0])
                                 )
                                 .expect("Failed to add clear command")
                                 .copy_image_to_buffer(image.clone(),
                                                       buf.clone())
                                 .expect("Failed to add copy command")
                                 .build()
                                 .expect("Failed to build command buffer");

    // Execute and await the command buffer
    command_buffer.execute(queue.clone())
                  .expect("Failed to submit the command buffer")
                  .then_signal_fence_and_flush()
                  .expect("Failed to flush the future")
                  .wait(None)
                  .expect("Failed to await the computation");

    // Extract the image data from the buffer where it's been copied
    let buf_content = buf.read().expect("Failed to read buffer");
    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(1024,
                                                     1024,
                                                     &buf_content[..])
                                           .expect("Failed to decode image");

    // Save the image to a PNG file
    image.save("deep_purple.png").expect("Failed to save image");
}

// And now, let's comput an image using a compute shader
fn test_image_compute(device: &Arc<Device>, queue: &Arc<Queue>) {
    // This is the image which we will eventually write into
    let (image, init) =
        StorageImage::uninitialized(device.clone(),
                                    Dimensions::Dim2d { width: 1024,
                                                        height: 1024 },
                                    Format::R8G8B8A8Unorm,
                                    ImageUsage {
                                       transfer_source: true,
                                       transfer_destination: true,
                                       storage: true,
                                       .. ImageUsage::none()
                                    },
                                    Some(queue.family()))
                     .expect("Failed to create image");

    // Build a command buffer that clears the image
    let command_buffer =
        AutoCommandBufferBuilder::new(device.clone(), queue.family())
                                 .expect("Failed to start a command buffer")
                                 .clear_color_image(
                                    init,
                                    ClearValue::Float([0.0, 0.0, 0.0, 1.0])
                                 )
                                 .expect("Failed to add clear command")
                                 .build()
                                 .expect("Failed to build command buffer");

    // Start running the clear command
    let clear_finished =
        command_buffer.execute(queue.clone())
                      .expect("Failed to submit command buffer to driver");

    // We will generate this image using the following compute shader
    #[allow(unused)]
    mod cs {
        #[derive(VulkanoShader)]
        #[ty = "compute"]
        #[src = "
#version 460

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, rgba8) uniform writeonly image2D img;

void main() {
    vec2 norm_coordinates = (gl_GlobalInvocationID.xy + vec2(0.5)) / vec2(imageSize(img));
    vec2 c = (norm_coordinates - vec2(0.5)) * 2.0 - vec2(1.0, 0.0);

    vec2 z = vec2(0.0, 0.0);
    float i;
    for (i = 0.0; i < 1.0; i += 0.005) {
        z = vec2(
            z.x * z.x - z.y * z.y + c.x,
            z.y * z.x + z.x * z.y + c.y
        );

        if (length(z) > 4.0) {
            break;
        }
    }

    vec4 to_write = vec4(0.0, i, 0.0, 1.0);
    imageStore(img, ivec2(gl_GlobalInvocationID.xy), to_write);
}
        "]
        struct Dummy;
    }

    // Load the shader into the Vulkan implementation
    let shader = cs::Shader::load(device.clone())
                            .expect("Failed to load shader module");

    // Set up a compute pipeline containing that shader
    let pipeline =
        Arc::new(ComputePipeline::new(device.clone(),
                                      &shader.main_entry_point(),
                                      &())
                                 .expect("Failed to create compute pipeline"));

    // Build a descriptor set to attach that pipeline to our buffer
    let descriptor_set = Arc::new(
        PersistentDescriptorSet::start(pipeline.clone(), 0)
                                .add_image(image.clone())
                                .expect("Failed to add image to descriptors")
                                .build()
                                .expect("Failed to build descriptor set")
    );

    // Create a buffer to copy the final image contents in
    let buf = CpuAccessibleBuffer::from_iter(device.clone(),
                                             BufferUsage {
                                                transfer_destination: true,
                                                .. BufferUsage::none()
                                             },
                                             (0 .. 1024 * 1024 *4).map(|_| 0u8))
                                  .expect("Failed to create dest buffer");

    // Build a command buffer that runs the computation
    let command_buffer =
        AutoCommandBufferBuilder::new(device.clone(), queue.family())
                                 .expect("Failed to start a command buffer")
                                 .dispatch([1024 / 8, 1024 / 8, 1],
                                           pipeline.clone(),
                                           descriptor_set.clone(),
                                           ())
                                 .expect("Failed to add dispatch command")
                                 .copy_image_to_buffer(image.clone(),
                                                       buf.clone())
                                 .expect("Failed to add image copy")
                                 .build()
                                 .expect("Failed to build command buffer");

    // Schedule the computation and wait for it
    command_buffer.execute_after(clear_finished, queue.clone())
                  .expect("Failed to submit command buffer to driver")
                  .then_signal_fence_and_flush()
                  .expect("Failed to flush the future")
                  .wait(None)
                  .expect("Failed to await the computation");

    // Extract the image data from the buffer where it's been copied
    let buf_content = buf.read().expect("Failed to read buffer");
    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(1024,
                                                     1024,
                                                     &buf_content[..])
                                           .expect("Failed to decode image");

    // Save the image to a PNG file
    image.save("mandelbrot.png").expect("Failed to save image");
}

// Draw a triangle. How hard could that get?
fn test_triangle(device: &Arc<Device>, queue: &Arc<Queue>) {
    // Our triangle vertices will be described by this struct
    #[derive(Copy, Clone)]
    struct Vertex {
        position: [f32; 2],
    }
    impl_vertex!(Vertex, position);

    // Here they are!
    let vertex1 = Vertex { position: [-0.5, -0.5] };
    let vertex2 = Vertex { position: [ 0.0,  0.5] };
    let vertex3 = Vertex { position: [ 0.5, -0.75] };
    let vertices = [vertex1, vertex2, vertex3];

    // Let's store them in a buffer
    let vx_buf = CpuAccessibleBuffer::from_iter(device.clone(),
                                                BufferUsage {
                                                    vertex_buffer: true,
                                                    .. BufferUsage::none()
                                                },
                                                vertices.iter().cloned())
                                     .expect("Failed to create buffer");

    // We will process them using the following vertex shader...
    #[allow(unused)]
    mod vs {
        #[derive(VulkanoShader)]
        #[ty = "vertex"]
        #[src = "
#version 460

layout(location = 0) in vec2 position;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
}
        "]
        struct Dummy;
    }
    let vs = vs::Shader::load(device.clone())
                        .expect("Failed to create vertex shader module");

    // ...and the following fragment shader
    #[allow(unused)]
    mod fs {
        #[derive(VulkanoShader)]
        #[ty = "fragment"]
        #[src = "
#version 460

layout(location = 0) out vec4 f_color;

void main() {
    f_color = vec4(1.0, 0.5, 0.0, 1.0);
}
        "]
        struct Dummy;
    }
    let fs = fs::Shader::load(device.clone())
                        .expect("Failed to create fragment shader module");

    // Vulkan wants to know more about our rendering intents, via a "renderpass"
    let render_pass = Arc::new(
        single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: Format::R8G8B8A8Unorm,
                    samples: 1,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {}
            }
        ).expect("Failed to create renderpass")
    );


    // We bring these together in a single graphics pipeline
    let pipeline = Arc::new(
        GraphicsPipeline::start()
                         // - We have only one source of vertex inputs
                         .vertex_input_single_buffer::<Vertex>()
                         // - This is our vertex shader
                         .vertex_shader(vs.main_entry_point(), ())
                         // - This sets up the target image region (viewport)
                         .viewports_dynamic_scissors_irrelevant(1)
                         // - This is our fragment shader
                         .fragment_shader(fs.main_entry_point(), ())
                         // - The pipeline is used in this render pass
                         .render_pass(Subpass::from(render_pass.clone(), 0)
                                              .expect("Failed to set subpass"))
                         // - Here is the target device, now build the pipeline!
                         .build(device.clone())
                         .expect("Failed to build graphics pipeline")
    );

    // This is the image which we will eventually write into
    let image =
        AttachmentImage::with_usage(device.clone(),
                                    [1024, 1024],
                                    Format::R8G8B8A8Unorm,
                                    ImageUsage {
                                       transfer_source: true,
                                       color_attachment: true,
                                       .. ImageUsage::none()
                                    })
                        .expect("Failed to create image");

    // Create a buffer to copy the final image contents in
    let buf = CpuAccessibleBuffer::from_iter(device.clone(),
                                             BufferUsage {
                                                transfer_destination: true,
                                                .. BufferUsage::none()
                                             },
                                             (0 .. 1024 * 1024 *4).map(|_| 0u8))
                                  .expect("Failed to create dest buffer");

    // A renderpass must be attached to its drawing target(s) via a framebuffer
    let framebuffer = Arc::new(
        Framebuffer::start(render_pass.clone())
                    .add(image.clone())
                    .expect("Failed to add an image to the framebuffer")
                    .build()
                    .expect("Failed to build the framebuffer")
    );

    // And now, all we need is a viewport...
    let dynamic_state = DynamicState {
        viewports: Some(vec![Viewport {
            origin: [0.0, 0.0],
            dimensions: [1024.0, 1024.0],
            depth_range: 0.0 .. 1.0,
        }]),
        .. DynamicState::none()
    };

    // ...and we are ready to build our command buffer
    let clear_values = vec![[0.0, 0.2, 0.8, 1.0].into()];
    let command_buffer =
        AutoCommandBufferBuilder::primary_one_time_submit(device.clone(),
                                                          queue.family())
                                 .expect("Failed to start command buffer")
                                 .begin_render_pass(framebuffer.clone(),
                                                    false,
                                                    clear_values)
                                 .expect("Failed to start renderpass")
                                 .draw(pipeline.clone(),
                                       dynamic_state,
                                       vx_buf.clone(),
                                       (),
                                       ())
                                 .expect("Failed to build draw call")
                                 .end_render_pass()
                                 .expect("Failed to exit renderpass")
                                 .copy_image_to_buffer(image.clone(),
                                                       buf.clone())
                                 .expect("Failed to final copy")
                                 .build()
                                 .expect("Failed to build command buffer");

    // The rest is business as usual: run the computation...
    command_buffer.execute(queue.clone())
                  .expect("Failed to submit commands")
                  .then_signal_fence_and_flush()
                  .expect("Failed to set up a fence")
                  .wait(None)
                  .expect("Failed to await work completion");

    // ...and save it to disk. Phew!
    let content = buf.read().unwrap();
    ImageBuffer::<Rgba<u8>, _>::from_raw(1024,
                                         1024,
                                         &content[..])
                .expect("Failed to decode image")
                .save("mighty_triangle.png")
                .expect("Failed to save image to disk");
}

// Application entry point
fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();

    // This is a Vulkan test program
    println!("Hello and welcome to this Vulkan test program!");

    // Create our Vulkan instance
    println!("* Setting up instance...");
    let instance = create_instance(
        Some(&app_info_from_cargo_toml!()),
        &InstanceExtensions {
            ext_debug_report: true,
            .. InstanceExtensions::none()
        },
        &["VK_LAYER_LUNARG_standard_validation"]
    )?;

    // Set up debug logging
    println!("* Setting up debug logging...");
    let _debug_callback = new_debug_callback(&instance)?;

    // Decide which device features and extensions we want to use
    let features = Features {
        robust_buffer_access: true,
        .. Features::none()
    };
    let extensions = DeviceExtensions::none();

    // Select our physical device accordingly
    println!("* Selecting physical device and queue family...");
    let phys_device = select_physical_device(
        &instance,
        |dev| device_filter(dev, &features, &extensions),
        device_preference
    )?;

    // Find a family of graphics + compute queues
    //
    // Right now, we only intend to do graphics and compute, on a single queue,
    // without sparse binding magic, so any graphics- and compute-capable queue
    // family is the same by our standards. We take the first queue we find,
    // assuming the GPU driver developer picked the queue family order wisely.
    // We can assume that there will be such a queue because we checked for it
    // in our device filter.
    //
    let graphics_and_compute_family =
        phys_device.queue_families()
                   .find(|q| q.supports_graphics() && q.supports_compute())
                   .expect("This error should be handled by the device filter");

    // Create our logical device
    println!("* Setting up logical device and queue...");
    let (device, mut queues_iter) = Device::new(
        phys_device,
        &features,
        &extensions,
        [(graphics_and_compute_family, 1.0)].iter().cloned()
    )?;

    // As we only use one queue at the moment, we can use a logical shortcut
    let queue = queues_iter.next().expect("Vulkan failed to provide a queue");
    assert!(queues_iter.next().is_none(),
            "This code must be updated, it assumes there is only one queue");

    // Let's play a bit with vulkano's buffer abstraction
    println!("* Playing with command buffers...");
    test_buffer_read_write(&device);
    test_buffer_copy(&device, &queue);
    test_buffer_compute(&device, &queue);

    // And then let's play with the image abstraction too!
    println!("* Playing with images...");
    test_image_basics(&device, &queue);
    test_image_compute(&device, &queue);

    // Finally, we can achieve the pinnacle of any modern graphics API, namely
    // drawing a colored triangle. Everything goes downhill from there.
    println!("* And finally, drawing a triangle...");
    test_triangle(&device, &queue);

    // ...and then everything will be teared down automagically
    println!("We're done!");
    Ok(())
}