#[macro_use]
extern crate vulkano;

use std::{
    cmp::Ordering,
    collections::HashSet,
    sync::Arc,
};

use vulkano::{
    device::{
        Device,
        QueuesIter,
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
        QueueFamily,
        Version,
    }
};


// === GENERAL-PURPOSE LOGIC (+ DEBUG PRINTOUT) ===

// TODO: Make the debug output conditional and isolated
// TODO: Manage errors more carefully

// Create an instance of a Vulkan context, with lots of debug printout
fn create_instance(application_info: Option<&ApplicationInfo>,
                   extensions: &InstanceExtensions,
                   layers: &[&str]) -> Arc<Instance> {
    // Display the application info
    println!("Application info: {:?}\n", application_info);

    // Enumerate and display available instance extensions
    let supported_instance_exts =
        InstanceExtensions::supported_by_core()
                           .expect("Failed to load Vulkan loader");
    println!("Supported instance extensions: {:?}\n", supported_instance_exts);

    // Make sure that the Vulkan implementation supports all required extensions
    assert_eq!(extensions.difference(&supported_instance_exts),
               InstanceExtensions::none());

    // Enumerate and display available instance layers
    println!("Available instance layers:");
    let mut available_instance_layers = HashSet::new();
    for layer in instance::layers_list().expect("Failed to load Vulkan lib") {
        println!("    - {} ({}) [Version {}, targeting Vulkan v{}]",
                 layer.name(),
                 layer.description(),
                 layer.implementation_version(),
                 layer.vulkan_version());
        available_instance_layers.insert(layer.name().to_owned());
    }
    println!();

    // Make sure that the Vulkan implementation supports all requested layers
    for layer in layers {
        assert!(available_instance_layers.contains(*layer));
    }

    // Create our instance
    Instance::new(application_info, extensions, layers)
             .expect("Failed to create instance")
}

// Select a single physical device, with lots of debug printout
fn select_physical_device(
    instance: &Arc<Instance>,
    filter: impl Fn(PhysicalDevice) -> bool,
    preference: impl Fn(PhysicalDevice, PhysicalDevice) -> Ordering
) -> PhysicalDevice {
        // Enumerate the physical devices
    println!("---- BEGINNING OF PHYSICAL DEVICE LIST ----\n");
    let mut favorite_device = None;
    for device in PhysicalDevice::enumerate(instance) {
        // Low-level device and driver information
        println!("Device #{}: {}", device.index(), device.name());
        println!("Type: {:?}", device.ty());
        println!("Driver version: {}", device.driver_version());
        println!("PCI vendor/device id: {:x}/{:x}",
                 device.pci_vendor_id(),
                 device.pci_device_id());
        print!("UUID: ");
        for byte in device.uuid() {
            print!("{:x}", byte);
        }
        println!();

        // Supported Vulkan API version and extensions
        println!("Vulkan API version: {}", device.api_version());
        let supported_exts = DeviceExtensions::supported_by_device(device);
        println!("Supported device extensions: {:?}", supported_exts);

        // Supported Vulkan features
        let supported_features = device.supported_features();
        println!("{:#?}", supported_features);
        assert!(supported_features.robust_buffer_access,
                "Robust buffer access support is mandated by the Vulkan spec");

        // Queue families
        println!("Queue familie(s):");
        for family in device.queue_families() {
            print!("    {}: {} queue(s) for ",
                   family.id(),
                   family.queues_count());
            if family.supports_graphics() {
                print!("graphics, ");
            }
            if family.supports_compute() {
                print!("compute, " );
            }
            if family.supports_transfers() {
                print!("transfers, ");
            }
            if family.supports_sparse_binding() {
                print!("sparse resource bindings, ");
            }
            println!();
        }

        // Memory types
        println!("Memory type(s):");
        for memory_type in device.memory_types() {
            print!("    {}: from heap #{}, ",
                   memory_type.id(),
                   memory_type.heap().id());
            if memory_type.is_device_local() {
                print!("on device, ");
            } else {
                print!("on host, ");
            }
            if memory_type.is_host_visible() {
                print!("host-visible, ");
            } else {
                print!("only accessible by device, ");
            }
            if memory_type.is_host_coherent() {
                print!("host-coherent, ");
            }
            if memory_type.is_host_cached() {
                print!("host-cached, ");
            }
            if memory_type.is_lazily_allocated() {
                print!("lazily allocated, ");
            }
            println!();
        }

        // Memory heaps
        println!("Memory heap(s):");
        for heap in device.memory_heaps() {
            print!("    {}: {} bytes, ", heap.id(), heap.size());
            if heap.is_device_local() {
                print!("on device, ");
            } else {
                print!("on host, ");
            }
            println!();
        }

        // Device limits
        println!("Device limits:");
        let limits = device.limits();
        println!("    - Max image dimension:");
        println!("        * 1D: {}", limits.max_image_dimension_1d());
        println!("        * 2D: {}", limits.max_image_dimension_2d());
        println!("        * 3D: {}", limits.max_image_dimension_3d());
        println!("        * Cube: {}", limits.max_image_dimension_cube());
        println!("    - Max image array layers: {}", limits.max_image_array_layers());
        println!("    - Max texel buffer elements: {}", limits.max_texel_buffer_elements());
        println!("    - Max uniform buffer range: {}", limits.max_uniform_buffer_range());
        println!("    - Max storage buffer range: {}", limits.max_storage_buffer_range());
        println!("    - Max push constants size: {} bytes", limits.max_push_constants_size());
        println!("    - Max memory allocation count: {}", limits.max_memory_allocation_count());
        println!("    - Max sampler allocation count: {}", limits.max_sampler_allocation_count());
        println!("    - Buffer image granularity: {} bytes", limits.buffer_image_granularity());
        println!("    - Sparse address space size: {} bytes", limits.sparse_address_space_size());
        println!("    - Max bound descriptor sets: {}", limits.max_bound_descriptor_sets());
        println!("    - Max per-stage descriptors:");
        println!("        * Samplers: {}", limits.max_per_stage_descriptor_samplers());
        println!("        * Uniform buffers: {}", limits.max_per_stage_descriptor_uniform_buffers());
        println!("        * Storage buffers: {}", limits.max_per_stage_descriptor_storage_buffers());
        println!("        * Sampled images: {}", limits.max_per_stage_descriptor_sampled_images());
        println!("        * Storage images: {}", limits.max_per_stage_descriptor_storage_images());
        println!("        * Input attachments: {}", limits.max_per_stage_descriptor_input_attachments());
        println!("    - Max per-stage resources: {}", limits.max_per_stage_resources());
        println!("    - Max descriptor set:");
        println!("        * Samplers: {}", limits.max_descriptor_set_samplers());
        println!("        * Uniform buffers: {}", limits.max_descriptor_set_uniform_buffers());
        println!("        * Dynamic uniform buffers: {}", limits.max_descriptor_set_uniform_buffers_dynamic());
        println!("        * Storage buffers: {}", limits.max_descriptor_set_storage_buffers());
        println!("        * Dynamic storage buffers: {}", limits.max_descriptor_set_storage_buffers_dynamic());
        println!("        * Sampled images: {}", limits.max_descriptor_set_sampled_images());
        println!("        * Storage images: {}", limits.max_descriptor_set_storage_images());
        println!("        * Input attachments: {}", limits.max_descriptor_set_input_attachments());
        println!("    - Vertex input limits:");
        println!("        * Max attributes: {}", limits.max_vertex_input_attributes());
        println!("        * Max bindings: {}", limits.max_vertex_input_bindings());
        println!("        * Max attribute offset: {}", limits.max_vertex_input_attribute_offset());
        println!("        * Max binding stride: {}", limits.max_vertex_input_binding_stride());
        println!("    - Max vertex output components: {}", limits.max_vertex_output_components());
        println!("    - Max tesselation generation level: {}", limits.max_tessellation_generation_level());
        println!("    - Max tesselation patch size: {} vertices", limits.max_tessellation_patch_size());
        println!("    - Tesselation control shader limits:");
        println!("        * Inputs per vertex: {}", limits.max_tessellation_control_per_vertex_input_components());
        println!("        * Outputs per vertex: {}", limits.max_tessellation_control_per_vertex_output_components());
        println!("        * Outputs per patch: {}", limits.max_tessellation_control_per_patch_output_components());
        println!("        * Total outputs: {}", limits.max_tessellation_control_total_output_components());
        println!("    - Tesselation evaluation shader limits:");
        println!("        * Inputs: {}", limits.max_tessellation_evaluation_input_components());
        println!("        * Outputs: {}", limits.max_tessellation_evaluation_output_components());
        println!("    - Geometry shader limits:");
        println!("        * Invocations: {}", limits.max_geometry_shader_invocations());
        println!("        * Inputs per vertex: {}", limits.max_geometry_input_components());
        println!("        * Outputs per vertex: {}", limits.max_geometry_output_components());
        println!("        * Emitted vertices: {}", limits.max_geometry_output_vertices());
        println!("        * Total outputs: {}", limits.max_geometry_total_output_components());
        println!("    - Fragment shader limits:");
        println!("        * Inputs: {}", limits.max_fragment_input_components());
        println!("        * Output attachmnents: {}", limits.max_fragment_output_attachments());
        println!("        * Dual-source output attachments: {}", limits.max_fragment_dual_src_attachments());
        println!("        * Combined output resources: {}", limits.max_fragment_combined_output_resources());
        println!("    - Compute shader limits:");
        println!("        * Shared memory: {} bytes", limits.max_compute_shared_memory_size());
        println!("        * Work group count: {:?}", limits.max_compute_work_group_count());
        println!("        * Work group invocations: {}", limits.max_compute_work_group_invocations());
        println!("        * Work group size: {:?}", limits.max_compute_work_group_size());
        println!("    - Sub-pixel precision: {} bits", limits.sub_pixel_precision_bits());
        println!("    - Sub-texel precision: {} bits", limits.sub_texel_precision_bits());
        println!("    - Mipmap precision: {} bits", limits.mipmap_precision_bits());
        println!("    - Max draw index: {}", limits.max_draw_indexed_index_value());
        println!("    - Max draws per indirect call: {}", limits.max_draw_indirect_count());
        println!("    - Max sampler LOD bias: {}", limits.max_sampler_lod_bias());
        println!("    - Max anisotropy: {}", limits.max_sampler_anisotropy());
        println!("    - Max viewports: {}", limits.max_viewports());
        println!("    - Max viewport dimensions: {:?}", limits.max_viewport_dimensions());
        println!("    - Viewport bounds range: {:?}", limits.viewport_bounds_range());
        println!("    - Viewport subpixel precision: {} bits", limits.viewport_sub_pixel_bits());
        println!("    - Minimal alignments:");
        println!("        * Host allocations: {} bytes", limits.min_memory_map_alignment());
        println!("        * Texel buffer offset: {} bytes", limits.min_texel_buffer_offset_alignment());
        println!("        * Uniform buffer offset: {} bytes", limits.min_uniform_buffer_offset_alignment());
        println!("        * Storage buffer offset: {} bytes", limits.min_storage_buffer_offset_alignment());
        println!("    - Offset ranges:");
        println!("        * Texel fetch: [{}, {}]", limits.min_texel_offset(), limits.max_texel_offset());
        println!("        * Texel gather: [{}, {}]", limits.min_texel_gather_offset(), limits.max_texel_gather_offset());
        println!("        * Interpolation: [{}, {}]", limits.min_interpolation_offset(), limits.max_interpolation_offset());
        println!("    - Sub-pixel interpolation rounding: {} bits", limits.sub_pixel_interpolation_offset_bits());
        println!("    - Framebuffer limits:");
        println!("        * Max size: [{}, {}]", limits.max_framebuffer_width(), limits.max_framebuffer_height());
        println!("        * Max layers: {}", limits.max_framebuffer_layers());
        println!("        * Supported color sample counts: 0b{:b}", limits.framebuffer_color_sample_counts());
        println!("        * Supported depth sample counts: 0b{:b}", limits.framebuffer_depth_sample_counts());
        println!("        * Supported stencil sample counts: 0b{:b}", limits.framebuffer_stencil_sample_counts());
        println!("        * Supported detached sample counts: 0b{:b}", limits.framebuffer_no_attachments_sample_counts());
        println!("    - Max subpass color attachments: {}", limits.max_color_attachments());
        println!("    - Supported sample counts for sampled images:");
        println!("        * Non-integer color: 0b{:b}", limits.sampled_image_color_sample_counts());
        println!("        * Integer color: 0b{:b}", limits.sampled_image_integer_sample_counts());
        println!("        * Depth: 0b{:b}", limits.sampled_image_depth_sample_counts());
        println!("        * Stencil: 0b{:b}", limits.sampled_image_stencil_sample_counts());
        println!("    - Supported storage image sample counts: 0b{:b}", limits.storage_image_sample_counts());
        println!("    - Max SampleMask words: {}", limits.max_sample_mask_words());
        println!("    - Timestamp support on compute and graphics queues: {}", limits.timestamp_compute_and_graphics() != 0);
        println!("    - Timestamp period: {} ns", limits.timestamp_period());
        println!("    - Max clip distances: {}", limits.max_clip_distances());
        println!("    - Max cull distances: {}", limits.max_cull_distances());
        println!("    - Max clip and cull distances: {}", limits.max_combined_clip_and_cull_distances());
        println!("    - Discrete queue priorities: {}", limits.discrete_queue_priorities());
        println!("    - Point size range: {:?}", limits.point_size_range());
        println!("    - Line width range: {:?}", limits.line_width_range());
        println!("    - Point size granularity: {}", limits.point_size_granularity());
        println!("    - Line width granularity: {}", limits.line_width_granularity());
        println!("    - Strict line rasterization: {}", limits.strict_lines() != 0);
        println!("    - Standard sample locations: {}", limits.standard_sample_locations() != 0);
        println!("    - Optimal buffer copy offset alignment: {} bytes", limits.optimal_buffer_copy_offset_alignment());
        println!("    - Optimal buffer copy row pitch alignment: {} bytes", limits.optimal_buffer_copy_row_pitch_alignment());
        println!("    - Non-coherent atom size: {} bytes", limits.non_coherent_atom_size());

        // Does it fit our selection criteria?
        let is_selected = filter(device);
        println!("Selected: {}", is_selected);

        // If so, do we consider it better than devices seen before (if any)?
        if is_selected {
            let is_better = if let Some(best_so_far) = favorite_device {
                preference(device, best_so_far) == Ordering::Greater
            } else {
                true
            };
            if is_better { favorite_device = Some(device); }
            println!("Preferred: {}", is_better);
        }

        // A newline separates two physical device descriptions
        println!();
    }
    println!("---- END OF PHYSICAL DEVICE LIST ----");

    // Return our physical device of choice (hopefully there is one)
    favorite_device.expect("No suitable physical device found")
}

// Set up a logical device from a physical device
fn new_logical_device<'a>(
    dev: PhysicalDevice,
    features: &Features,
    extensions: &DeviceExtensions,
    queues: impl IntoIterator<Item=(QueueFamily<'a>, f32)>
) -> (Arc<Device>, QueuesIter) {
    // Check that requested features are supported
    assert!(dev.supported_features().superset_of(features),
            "Some requested features are not supported by the physical device");

    // Check that requested extensions are supported
    assert_eq!(
        extensions.difference(&DeviceExtensions::supported_by_device(dev)),
        DeviceExtensions::none(),
        "Some requested extensions are not supported by the physical device"
    );

    // We'll leave the queue family check to Vulkano
    Device::new(dev, features, extensions, queues)
           .expect("Failed to create a logical device")
}


// === APPLICATION-SPECIFIC LOGIC ===

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

    // TODO: May end up looking at device limits as well.

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

    // Device-local memory is very important for performance too
    let has_local_mem = |dev: PhysicalDevice| {
        dev.memory_heaps().find(|h| h.is_device_local()).is_some()

    };
    let mem_pref = has_local_mem(dev1).cmp(&has_local_mem(dev2));
    if mem_pref != Ordering::Equal { return mem_pref; }

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
    //       up doing compute too.

    // If control reaches this point, we like both devices equally
    Ordering::Equal
}

// Application entry point
fn main() {
    // This is a Vulkan test program
    println!("Hello and welcome to this Vulkan test program!\n");

    // Create our Vulkan instance
    let instance = create_instance(
        Some(&app_info_from_cargo_toml!()),
        &InstanceExtensions {
            ext_debug_report: true,
            .. InstanceExtensions::none()
        },
        &["VK_LAYER_LUNARG_standard_validation"]
    );

    // Set up debug logging
    let _debug_callback = DebugCallback::new(
        &instance,
        MessageTypes {
            error: true,
            warning: true,
            performance_warning: true,
            information: false,
            debug: true,
        },
        |msg| {
            println!("#{}{}{}{}{}: {} \t=> {}",
                     if msg.ty.error { " ERRO" } else { "" },
                     if msg.ty.warning { " WARN" } else { "" },
                     if msg.ty.performance_warning { " PERF" } else { "" },
                     if msg.ty.information { " INFO" } else { "" },
                     if msg.ty.debug { " DEBG" } else { "" },
                     msg.layer_prefix, msg.description);
        }
    ).expect("Failed to setup a debug callback");

    // Decide which device features and extensions we want to use
    let features = Features {
        robust_buffer_access: true,
        .. Features::none()
    };
    let extensions = DeviceExtensions::none();

    // Select our physical device accordingly
    let phys_device = select_physical_device(
        &instance,
        |dev| device_filter(dev, &features, &extensions),
        device_preference
    );

    // Find a family of graphics queues
    //
    // Right now, we only intend to do graphics, on a single queue, without
    // sparse binding magic, so any graphics-capable queue family is the same by
    // our standards. We take the first queue we find, assuming the GPU driver
    // developer picked the queue family order wisely. We can assume that there
    // will be such a queue because we checked for it in our device filter.
    //
    let graphics_family =
        phys_device.queue_families()
                   .find(|q| q.supports_graphics())
                   .expect("This error should be handled by the device filter");

    // Create our logical device
    let (device, mut queues_iter) = new_logical_device(
        phys_device,
        &features,
        &extensions,
        [(graphics_family, 1.0)].iter().cloned()
    );

    // As we only use one queue at the moment, we can use a logical shortcut
    let queue = queues_iter.next();
    assert!(queues_iter.next().is_none(),
            "This code must be updated, it assumes there is only one queue");

    // TODO: Explorer buffers
}