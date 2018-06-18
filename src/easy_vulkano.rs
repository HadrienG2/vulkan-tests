use failure;

use log::{self, Level};

use std::{
    cmp::Ordering,
    ffi::CString,
    fmt::Write,
    result,
    sync::Arc,
};

use vulkano::{
    device::{
        Device,
        Queue,
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
        QueueFamily,
        RawInstanceExtensions,
    },
};


/// We use failure's type-erased error handling
pub type Result<T> = result::Result<T, failure::Error>;

/// A convenience abstraction for quickly prototyping vulkano-based code
///
/// You will likely want to keep the EasyVulkano object alive througout your
/// application's lifetime, even when you're not using it, as it connects
/// Vulkan's debug printout mechanism to your logging system.
///
pub struct EasyVulkano {
    /// Vulkan instance
    instance: Arc<Instance>,

    /// Vulkan debug callback
    _debug_callback: DebugCallback,
}

impl EasyVulkano {
    /// Sets up a Vulkan instance with debug logging
    ///
    /// This constructor is mostly a convenience wrapper around Vulkano's
    /// Instance::new() and DebugCallback::new() methods, please refer to the
    /// Instance::new() documentation to know what its parameters do.
    ///
    /// The Vulkan implementation capabilities will be described at the INFO
    /// logging level, which will help you quickly debug instance creation
    /// errors without needing to resort to use of external and finicky programs
    /// like vulkaninfo.
    ///
    /// In addition to the extensions that you specify, we will also enable the
    /// VK_EXT_debug_report extension as it is needed for debug logging.
    ///
    /// By default, debug messages are enabled based on the logger configuration
    /// at the time where this method is called. If this is not what you want
    /// (for example, if you want to adjust the logging level at runtime),
    /// please use the with_debug_config() constructor.
    ///
    pub fn new<'a>(
        app_infos: Option<&ApplicationInfo>,
        extensions: impl Into<RawInstanceExtensions>,
        layers: impl IntoIterator<Item=&'a &'a str>,
    ) -> Result<Self> {
        let max_log_level = log::max_level();
        Self::with_debug_config(
            app_infos,
            extensions,
            layers,
            MessageTypes {
                error: (max_log_level >= log::LevelFilter::Error),
                warning: (max_log_level >= log::LevelFilter::Warn),
                performance_warning: (max_log_level >= log::LevelFilter::Warn),
                information: (max_log_level >= log::LevelFilter::Info),
                debug: (max_log_level >= log::LevelFilter::Debug),
            }
        )
    }

    /// Like new(), but lets you specify manually which types of Vulkan debug
    /// reports you want to listen to.
    pub fn with_debug_config<'a>(
        app_infos: Option<&ApplicationInfo>,
        extensions: impl Into<RawInstanceExtensions>,
        layers: impl IntoIterator<Item=&'a &'a str>,
        messages: MessageTypes,
    ) -> Result<Self> {
        // Display Vulkan implementation information
        if log_enabled!(Level::Info) {
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

        let mut raw_extensions = extensions.into();
        raw_extensions.insert(CString::new("VK_EXT_debug_report")?);

        // Create our Vulkan instance
        let instance = Instance::new(app_infos, raw_extensions, layers)?;

        // Set up a debug callback
        let _debug_callback = DebugCallback::new(
            &instance,
            messages,
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
        )?;

        // Return the freshly built wrapper
        Ok(EasyVulkano {
            instance,
            _debug_callback,
        })
    }

    /// Select a physical device
    ///
    /// At the moment, EasyVulkano does not provide built-in support for multi-
    /// device Vulkan workflows, so you will need to pick which device you want
    /// to use. You can do so by telling us two things:
    ///
    /// - Which devices you can or cannot use (the "filter")
    /// - Given two devices, which of the two you prefer (the "preference")
    ///
    /// Like the EasyVulkano constructor, the physical device selector emits a
    /// lot of debug logs about your physical devices' actual capabilities,
    /// enabling you to promptly resolve device selection problems.
    ///
    pub fn select_physical_device(
        &self,
        filter: impl Fn(PhysicalDevice) -> bool,
        preference: impl Fn(PhysicalDevice, PhysicalDevice) -> Ordering
    ) -> Result<Option<PhysicalDevice>> {
        // Enumerate the physical devices
        info!("---- BEGINNING OF PHYSICAL DEVICE LIST ----");
        let mut favorite_device = None;
        for device in PhysicalDevice::enumerate(&self.instance) {
            // Low-level device and driver information
            info!("");
            info!("Device #{}: {}", device.index(), device.name());
            info!("Type: {:?}", device.ty());
            info!("Driver version: {}", device.driver_version());
            info!("PCI vendor/device id: 0x{:x}/0x{:x}",
                  device.pci_vendor_id(),
                  device.pci_device_id());
            if log_enabled!(Level::Info) {
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
                    "Robust buffer access support is mandated by the spec");

            // Queue families
            if log_enabled!(Level::Info) {
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
            if log_enabled!(Level::Info) {
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
            if log_enabled!(Level::Info) {
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
            info!("        * 1D: {}",
                  limits.max_image_dimension_1d());
            info!("        * 2D: {}",
                  limits.max_image_dimension_2d());
            info!("        * 3D: {}",
                  limits.max_image_dimension_3d());
            info!("        * Cube: {}",
                  limits.max_image_dimension_cube());
            info!("    - Max image array layers: {}",
                  limits.max_image_array_layers());
            info!("    - Max texel buffer elements: {}",
                  limits.max_texel_buffer_elements());
            info!("    - Max uniform buffer range: {}",
                  limits.max_uniform_buffer_range());
            info!("    - Max storage buffer range: {}",
                  limits.max_storage_buffer_range());
            info!("    - Max push constants size: {} bytes",
                  limits.max_push_constants_size());
            info!("    - Max memory allocation count: {}",
                  limits.max_memory_allocation_count());
            info!("    - Max sampler allocation count: {}",
                  limits.max_sampler_allocation_count());
            info!("    - Buffer image granularity: {} bytes",
                  limits.buffer_image_granularity());
            info!("    - Sparse address space size: {} bytes",
                  limits.sparse_address_space_size());
            info!("    - Max bound descriptor sets: {}",
                  limits.max_bound_descriptor_sets());
            info!("    - Max per-stage descriptors:");
            info!("        * Samplers: {}",
                  limits.max_per_stage_descriptor_samplers());
            info!("        * Uniform buffers: {}",
                  limits.max_per_stage_descriptor_uniform_buffers());
            info!("        * Storage buffers: {}",
                  limits.max_per_stage_descriptor_storage_buffers());
            info!("        * Sampled images: {}",
                  limits.max_per_stage_descriptor_sampled_images());
            info!("        * Storage images: {}",
                  limits.max_per_stage_descriptor_storage_images());
            info!("        * Input attachments: {}",
                  limits.max_per_stage_descriptor_input_attachments());
            info!("    - Max per-stage resources: {}",
                  limits.max_per_stage_resources());
            info!("    - Max descriptor set:");
            info!("        * Samplers: {}",
                  limits.max_descriptor_set_samplers());
            info!("        * Uniform buffers: {}",
                  limits.max_descriptor_set_uniform_buffers());
            info!("        * Dynamic uniform buffers: {}",
                  limits.max_descriptor_set_uniform_buffers_dynamic());
            info!("        * Storage buffers: {}",
                  limits.max_descriptor_set_storage_buffers());
            info!("        * Dynamic storage buffers: {}",
                  limits.max_descriptor_set_storage_buffers_dynamic());
            info!("        * Sampled images: {}",
                  limits.max_descriptor_set_sampled_images());
            info!("        * Storage images: {}",
                  limits.max_descriptor_set_storage_images());
            info!("        * Input attachments: {}",
                  limits.max_descriptor_set_input_attachments());
            info!("    - Vertex input limits:");
            info!("        * Max attributes: {}",
                  limits.max_vertex_input_attributes());
            info!("        * Max bindings: {}",
                  limits.max_vertex_input_bindings());
            info!("        * Max attribute offset: {}",
                  limits.max_vertex_input_attribute_offset());
            info!("        * Max binding stride: {}",
                  limits.max_vertex_input_binding_stride());
            info!("    - Max vertex output components: {}",
                  limits.max_vertex_output_components());
            info!("    - Max tesselation generation level: {}",
                  limits.max_tessellation_generation_level());
            info!("    - Max tesselation patch size: {} vertices",
                  limits.max_tessellation_patch_size());
            info!("    - Tesselation control shader limits:");
            info!("        * Inputs per vertex: {}",
                  limits.max_tessellation_control_per_vertex_input_components());
            info!("        * Outputs per vertex: {}",
                  limits.max_tessellation_control_per_vertex_output_components());
            info!("        * Outputs per patch: {}",
                  limits.max_tessellation_control_per_patch_output_components());
            info!("        * Total outputs: {}",
                  limits.max_tessellation_control_total_output_components());
            info!("    - Tesselation evaluation shader limits:");
            info!("        * Inputs: {}",
                  limits.max_tessellation_evaluation_input_components());
            info!("        * Outputs: {}",
                  limits.max_tessellation_evaluation_output_components());
            info!("    - Geometry shader limits:");
            info!("        * Invocations: {}",
                  limits.max_geometry_shader_invocations());
            info!("        * Inputs per vertex: {}",
                  limits.max_geometry_input_components());
            info!("        * Outputs per vertex: {}",
                  limits.max_geometry_output_components());
            info!("        * Emitted vertices: {}",
                  limits.max_geometry_output_vertices());
            info!("        * Total outputs: {}",
                  limits.max_geometry_total_output_components());
            info!("    - Fragment shader limits:");
            info!("        * Inputs: {}",
                  limits.max_fragment_input_components());
            info!("        * Output attachmnents: {}",
                  limits.max_fragment_output_attachments());
            info!("        * Dual-source output attachments: {}",
                  limits.max_fragment_dual_src_attachments());
            info!("        * Combined output resources: {}",
                  limits.max_fragment_combined_output_resources());
            info!("    - Compute shader limits:");
            info!("        * Shared memory: {} bytes",
                  limits.max_compute_shared_memory_size());
            info!("        * Work group count: {:?}",
                  limits.max_compute_work_group_count());
            info!("        * Work group invocations: {}",
                  limits.max_compute_work_group_invocations());
            info!("        * Work group size: {:?}",
                  limits.max_compute_work_group_size());
            info!("    - Sub-pixel precision: {} bits",
                  limits.sub_pixel_precision_bits());
            info!("    - Sub-texel precision: {} bits",
                  limits.sub_texel_precision_bits());
            info!("    - Mipmap precision: {} bits",
                  limits.mipmap_precision_bits());
            info!("    - Max draw index: {}",
                  limits.max_draw_indexed_index_value());
            info!("    - Max draws per indirect call: {}",
                  limits.max_draw_indirect_count());
            info!("    - Max sampler LOD bias: {}",
                  limits.max_sampler_lod_bias());
            info!("    - Max anisotropy: {}",
                  limits.max_sampler_anisotropy());
            info!("    - Max viewports: {}",
                  limits.max_viewports());
            info!("    - Max viewport dimensions: {:?}",
                  limits.max_viewport_dimensions());
            info!("    - Viewport bounds range: {:?}",
                  limits.viewport_bounds_range());
            info!("    - Viewport subpixel precision: {} bits",
                  limits.viewport_sub_pixel_bits());
            info!("    - Minimal alignments:");
            info!("        * Host allocations: {} bytes",
                  limits.min_memory_map_alignment());
            info!("        * Texel buffer offset: {} bytes",
                  limits.min_texel_buffer_offset_alignment());
            info!("        * Uniform buffer offset: {} bytes",
                  limits.min_uniform_buffer_offset_alignment());
            info!("        * Storage buffer offset: {} bytes",
                  limits.min_storage_buffer_offset_alignment());
            info!("    - Offset ranges:");
            info!("        * Texel fetch: [{}, {}]",
                  limits.min_texel_offset(),
                  limits.max_texel_offset());
            info!("        * Texel gather: [{}, {}]",
                  limits.min_texel_gather_offset(),
                  limits.max_texel_gather_offset());
            info!("        * Interpolation: [{}, {}]",
                  limits.min_interpolation_offset(),
                  limits.max_interpolation_offset());
            info!("    - Sub-pixel interpolation rounding: {} bits",
                  limits.sub_pixel_interpolation_offset_bits());
            info!("    - Framebuffer limits:");
            info!("        * Max size: [{}, {}]",
                  limits.max_framebuffer_width(),
                  limits.max_framebuffer_height());
            info!("        * Max layers: {}",
                  limits.max_framebuffer_layers());
            info!("        * Supported color sample counts: 0b{:b}",
                  limits.framebuffer_color_sample_counts());
            info!("        * Supported depth sample counts: 0b{:b}",
                  limits.framebuffer_depth_sample_counts());
            info!("        * Supported stencil sample counts: 0b{:b}",
                  limits.framebuffer_stencil_sample_counts());
            info!("        * Supported detached sample counts: 0b{:b}",
                  limits.framebuffer_no_attachments_sample_counts());
            info!("    - Max subpass color attachments: {}",
                  limits.max_color_attachments());
            info!("    - Supported sample counts for sampled images:");
            info!("        * Non-integer color: 0b{:b}",
                  limits.sampled_image_color_sample_counts());
            info!("        * Integer color: 0b{:b}",
                  limits.sampled_image_integer_sample_counts());
            info!("        * Depth: 0b{:b}",
                  limits.sampled_image_depth_sample_counts());
            info!("        * Stencil: 0b{:b}",
                  limits.sampled_image_stencil_sample_counts());
            info!("    - Supported storage image sample counts: 0b{:b}",
                  limits.storage_image_sample_counts());
            info!("    - Max SampleMask words: {}",
                  limits.max_sample_mask_words());
            info!("    - Timestamp support on compute and graphics queues: {}",
                  limits.timestamp_compute_and_graphics() != 0);
            info!("    - Timestamp period: {} ns",
                  limits.timestamp_period());
            info!("    - Max clip distances: {}",
                  limits.max_clip_distances());
            info!("    - Max cull distances: {}",
                  limits.max_cull_distances());
            info!("    - Max clip and cull distances: {}",
                  limits.max_combined_clip_and_cull_distances());
            info!("    - Discrete queue priorities: {}",
                  limits.discrete_queue_priorities());
            info!("    - Point size range: {:?}",
                  limits.point_size_range());
            info!("    - Line width range: {:?}",
                  limits.line_width_range());
            info!("    - Point size granularity: {}",
                  limits.point_size_granularity());
            info!("    - Line width granularity: {}",
                  limits.line_width_granularity());
            info!("    - Strict line rasterization: {}",
                  limits.strict_lines() != 0);
            info!("    - Standard sample locations: {}",
                  limits.standard_sample_locations() != 0);
            info!("    - Optimal buffer copy offset alignment: {} bytes",
                  limits.optimal_buffer_copy_offset_alignment());
            info!("    - Optimal buffer copy row pitch alignment: {} bytes",
                  limits.optimal_buffer_copy_row_pitch_alignment());
            info!("    - Non-coherent atom size: {} bytes",
                  limits.non_coherent_atom_size());

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
        info!("");
        info!("---- END OF PHYSICAL DEVICE LIST ----");

        // Return our physical device of choice (hopefully there is one)
        Ok(favorite_device)
    }

    /// Setup a device in a single-queue configuration
    ///
    /// The use of multiple command queues is key to making the most of the
    /// Vulkan API. But during prototyping, it is often convenient to stick with
    /// a simpler single-queue setup. This method provides you with such a setup
    /// with minimal fuss.
    ///
    /// Note that if you use select_physical_device() to pick your physical
    /// device, you may want to integrate your queue filter/preference into
    /// your device filter/preference.
    ///
    pub fn setup_single_queue_device(
        &self,
        phys_device: PhysicalDevice,
        features: &Features,
        extensions: &DeviceExtensions,
        filter: impl Fn(&QueueFamily) -> bool,
        preference: impl Fn(&QueueFamily, &QueueFamily) -> Ordering
    ) -> Result<Option<(Arc<Device>, Arc<Queue>)>> {
        // Select the appropriate queue family (if any)
        if let Some(queue_family) = phys_device.queue_families()
                                               .filter(filter)
                                               .max_by(preference)
        {
            // Build a single-queue device
            let (device, mut queues_iter) = Device::new(
                phys_device,
                features,
                extensions,
                [(queue_family, 1.0)].iter().cloned()
            )?;

            // Extract the only queue from the iterator (should always succeed,
            // if not it is a bug in vulkano or the Vulkan implementation)
            let queue = queues_iter.next().unwrap();
            assert!(queues_iter.next().is_none());

            // And now we can return the device and the queue
            Ok(Some((device, queue)))
        } else {
            // No suitable queue family was found :-/
            Ok(None)
        }
    }
}