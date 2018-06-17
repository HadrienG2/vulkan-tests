#[macro_use] extern crate failure;
#[macro_use] extern crate log;
#[macro_use] extern crate vulkano;
#[macro_use] extern crate vulkano_shader_derive;

extern crate env_logger;
extern crate image;

mod easy_vulkano;

use easy_vulkano::{EasyVulkano, Result};

use image::{
    ImageBuffer,
    Rgba,
};

use std::{
    cmp::Ordering,
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
        DeviceExtensions,
        Features,
        InstanceExtensions,
        PhysicalDevice,
        PhysicalDeviceType,
        QueueFamily,
        Version,
    },
    pipeline::{
        ComputePipeline,
        GraphicsPipeline,
        viewport::Viewport,
    },
    sync::GpuFuture,
};


// TODO: Review device selection in light of new program requirements
// TODO: Split this code up in multiple modules

// === DEVICE AND QUEUE SELECTION CRITERIA ===

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

    // At least one device queue family should fit our needs
    if dev.queue_families().find(queue_filter).is_none() {
        return false;
    }

    // Some features may be requested by the user, we need to look at them
    if !dev.supported_features().superset_of(features) {
        return false;
    }

    // Same goes for device extensions
    let unsupported_exts =
        extensions.difference(&DeviceExtensions::supported_by_device(dev));
    if unsupported_exts != DeviceExtensions::none() {
        return false;
    }

    // TODO: May end up looking at device limits and memory heap sizes as well.

    // If control reaches this point, we can use this device
    true
}

// Tells whether we can use a certain queue family or not
fn queue_filter(family: &QueueFamily) -> bool {
    // For this learning exercise, we want at least a hybrid graphics + compute
    // queue (this implies data transfer support)
    family.supports_graphics() && family.supports_compute()
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

    // Figure out which queue family we would pick on each device
    fn target_family(dev: PhysicalDevice) -> QueueFamily {
        dev.queue_families()
           .filter(queue_filter)
           .max_by(queue_preference)
           .expect("Device filtering failed")
    }
    let (fam1, fam2) = (target_family(dev1), target_family(dev2));
    let queue_pref = queue_preference(&fam1, &fam2);
    if queue_pref != Ordering::Equal { return queue_pref; }

    // If control reaches this point, we like both devices equally
    Ordering::Equal
}

// Tells whether we like a certain queue family or not
fn queue_preference(_fam1: &QueueFamily, _fam2: &QueueFamily) -> Ordering {
    // Right now, we only intend to do graphics and compute, on a single queue,
    // without sparse binding magic, so any graphics- and compute-capable queue
    // family is the same by our standards.
    Ordering::Equal
}


// === VULKANO EXAMPLES ===

// Try reading from and writing to vulkano's simplest buffer type
fn test_buffer_read_write(device: &Arc<Device>) -> Result<()> {
    // This is a basic buffer type that can contain anything
    let buffer = CpuAccessibleBuffer::from_data(device.clone(),
                                                BufferUsage::all(),
                                                42usize)?;

    // We can access it with RWLock-like semantics
    let mut writer = buffer.write()?;
    ensure!(*writer == 42, failure::err_msg("Initial buffer value is wrong"));
    *writer = 43;
    mem::drop(writer);
    let reader = buffer.read()?;
    ensure!(*reader == 43, failure::err_msg("Final buffer value is wrong"));
    Ok(())
}

// Try it again, but this time make the GPU perform the memory copy
fn test_buffer_copy(device: &Arc<Device>, queue: &Arc<Queue>) -> Result<()> {
    // Build a source and destination buffer
    let source = CpuAccessibleBuffer::from_iter(device.clone(),
                                                BufferUsage::transfer_source(),
                                                1..42)?;
    let dest =
        CpuAccessibleBuffer::from_iter(device.clone(),
                                       BufferUsage::transfer_destination(),
                                       (1..42).map(|_| 0))?;

    // Tell the GPU to eagerly copy from the source to the destination
    let command_buffer =
        AutoCommandBufferBuilder::new(device.clone(), queue.family())?
                                 .copy_buffer(source.clone(), dest.clone())?
                                 .build()?;
    let future = command_buffer.execute(queue.clone())?;
    future.then_signal_fence_and_flush()?
          .wait(None)?;

    // Check that the copy was performed correctly
    let source_data = source.read()?;
    let dest_data = dest.read()?;
    ensure!(*source_data == *dest_data,
            failure::err_msg("Buffer copy failed (dest != source)"));
    Ok(())
}

// Do a computation on a buffer using a compute pipeline
fn test_buffer_compute(device: &Arc<Device>, queue: &Arc<Queue>) -> Result<()> {
    // Here is some data
    let data_buffer =
        CpuAccessibleBuffer::from_iter(device.clone(),
                                       BufferUsage {
                                           storage_buffer: true,
                                           .. BufferUsage::none()
                                       },
                                       0..65536)?;

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
    let shader = cs::Shader::load(device.clone())?;

    // Set up a compute pipeline containing that shader
    let pipeline =
        Arc::new(ComputePipeline::new(device.clone(),
                                      &shader.main_entry_point(),
                                      &())?);

    // Build a descriptor set to attach that pipeline to our buffer
    let descriptor_set = Arc::new(
        PersistentDescriptorSet::start(pipeline.clone(), 0)
                                .add_buffer(data_buffer.clone())?
                                .build()?
    );

    // Build a command buffer that runs the computation
    let command_buffer =
        AutoCommandBufferBuilder::new(device.clone(), queue.family())?
                                 .dispatch([1024, 1, 1],
                                           pipeline.clone(),
                                           descriptor_set.clone(),
                                           ())?
                                 .build()?;

    // Schedule the computation and wait for it
    command_buffer.execute(queue.clone())?
                  .then_signal_fence_and_flush()?
                  .wait(None)?;

    // Check that it worked
    let content = data_buffer.read()?;
    for (n, val) in content.iter().enumerate() {
        ensure!(*val == (n as u32) * 42,
                failure::err_msg("Final buffer content is wrong"));
    }
    Ok(())
}

// Let's perform some basic operations with an image
fn test_image_basics(device: &Arc<Device>, queue: &Arc<Queue>) -> Result<()> {
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
                                    Some(queue.family()))?;

    // Create a buffer to copy the final image contents in
    let buf =
        CpuAccessibleBuffer::from_iter(device.clone(),
                                       BufferUsage {
                                          transfer_destination: true,
                                          .. BufferUsage::none()
                                       },
                                       (0 .. 1024 * 1024 *4).map(|_| 0u8))?;

    // Ask the GPU to fill the image with purple
    let command_buffer =
        AutoCommandBufferBuilder::new(device.clone(), queue.family())?
                                 .clear_color_image(
                                    init,
                                    ClearValue::Float([0.5, 0.0, 0.5, 1.0])
                                 )?
                                 .copy_image_to_buffer(image.clone(),
                                                       buf.clone())?
                                 .build()?;

    // Execute and await the command buffer
    command_buffer.execute(queue.clone())?
                  .then_signal_fence_and_flush()?
                  .wait(None)?;

    // Extract the image data from the buffer where it's been copied
    let buf_content = buf.read()?;
    let image =
        ImageBuffer::<Rgba<u8>, _>::from_raw(1024,
                                             1024,
                                             &buf_content[..])
                    .ok_or(failure::err_msg("Container is not big enough"))?;

    // Save the image to a PNG file
    Ok(image.save("deep_purple.png")?)
}

// And now, let's comput an image using a compute shader
fn test_image_compute(device: &Arc<Device>, queue: &Arc<Queue>) -> Result<()> {
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
                                    Some(queue.family()))?;

    // Build a command buffer that clears the image
    let command_buffer =
        AutoCommandBufferBuilder::new(device.clone(), queue.family())?
                                 .clear_color_image(
                                    init,
                                    ClearValue::Float([0.0, 0.0, 0.0, 1.0])
                                 )?
                                 .build()?;

    // Start running the clear command
    let clear_finished =
        command_buffer.execute(queue.clone())?;

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
    let shader = cs::Shader::load(device.clone())?;

    // Set up a compute pipeline containing that shader
    let pipeline =
        Arc::new(ComputePipeline::new(device.clone(),
                                      &shader.main_entry_point(),
                                      &())?);

    // Build a descriptor set to attach that pipeline to our buffer
    let descriptor_set = Arc::new(
        PersistentDescriptorSet::start(pipeline.clone(), 0)
                                .add_image(image.clone())?
                                .build()?
    );

    // Create a buffer to copy the final image contents in
    let buf =
        CpuAccessibleBuffer::from_iter(device.clone(),
                                       BufferUsage {
                                          transfer_destination: true,
                                          .. BufferUsage::none()
                                       },
                                       (0 .. 1024 * 1024 *4).map(|_| 0u8))?;

    // Build a command buffer that runs the computation
    let command_buffer =
        AutoCommandBufferBuilder::new(device.clone(), queue.family())?
                                 .dispatch([1024 / 8, 1024 / 8, 1],
                                           pipeline.clone(),
                                           descriptor_set.clone(),
                                           ())?
                                 .copy_image_to_buffer(image.clone(),
                                                       buf.clone())?
                                 .build()?;

    // Schedule the computation and wait for it
    command_buffer.execute_after(clear_finished, queue.clone())?
                  .then_signal_fence_and_flush()?
                  .wait(None)?;

    // Extract the image data from the buffer where it's been copied
    let buf_content = buf.read()?;
    let image =
        ImageBuffer::<Rgba<u8>, _>::from_raw(1024,
                                             1024,
                                             &buf_content[..])
                    .ok_or(failure::err_msg("Container is not big enough"))?;

    // Save the image to a PNG file
    Ok(image.save("mandelbrot.png")?)
}

// Draw a triangle. How hard could that get?
fn test_triangle(device: &Arc<Device>, queue: &Arc<Queue>) -> Result<()> {
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
                                                vertices.iter().cloned())?;

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
    let vs = vs::Shader::load(device.clone())?;

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
    let fs = fs::Shader::load(device.clone())?;

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
        )?
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
                         .render_pass(
                            Subpass::from(render_pass.clone(), 0)
                                    .ok_or(failure::err_msg("No such subpass"))?
                         )
                         // - Here is the target device, now build the pipeline!
                         .build(device.clone())?
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
                                    })?;

    // Create a buffer to copy the final image contents in
    let buf =
        CpuAccessibleBuffer::from_iter(device.clone(),
                                       BufferUsage {
                                          transfer_destination: true,
                                          .. BufferUsage::none()
                                       },
                                       (0 .. 1024 * 1024 *4).map(|_| 0u8))?;

    // A renderpass must be attached to its drawing target(s) via a framebuffer
    let framebuffer = Arc::new(
        Framebuffer::start(render_pass.clone())
                    .add(image.clone())?
                    .build()?
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
                                                          queue.family())?
                                 .begin_render_pass(framebuffer.clone(),
                                                    false,
                                                    clear_values)?
                                 .draw(pipeline.clone(),
                                       dynamic_state,
                                       vx_buf.clone(),
                                       (),
                                       ())?
                                 .end_render_pass()?
                                 .copy_image_to_buffer(image.clone(),
                                                       buf.clone())?
                                 .build()?;

    // The rest is business as usual: run the computation...
    command_buffer.execute(queue.clone())?
                  .then_signal_fence_and_flush()?
                  .wait(None)?;

    // ...and save it to disk. Phew!
    let content = buf.read()?;
    Ok(ImageBuffer::<Rgba<u8>, _>::from_raw(1024,
                                            1024,
                                            &content[..])
                   .ok_or(failure::err_msg("Container is not big enough"))?
                   .save("mighty_triangle.png")?)
}

// Application entry point
fn main() -> Result<()> {
    // This is a Vulkan test program
    println!("Hello and welcome to this Vulkan test program!");

    // Set up our Vulkan instance
    println!("* Setting up Vulkan instance...");
    env_logger::init();
    let easy_vulkano = EasyVulkano::new(
        Some(&app_info_from_cargo_toml!()),
        &InstanceExtensions::none(),
        &["VK_LAYER_LUNARG_standard_validation"]
    )?;

    // Select which physical device we are going to use
    println!("* Selecting physical device...");
    let features = Features {
        robust_buffer_access: true,
        .. Features::none()
    };
    let extensions = DeviceExtensions::none();
    let phys_device = easy_vulkano.select_physical_device(
        |dev| device_filter(dev, &features, &extensions),
        device_preference
    )?.ok_or(failure::err_msg("No suitable physical device found"))?;

    // Set up our logical device and command queue
    println!("* Setting up logical device and queue...");
    let (device, queue) = easy_vulkano.setup_single_queue_device(
        phys_device,
        &features,
        &extensions,
        queue_filter,
        queue_preference
    )?.expect("Our physical device filter should ensure success here");

    // Let's play a bit with vulkano's buffer abstraction
    println!("* Manipulating a CPU-accessible buffer...");
    test_buffer_read_write(&device)?;
    println!("* Making the GPU copy buffers for us...");
    test_buffer_copy(&device, &queue)?;
    println!("* Filling up a buffer using a compute shader...");
    test_buffer_compute(&device, &queue)?;

    // And then let's play with the image abstraction too!
    println!("* Drawing our first image...");
    test_image_basics(&device, &queue)?;
    println!("* Drawing a fractal with a compute shader...");
    test_image_compute(&device, &queue)?;

    // Finally, we can achieve the pinnacle of any modern graphics API, namely
    // drawing a colored triangle. Everything goes downhill from there.
    println!("* Drawing a triangle with the full graphics pipeline...");
    test_triangle(&device, &queue)?;

    // ...and then everything will be teared down automagically
    println!("We're done!");
    Ok(())
}