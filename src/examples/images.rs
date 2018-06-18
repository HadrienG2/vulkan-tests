use failure;

use image::{ImageBuffer, Rgba};

use std::sync::Arc;

use vulkano::{
    buffer::{
        BufferUsage,
        CpuAccessibleBuffer,
    },
    command_buffer::{
        AutoCommandBufferBuilder,
        CommandBuffer,
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
    image::{
        Dimensions,
        ImageLayout,
        ImageUsage,
        ImmutableImage,
        StorageImage,
    },
    pipeline::ComputePipeline,
    sync::GpuFuture,
};

use vulkanoob::Result;


// Let's perform some basic operations with an image
pub(crate) fn basics(device: &Arc<Device>, queue: &Arc<Queue>) -> Result<()> {
    // Create an image
    let (image, init) =
        ImmutableImage::uninitialized(device.clone(),
                                      Dimensions::Dim2d { width: 1024,
                                                          height: 1024 },
                                      Format::R8G8B8A8Unorm,
                                      1,
                                      ImageUsage {
                                         transfer_source: true,
                                         transfer_destination: true,
                                         storage: true,
                                         .. ImageUsage::none()
                                      },
                                      ImageLayout::TransferSrcOptimal,
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

// And now, let's compute an image using a compute shader
pub(crate) fn compute(device: &Arc<Device>, queue: &Arc<Queue>) -> Result<()> {
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

    // Schedule the clear command
    let clear_finished = command_buffer.execute(queue.clone())?;

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