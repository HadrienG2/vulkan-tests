use failure;

use image::{ImageBuffer, Rgba};

use std::sync::Arc;

use super::{IMG_HEIGHT, IMG_PIXELS, IMG_WIDTH};

use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, ImmutableBuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBuffer, DynamicState},
    device::{Device, Queue},
    format::Format,
    framebuffer::{Framebuffer, Subpass},
    image::{AttachmentImage, ImageUsage},
    pipeline::{
        GraphicsPipeline,
        viewport::Viewport,
    },
    sync::GpuFuture,
};

use vulkanoob::Result;


/// Let's draw a triangle. How hard could that get?
pub(crate) fn triangle(device: &Arc<Device>, queue: &Arc<Queue>) -> Result<()> {
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
    let (vx_buf, vx_future) =
        ImmutableBuffer::from_iter(vertices.iter().cloned(),
                                   BufferUsage {
                                       vertex_buffer: true,
                                       .. BufferUsage::none()
                                   },
                                   queue.clone())?;

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
                                    [IMG_WIDTH, IMG_HEIGHT],
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
                                       (0 .. IMG_PIXELS *4).map(|_| 0u8))?;

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
            dimensions: [IMG_WIDTH as f32, IMG_HEIGHT as f32],
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
                                       &dynamic_state,
                                       vx_buf.clone(),
                                       (),
                                       ())?
                                 .end_render_pass()?
                                 .copy_image_to_buffer(image.clone(),
                                                       buf.clone())?
                                 .build()?;

    // The rest is business as usual: run the computation...
    command_buffer.execute_after(vx_future, queue.clone())?
                  .then_signal_fence_and_flush()?
                  .wait(None)?;

    // ...and save it to disk. Phew!
    let content = buf.read()?;
    Ok(ImageBuffer::<Rgba<u8>, _>::from_raw(IMG_WIDTH,
                                            IMG_HEIGHT,
                                            &content[..])
                   .ok_or(failure::err_msg("Container is not big enough"))?
                   .save("mighty_triangle.png")?)
}