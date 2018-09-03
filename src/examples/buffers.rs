use failure;

use std::{
    mem,
    sync::Arc,
};

use super::STORAGE_BUF_SIZE;

use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBuffer},
    descriptor::descriptor_set::PersistentDescriptorSet,
    device::{Device, Queue},
    pipeline::ComputePipeline,
    sync::GpuFuture,
};

use vulkanoob::Result;


/// Try reading from and writing to vulkano's simplest buffer type
pub(crate) fn read_write(device: &Arc<Device>) -> Result<()> {
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

/// Try it again, but this time make the GPU perform the memory copy
pub(crate) fn copy(device: &Arc<Device>, queue: &Arc<Queue>) -> Result<()> {
    // Build a source and destination buffer
    let source = CpuAccessibleBuffer::from_iter(device.clone(),
                                                BufferUsage::transfer_source(),
                                                1..42)?;
    let dest =
        CpuAccessibleBuffer::from_iter(device.clone(),
                                       BufferUsage::transfer_destination(),
                                       (1..42).map(|_| 0))?;

    // We tell the GPU to make a copy using a command buffer
    let command_buffer =
        AutoCommandBufferBuilder::new(device.clone(), queue.family())?
                                 .copy_buffer(source.clone(), dest.clone())?
                                 .build()?;

    // Some sophisticated mechanisms are available to synchronize command
    // buffers with one another. For now, we'll just wait for completion.
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

/// Do a computation on a buffer using a compute pipeline
pub(crate) fn compute(device: &Arc<Device>, queue: &Arc<Queue>) -> Result<()> {
    // Here is some data
    let data_buffer =
        CpuAccessibleBuffer::from_iter(device.clone(),
                                       BufferUsage {
                                           storage_buffer: true,
                                           .. BufferUsage::none()
                                       },
                                       0..STORAGE_BUF_SIZE)?;

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
                                 .dispatch([STORAGE_BUF_SIZE/64, 1, 1],
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