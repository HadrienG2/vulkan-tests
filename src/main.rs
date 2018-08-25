#[macro_use] extern crate failure;
#[macro_use] extern crate vulkano;
#[macro_use] extern crate vulkano_shader_derive;

extern crate env_logger;
extern crate image;
extern crate vulkanoob;

mod examples;

use vulkanoob::{
    instance::EasyInstance,
    Result,
};

use vulkano::instance::{
    DeviceExtensions,
    Features,
    InstanceExtensions,
    PhysicalDeviceType,
};


// Application entry point
fn main() -> Result<()> {
    // This is a Vulkan test program
    println!("Hello and welcome to this Vulkan test program!");

    // Set up our Vulkan instance
    println!("* Setting up Vulkan instance...");
    env_logger::init();
    let instance = EasyInstance::new(
        Some(&app_info_from_cargo_toml!()),
        &InstanceExtensions::none(),
        vec!["VK_LAYER_LUNARG_standard_validation"]
    )?;

    // Select which physical device we are going to use
    println!("* Selecting physical device...");
    let features = Features {
        robust_buffer_access: true,
        .. Features::none()
    };
    let extensions = DeviceExtensions::none();
    let phys_device = instance.select_physical_device(
        |dev| examples::device_filter(dev, &features, &extensions),
        examples::device_preference
    )?.ok_or(failure::err_msg("No suitable physical device found"))?;

    // Set up our logical device and command queue
    println!("* Setting up logical device and queue...");
    let (device, queue) = phys_device.setup_single_queue_device(
        &features,
        &extensions,
        examples::queue_filter,
        examples::queue_preference
    )?.expect("Our physical device filter should ensure success here");

    // Let's play a bit with vulkano's buffer abstraction
    println!("* Manipulating a CPU-accessible buffer...");
    examples::buffers::read_write(&device)?;
    println!("* Making the GPU copy buffers for us...");
    examples::buffers::copy(&device, &queue)?;
    println!("* Filling up a buffer using a compute shader...");
    examples::buffers::compute(&device, &queue)?;

    // And then let's play with the image abstraction too!
    println!("* Drawing our first image...");
    examples::images::basics(&device, &queue)?;
    println!("* Drawing a fractal with a compute shader...");
    examples::images::compute(&device, &queue)?;

    // Finally, we can achieve the pinnacle of any modern graphics API, namely
    // drawing a colored triangle. Everything goes downhill from there.
    println!("* Drawing a triangle with the full graphics pipeline...");
    examples::graphics::triangle(&device, &queue)?;

    // ...and then everything will be teared down automagically
    println!("We're done!");
    Ok(())
}