pub(crate) mod buffers;
pub(crate) mod images;
pub(crate) mod graphics;

use std::cmp::Ordering;

use vulkano::instance::{
    PhysicalDevice,
    QueueFamily,
};


// Dimensions of the buffers that we will manipulate
const STORAGE_BUF_SIZE: u32 = 65536;

// Dimensions of the images that we will manipulate
const IMG_WIDTH: u32 = 1024;
const IMG_HEIGHT: u32 = 1024;
const IMG_PIXELS: usize = (IMG_WIDTH as usize) * (IMG_HEIGHT as usize);

// Tells whether we can use a certain physical device or not
pub(crate) fn device_filter(dev: PhysicalDevice) -> bool {
    // We'll make 1024x1024 images
    let limits = dev.limits();
    if limits.max_image_dimension_2d() < IMG_WIDTH.max(IMG_HEIGHT) {
        return false;
    }

    // We'll use buffers of up to 65536 elements as shader outputs
    if limits.max_storage_buffer_range() < STORAGE_BUF_SIZE {
        return false;
    }

    // If control reaches this point, we can use this device
    true
}

// Tells whether we can use a certain queue family or not
pub(crate) fn queue_filter(family: &QueueFamily) -> bool {
    // For this learning exercise, we want at least a hybrid graphics + compute
    // queue (this implies data transfer support)
    family.supports_graphics() && family.supports_compute()
}

// Tells how acceptable device "dev1" compares to alternative "dev2"
pub(crate) fn device_preference(dev1: PhysicalDevice,
                                dev2: PhysicalDevice) -> Ordering {
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
pub(crate) fn queue_preference(_: &QueueFamily, _: &QueueFamily) -> Ordering {
    // Right now, we only intend to do graphics and compute, on a single queue,
    // without sparse binding magic, so any graphics- and compute-capable queue
    // family is the same by our standards.
    Ordering::Equal
}