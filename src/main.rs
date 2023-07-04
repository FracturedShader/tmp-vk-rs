use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::Arc;

use vulkano::device::physical::PhysicalDeviceError;
use vulkano::device::{self, Device, DeviceCreateInfo, QueueCreateInfo, QueueFlags};
use vulkano::image::view::ImageView;
use vulkano::image::SwapchainImage;
use vulkano::swapchain::{Swapchain, SwapchainCreateInfo};
use vulkano::{
    device::physical::{PhysicalDevice, PhysicalDeviceType},
    instance::{debug::DebugUtilsMessenger, Instance, InstanceExtensions},
    swapchain::Surface,
    VulkanLibrary,
};
use winit::dpi::LogicalSize;
use winit::{
    event_loop::EventLoop,
    window::{Window, WindowBuilder},
};

const WIN_WIDTH: u32 = 1280;
const WIN_HEIGHT: u32 = 720;

#[derive(Debug, PartialEq, Eq)]
struct PriorityItem<P, T>(P, T);

impl<P, T> PriorityItem<P, T> {
    pub fn new(priority: P, item: T) -> Self {
        Self(priority, item)
    }

    pub fn into_inner(self) -> T {
        self.1
    }
}

impl<P, T> PartialOrd for PriorityItem<P, T>
where
    P: PartialOrd + PartialEq,
    T: PartialEq,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<P, T> Ord for PriorityItem<P, T>
where
    P: Ord + Eq,
    T: Eq,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct QueueFamilyIndices {
    graphics: u32,
    present: u32,
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
struct QueueFamilyIndicesPartial {
    graphics: Option<u32>,
    present: Option<u32>,
}

impl QueueFamilyIndicesPartial {
    fn is_complete(&self) -> bool {
        self.graphics.is_some() && self.present.is_some()
    }

    fn try_build(self) -> Option<QueueFamilyIndices> {
        if let QueueFamilyIndicesPartial {
            graphics: Some(graphics),
            present: Some(present),
        } = self
        {
            Some(QueueFamilyIndices { graphics, present })
        } else {
            None
        }
    }
}

impl QueueFamilyIndices {
    pub fn try_build(device: &Arc<PhysicalDevice>, surface: &Arc<Surface>) -> Option<Self> {
        let qprops = device.queue_family_properties();
        let mut partial = QueueFamilyIndicesPartial::default();

        for (i, p) in qprops.iter().enumerate() {
            let i = u32::try_from(i).unwrap();

            if (p.queue_flags & QueueFlags::GRAPHICS) == QueueFlags::GRAPHICS {
                partial.graphics = Some(i);
            }

            if let Ok(supported) = device.surface_support(i, surface) {
                if supported {
                    partial.present = Some(i);
                }
            }

            if partial.is_complete() {
                return partial.try_build();
            }
        }

        None
    }

    fn sharing(self) -> vulkano::sync::Sharing<smallvec::SmallVec<[u32; 4]>> {
        use smallvec::smallvec;
        use vulkano::sync::Sharing;

        if self.graphics == self.present {
            Sharing::Exclusive
        } else {
            Sharing::Concurrent(smallvec![self.graphics, self.present])
        }
    }

    fn unique_create_infos(self) -> (Vec<QueueCreateInfo>, HashMap<u32, usize>) {
        let mut create_infos = Vec::new();
        let mut index_map = HashMap::new();

        for idx in [self.graphics, self.present] {
            index_map.entry(idx).or_insert_with(|| {
                let last = create_infos.len();

                create_infos.push(QueueCreateInfo {
                    queue_family_index: idx,
                    ..Default::default()
                });

                last
            });
        }

        (create_infos, index_map)
    }
}

type VulkanDevice = (Arc<PhysicalDevice>, QueueFamilyIndices);
type PrioritizedDevice = PriorityItem<(u32, u32), VulkanDevice>;

#[derive(Clone, Debug, PartialEq, Eq)]
struct DeviceQueues {
    graphics: Arc<device::Queue>,
    present: Arc<device::Queue>,
}

impl DeviceQueues {
    pub fn new(
        indices: QueueFamilyIndices,
        device: &Arc<PhysicalDevice>,
        enabled_extensions: &device::DeviceExtensions,
    ) -> Result<(Arc<Device>, Self), device::DeviceCreationError> {
        let (create_infos, index_map) = indices.unique_create_infos();

        Device::new(
            device.clone(),
            DeviceCreateInfo {
                queue_create_infos: create_infos,
                enabled_extensions: *enabled_extensions,
                ..Default::default()
            },
        )
        .map(|(d, qs)| {
            let qs = qs.collect::<Vec<_>>();

            (
                d,
                Self {
                    graphics: qs[index_map[&indices.graphics]].clone(),
                    present: qs[index_map[&indices.present]].clone(),
                },
            )
        })
    }
}

impl FromIterator<Arc<device::Queue>> for DeviceQueues {
    fn from_iter<T: IntoIterator<Item = Arc<device::Queue>>>(iter: T) -> Self {
        fn inner<I: Iterator<Item = Arc<device::Queue>>>(mut iter: I) -> DeviceQueues {
            DeviceQueues {
                graphics: iter.next().expect("The graphics queue should exist"),
                present: iter.next().expect("The present queue should exist"),
            }
        }

        inner(iter.into_iter())
    }
}

#[derive(Debug)]
struct VulkanAppContext {
    instance: Arc<Instance>,
    _debug_callback: Option<DebugUtilsMessenger>,
    surface: Arc<Surface>,
    physical: Arc<PhysicalDevice>,
    logical: Arc<Device>,
    queues: DeviceQueues,
    swapchain: Arc<Swapchain>,
    swapchain_format: vulkano::format::Format,
    swapchain_extent: [u32; 2],
    swapchain_images: Vec<Arc<SwapchainImage>>,
    swapchain_image_views: Vec<Arc<ImageView<SwapchainImage>>>,
    _window: Arc<Window>,
}

impl VulkanAppContext {
    pub fn new(window: Arc<winit::window::Window>) -> Self {
        let instance = Self::create_instance();
        let dc = Self::attach_debug_callback(&instance);
        let surface = vulkano_win::create_surface_from_winit(window.clone(), instance.clone())
            .expect("It should be possible to create a Vulkan surface for a window");

        let needed_exts = device::DeviceExtensions {
            khr_swapchain: true,
            ..Default::default()
        };

        let (physical, indices) = Self::find_suitable_device(&instance, &surface, &needed_exts)
            .expect("It should be possible to enumerate physical devices")
            .expect("At least one suitable graphics device should exist");

        let (logical, queues) = DeviceQueues::new(indices, &physical, &needed_exts)
            .expect("It should be possible to create a logical device");

        let create_info = Self::define_swapchain(
            &surface,
            &physical,
            indices,
            window.inner_size().to_logical(window.scale_factor()),
        )
        .expect("It should be possible to assemble swapchain details");

        let swapchain_format = create_info.image_format.unwrap();
        let swapchain_extent = create_info.image_extent;

        let (swapchain, swapchain_images) =
            Swapchain::new(logical.clone(), surface.clone(), create_info)
                .expect("Creating a simple swapchain should be possible");

        let swapchain_image_views = swapchain_images
            .iter()
            .cloned()
            .map(ImageView::new_default)
            .collect::<Result<Vec<_>, _>>()
            .expect("It should be possible to create direct views of the swapchain images");

        Self {
            instance,
            _debug_callback: dc,
            surface,
            physical,
            logical,
            queues,
            swapchain,
            swapchain_format,
            swapchain_extent,
            swapchain_images,
            swapchain_image_views,
            _window: window,
        }
    }
}

impl VulkanAppContext {
    #[cfg(not(debug_assertions))]
    fn attach_debug_callback(instance: &Arc<Instance>) -> Option<DebugUtilsMessenger> {
        None
    }

    #[cfg(debug_assertions)]
    fn attach_debug_callback(instance: &Arc<Instance>) -> Option<DebugUtilsMessenger> {
        use vulkano::instance::debug::{
            DebugUtilsMessageSeverity, DebugUtilsMessageType, DebugUtilsMessengerCreateInfo,
        };

        if instance.enabled_extensions().ext_debug_utils {
            let mut create_info = DebugUtilsMessengerCreateInfo::user_callback(Arc::new(|msg| {
                println!("[{:?}::{:?}] {}", msg.severity, msg.ty, msg.description);
            }));

            create_info.message_severity = DebugUtilsMessageSeverity::VERBOSE
                | DebugUtilsMessageSeverity::INFO
                | DebugUtilsMessageSeverity::WARNING
                | DebugUtilsMessageSeverity::ERROR;

            create_info.message_type = DebugUtilsMessageType::GENERAL
                | DebugUtilsMessageType::VALIDATION
                | DebugUtilsMessageType::PERFORMANCE;

            println!("Registering Vulkan debug callback.");

            // Safety: Callback does not make calls into the Vulkan API
            unsafe {
                Some(
                    DebugUtilsMessenger::new(instance.clone(), create_info)
                        .expect("Requirements for the debug message handler should be met"),
                )
            }
        } else {
            println!("Refusing to register Vulkan debug callback without ext_debug_utils.");

            None
        }
    }

    fn create_instance() -> Arc<Instance> {
        use vulkano::instance::InstanceCreateInfo;

        let library = VulkanLibrary::new().expect("Vulkan library should exist");
        let mut enabled_extensions = vulkano_win::required_extensions(&library);

        let vk_layers = Self::desired_layers(&mut enabled_extensions, &library);

        #[cfg(debug_assertions)]
        {
            println!("Loading layers:\n{vk_layers:#?}");
        }

        Instance::new(
            library,
            InstanceCreateInfo {
                application_name: Some("Ten Minute Physics - Vulkan".into()),
                application_version: vulkano::Version {
                    major: 0,
                    minor: 0,
                    patch: 0,
                },
                enabled_extensions,
                engine_name: Some("No Engine".into()),
                engine_version: vulkano::Version {
                    major: 0,
                    minor: 0,
                    patch: 0,
                },
                enabled_layers: vk_layers,
                ..Default::default()
            },
        )
        .expect("It should be possible to create a basic Vulkan instance")
    }

    fn define_swapchain(
        surface: &Arc<Surface>,
        device: &Arc<PhysicalDevice>,
        queues: QueueFamilyIndices,
        client_area: LogicalSize<u32>,
    ) -> Result<SwapchainCreateInfo, PhysicalDeviceError> {
        use vulkano::format::Format;
        use vulkano::image::ImageUsage;
        use vulkano::swapchain::{ColorSpace, PresentMode, SurfaceInfo};

        // This application currently doesn't care about full-screen mode
        let surface_info = SurfaceInfo::default();
        let formats = device.surface_formats(surface, surface_info.clone())?;
        let modes = device.surface_present_modes(surface)?.collect::<Vec<_>>();

        let &(image_format, image_color_space) = formats
            .iter()
            .find(|&(f, cs)| *f == Format::B8G8R8A8_SRGB && *cs == ColorSpace::SrgbNonLinear)
            .unwrap_or(&formats[0]);

        let present_mode = modes
            .into_iter()
            .find(|&m| m == PresentMode::Mailbox)
            .unwrap_or(PresentMode::Fifo);

        let caps = device.surface_capabilities(surface, surface_info)?;

        let mut image_extent = caps
            .current_extent
            .unwrap_or([client_area.width, client_area.height]);

        for ((&vmin, &vmax), v) in caps
            .min_image_extent
            .iter()
            .zip(caps.max_image_extent.iter())
            .zip(image_extent.iter_mut())
        {
            *v = Ord::clamp(*v, vmin, vmax);
        }

        let mut min_image_count = caps.min_image_count + 1;

        if let Some(maximum) = caps.max_image_count {
            min_image_count = min_image_count.min(maximum);
        }

        Ok(SwapchainCreateInfo {
            min_image_count,
            image_format: Some(image_format),
            image_color_space,
            image_extent,
            image_usage: ImageUsage::COLOR_ATTACHMENT,
            image_sharing: queues.sharing(),
            pre_transform: caps.current_transform,
            present_mode,
            ..Default::default()
        })
    }

    #[cfg(not(debug_assertions))]
    fn desired_layers(extensions: &mut InstanceExtensions, library: &VulkanLibrary) -> Vec<String> {
        Vec::new()
    }

    #[cfg(debug_assertions)]
    fn desired_layers(extensions: &mut InstanceExtensions, library: &VulkanLibrary) -> Vec<String> {
        let available_layers: HashSet<String> = library
            .layer_properties()
            .expect("There should be enough memory to enumerate available layers")
            .map(|p| p.name().into())
            .collect();

        [("VK_LAYER_KHRONOS_validation", || {
            extensions.ext_debug_utils = true;
        })]
        .iter_mut()
        .filter_map(|v| {
            if let Some(name) = available_layers.get(v.0) {
                v.1();
                Some(name.clone())
            } else {
                None
            }
        })
        .collect()
    }

    fn find_suitable_device(
        instance: &Arc<Instance>,
        surface: &Arc<Surface>,
        needed_exts: &device::DeviceExtensions,
    ) -> Result<Option<VulkanDevice>, vulkano::VulkanError> {
        let mut heap = instance
            .enumerate_physical_devices()?
            .filter_map(|device| {
                Self::device_suitability(device, surface, needed_exts)
                    .ok()
                    .flatten()
            })
            .collect::<BinaryHeap<_>>();

        Ok(heap.pop().map(PriorityItem::into_inner))
    }

    fn device_suitability(
        device: Arc<PhysicalDevice>,
        surface: &Arc<Surface>,
        needed_exts: &device::DeviceExtensions,
    ) -> Result<Option<PrioritizedDevice>, PhysicalDeviceError> {
        use vulkano::swapchain::SurfaceInfo;

        if !device.supported_extensions().contains(needed_exts) {
            return Ok(None);
        }

        // This application currently doesn't care about full-screen mode
        let surface_info = SurfaceInfo::default();
        let formats = device.surface_formats(surface, surface_info)?;
        let modes = device.surface_present_modes(surface)?.collect::<Vec<_>>();

        if formats.is_empty() || modes.is_empty() {
            return Ok(None);
        }

        let props = device.properties();

        let device_score = match props.device_type {
            PhysicalDeviceType::DiscreteGpu => 1000,
            PhysicalDeviceType::IntegratedGpu | PhysicalDeviceType::VirtualGpu => 500,
            PhysicalDeviceType::Cpu | PhysicalDeviceType::Other => 100,
            _ => 0,
        };

        let score = device_score;

        Ok(QueueFamilyIndices::try_build(&device, surface)
            .map(|queues| PriorityItem::new((score, device_score), (device, queues))))
    }
}

struct VulkanApp {
    event_loop: EventLoop<()>,
    _window: Arc<winit::window::Window>,
    vk_ctx: VulkanAppContext,
}

impl VulkanApp {
    pub fn new() -> Self {
        use winit::dpi::PhysicalSize;

        let event_loop = EventLoop::new();
        let window = WindowBuilder::new()
            .with_title("Ten Minute Physics - Vulkan")
            .with_inner_size(PhysicalSize::new(WIN_WIDTH, WIN_HEIGHT))
            .with_resizable(false)
            .build(&event_loop)
            .expect("It should be possible to create a simple window.");

        let window = Arc::new(window);
        let vk_ctx = VulkanAppContext::new(window.clone());

        Self {
            event_loop,
            _window: window,
            vk_ctx,
        }
    }

    pub fn run(self) {
        use winit::event::{Event, WindowEvent};

        self.event_loop
            .run(move |event, _window_target, control_flow| {
                control_flow.set_poll();

                match event {
                    Event::WindowEvent {
                        event: WindowEvent::CloseRequested,
                        ..
                    } => {
                        println!("Close button pressed; stopping");
                        control_flow.set_exit();
                    }
                    Event::MainEventsCleared => {
                        // Render Here
                    }
                    _ => (),
                }
            });
    }
}

fn main() {
    let app = VulkanApp::new();

    app.run();
}
