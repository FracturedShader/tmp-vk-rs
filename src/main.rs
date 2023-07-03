use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::Arc;

use vulkano::device::{self, Device, DeviceCreateInfo, QueueCreateInfo, QueueFlags};
use vulkano::{
    device::physical::{PhysicalDevice, PhysicalDeviceType},
    instance::{debug::DebugUtilsMessenger, Instance, InstanceExtensions},
    swapchain::Surface,
    VulkanLibrary,
};
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

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct QueueFamilyIndicesPartial {
    graphics: Option<u32>,
    present: Option<u32>,
}

impl QueueFamilyIndicesPartial {
    fn is_complete(&self) -> bool {
        self.graphics.is_some() && self.present.is_some()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct QueueFamilyIndices {
    graphics: u32,
    present: u32,
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
                return Self::try_build_partial(partial);
            }
        }

        None
    }

    fn try_build_partial(partial: QueueFamilyIndicesPartial) -> Option<Self> {
        if let QueueFamilyIndicesPartial {
            graphics: Some(graphics),
            present: Some(present),
        } = partial
        {
            Some(QueueFamilyIndices { graphics, present })
        } else {
            None
        }
    }

    fn unique_create_infos(&self) -> (Vec<QueueCreateInfo>, HashMap<u32, usize>) {
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

#[derive(Clone, Debug, PartialEq, Eq)]
struct DeviceQueues {
    graphics: Arc<device::Queue>,
    present: Arc<device::Queue>,
}

impl DeviceQueues {
    pub fn new(
        (device, indices): &VulkanDevice,
        enabled_extensions: device::DeviceExtensions,
    ) -> Result<(Arc<Device>, Self), device::DeviceCreationError> {
        let (create_infos, index_map) = indices.unique_create_infos();

        Device::new(
            device.clone(),
            DeviceCreateInfo {
                queue_create_infos: create_infos,
                enabled_extensions,
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

        let selected_device = Self::find_suitable_device(&instance, &surface, &needed_exts)
            .expect("It should be possible to enumerate physical devices")
            .expect("At least one suitable graphics device should exist");

        let (logical, queues) = DeviceQueues::new(&selected_device, needed_exts)
            .expect("It should be possible to create a logical device");

        Self {
            instance,
            _debug_callback: dc,
            surface,
            physical: selected_device.0,
            logical,
            queues,
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
            .filter_map(|device| Self::device_suitability(device, surface, needed_exts))
            .collect::<BinaryHeap<_>>();

        Ok(heap.pop().map(PriorityItem::into_inner))
    }

    fn device_suitability(
        device: Arc<PhysicalDevice>,
        surface: &Arc<Surface>,
        needed_exts: &device::DeviceExtensions,
    ) -> Option<PriorityItem<(u32, u32), VulkanDevice>> {
        let props = device.properties();

        let device_score = match props.device_type {
            PhysicalDeviceType::DiscreteGpu => 1000,
            PhysicalDeviceType::IntegratedGpu | PhysicalDeviceType::VirtualGpu => 500,
            PhysicalDeviceType::Cpu | PhysicalDeviceType::Other => 100,
            _ => 0,
        };

        let score = device_score;

        if !device.supported_extensions().contains(needed_exts) {
            return None;
        }

        if let Some(queues) = QueueFamilyIndices::try_build(&device, surface) {
            return Some(PriorityItem::new((score, device_score), (device, queues)));
        }

        None
    }
}

struct VulkanApp {
    event_loop: EventLoop<()>,
    _window: Arc<winit::window::Window>,
    vk_ctx: VulkanAppContext,
}

impl VulkanApp {
    pub fn new() -> Self {
        use winit::dpi::LogicalSize;

        let event_loop = EventLoop::new();
        let window = WindowBuilder::new()
            .with_title("Ten Minute Physics - Vulkan")
            .with_inner_size(LogicalSize::new(WIN_WIDTH, WIN_HEIGHT))
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
