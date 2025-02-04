#include "app.hpp"

namespace img_aligner
{

    static void glfw_error_callback(int error, const char* description);
    static void imgui_check_vk_result(VkResult err);

    void App::run()
    {
        try
        {
            init();
            main_loop();
            cleanup();
        }
        catch (const bv::Error& e)
        {
            throw std::runtime_error(e.to_string().c_str());
        }
    }

    void App::init()
    {
        init_window();
        init_context();
        setup_debug_messenger();
        create_surface();
        pick_physical_device();
        create_logical_device();
        create_memory_bank();
        create_imgui_descriptor_pool();
        init_imgui_vk_window_data();
        init_imgui();
    }

    void App::main_loop()
    {
        while (!glfwWindowShouldClose(state.window))
        {
            // poll and handle events (inputs, window resize, etc.)
            // you can read the io.WantCaptureMouse, io.WantCaptureKeyboard
            // flags to tell if Dear ImGui wants to use your inputs.
            // - when io.WantCaptureMouse is true, do not dispatch mouse input
            //   data to your main application, or clear/overwrite your copy of
            //   the mouse data.
            // - when io.WantCaptureKeyboard is true, do not dispatch keyboard
            //   input data to your main application, or clear/overwrite your
            //   copy of the keyboard data.
            // generally, you may always pass all inputs to Dear ImGui and hide
            // them from your application based on those two flags.
            glfwPollEvents();

            // resize swap chain if needed
            int fb_width, fb_height;
            glfwGetFramebufferSize(state.window, &fb_width, &fb_height);
            if (fb_width > 0 && fb_height > 0
                && (state.imgui_swapchain_rebuild
                    || state.imgui_vk_window_data.Width != fb_width
                    || state.imgui_vk_window_data.Height != fb_height))
            {
                ImGui_ImplVulkan_SetMinImageCount(
                    state.imgui_swapchain_min_image_count
                );
                ImGui_ImplVulkanH_CreateOrResizeWindow(
                    state.context->vk_instance(),
                    state.physical_device->handle(),
                    state.device->handle(),
                    &state.imgui_vk_window_data,
                    state.queue->queue_family_index(),
                    state.context->vk_allocator_ptr(),
                    fb_width,
                    fb_height,
                    state.imgui_swapchain_min_image_count
                );
                state.imgui_vk_window_data.FrameIndex = 0;
                state.imgui_swapchain_rebuild = false;
            }

            // sleep if window is iconified
            if (glfwGetWindowAttrib(state.window, GLFW_ICONIFIED) != 0)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            // update UI scale and reload fonts and style if needed
            if (state.ui_scale_updated)
            {
                state.ui_scale_updated = false;
                update_ui_scale_reload_fonts_and_style();
            }

            // start the Dear ImGui frame
            ImGui_ImplVulkan_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
            ImGui::DockSpaceOverViewport();

            // UI layout
            ImGui::PushFont(state.font);
            layout_image_viewer();
            layout_misc();
            layout_controls();
            ImGui::PopFont();

            // render
            ImGui::Render();
            ImDrawData* draw_data = ImGui::GetDrawData();
            const bool is_minimized =
                draw_data->DisplaySize.x <= 0.0f
                || draw_data->DisplaySize.y <= 0.0f;
            if (!is_minimized)
            {
                render_frame(draw_data);
                present_frame();
            }
        }
        state.device->wait_idle();
    }

    void App::cleanup()
    {
        state.device->wait_idle();

        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();

        // this destorys the surface as well
        ImGui_ImplVulkanH_DestroyWindow(
            state.context->vk_instance(),
            state.device->handle(),
            &state.imgui_vk_window_data,
            state.context->vk_allocator_ptr()
        );

        state.imgui_descriptor_pool = nullptr;

        state.transient_cmd_pool = nullptr;
        state.cmd_pool = nullptr;

        state.mem_bank = nullptr;
        state.queue = nullptr;
        state.device = nullptr;
        state.debug_messenger = nullptr;
        state.context = nullptr;

        glfwDestroyWindow(state.window);
        glfwTerminate();
    }

    void App::init_window()
    {
        glfwSetErrorCallback(glfw_error_callback);

        if (!glfwInit())
        {
            throw std::runtime_error("failed to initialize GLFW");
        }

        if (!glfwVulkanSupported())
        {
            throw std::runtime_error(
                "the Vulkan API isn't available. make sure you have installed "
                "the proper graphics drivers and that your graphics card "
                "supports Vulkan."
            );
        }

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

        state.window = glfwCreateWindow(
            INITIAL_WIDTH,
            INITIAL_HEIGHT,
            APP_TITLE,
            nullptr,
            nullptr
        );
        if (!state.window)
        {
            glfwTerminate();
            throw std::runtime_error("failed to create a window");
        }

        glfwSetWindowUserPointer(state.window, this);
    }

    void App::init_context()
    {
        std::vector<std::string> layers;
        if (DEBUG_MODE)
        {
            layers.push_back("VK_LAYER_KHRONOS_validation");
        }

        std::vector<std::string> extensions;
        {
            // extensions required by GLFW
            uint32_t glfw_ext_count = 0;
            const char** glfw_exts;
            glfw_exts = glfwGetRequiredInstanceExtensions(&glfw_ext_count);
            for (uint32_t i = 0; i < glfw_ext_count; i++)
            {
                extensions.emplace_back(glfw_exts[i]);
            }

            // debug utils extension
            if (DEBUG_MODE)
            {
                extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
            }
        }

        state.context = bv::Context::create({
            .will_enumerate_portability = false,
            .app_name = APP_TITLE,
            .app_version = bv::Version(1, 1, 0, 0),
            .engine_name = "no engine",
            .engine_version = bv::Version(1, 1, 0, 0),
            .vulkan_api_version = bv::VulkanApiVersion::Vulkan1_0,
            .layers = layers,
            .extensions = extensions
            });
    }

    void App::setup_debug_messenger()
    {
        if (!DEBUG_MODE)
        {
            return;
        }

        VkDebugUtilsMessageSeverityFlagsEXT severity_filter =
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
            | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;

        VkDebugUtilsMessageTypeFlagsEXT tpye_filter =
            VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
            | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
            | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT
            | VK_DEBUG_UTILS_MESSAGE_TYPE_DEVICE_ADDRESS_BINDING_BIT_EXT;

        state.debug_messenger = bv::DebugMessenger::create(
            state.context,
            severity_filter,
            tpye_filter,
            [](
                VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
                VkDebugUtilsMessageTypeFlagsEXT message_types,
                const bv::DebugMessageData& message_data
                )
            {
                std::cout << message_data.message << '\n';
            }
        );
    }

    void App::create_surface()
    {
        VkSurfaceKHR vk_surface;
        VkResult vk_result = glfwCreateWindowSurface(
            state.context->vk_instance(),
            state.window,
            state.context->vk_allocator_ptr(),
            &vk_surface
        );
        if (vk_result != VK_SUCCESS)
        {
            throw bv::Error(
                "failed to create window surface",
                vk_result,
                false
            );
        }
        state.surface = bv::Surface::create(state.context, vk_surface);
    }

    void App::pick_physical_device()
    {
        // make a list of devices we approve of
        auto all_physical_devices = state.context->fetch_physical_devices();
        std::vector<bv::PhysicalDevice> supported_physical_devices;
        for (const auto& pdev : all_physical_devices)
        {
            // make sure there's a queue family that supports graphics
            // operations and our window surface.
            if (pdev.find_queue_family_indices(
                VK_QUEUE_GRAPHICS_BIT,
                0,
                state.surface
            ).empty())
            {
                continue;
            }

            // make sure the device supports our window surface
            auto sc_support = pdev.fetch_swapchain_support(state.surface);
            if (!sc_support.has_value())
            {
                continue;
            }
            if (sc_support->present_modes.empty()
                || sc_support->surface_formats.empty())
            {
                continue;
            }

            // make sure the RGBA 32-bit float format is supported
            try
            {
                auto format_props = pdev.fetch_image_format_properties(
                    VK_FORMAT_R32G32B32A32_SFLOAT,
                    VK_IMAGE_TYPE_2D,
                    VK_IMAGE_TILING_OPTIMAL,

                    VK_IMAGE_USAGE_TRANSFER_DST_BIT
                    | VK_IMAGE_USAGE_SAMPLED_BIT,

                    0
                );
            }
            catch (const bv::Error&)
            {
                continue;
            }

            // make sure the desired level of multisampling is supported
            if (!(pdev.properties().limits.framebuffer_color_sample_counts
                & pdev.properties().limits.framebuffer_depth_sample_counts
                & MSAA_LEVEL))
            {
                continue;
            }

            supported_physical_devices.push_back(pdev);
        }
        if (supported_physical_devices.empty())
        {
            throw std::runtime_error("no supported physical devices");
        }

        std::cout << "pick a physical device by entering its index:\n";
        for (size_t i = 0; i < supported_physical_devices.size(); i++)
        {
            const auto& pdev = supported_physical_devices[i];

            std::string s_device_type = "unknown device type";
            switch (pdev.properties().device_type)
            {
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
                s_device_type = "integrated GPU";
                break;
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
                s_device_type = "discrete GPU";
                break;
            case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
                s_device_type = "virtual GPU";
                break;
            case VK_PHYSICAL_DEVICE_TYPE_CPU:
                s_device_type = "CPU";
                break;
            default:
                break;
            }

            std::cout << std::format(
                "{}: {} ({})\n",
                i,
                pdev.properties().device_name,
                s_device_type
            );
        }

        int32_t idx;
        while (true)
        {
            std::string s_idx;
            std::getline(std::cin, s_idx);
            try
            {
                idx = std::stoi(s_idx);
                if (idx < 0 || idx >= supported_physical_devices.size())
                {
                    throw std::exception();
                }
                break;
            }
            catch (const std::exception&)
            {
                std::cout << "enter a valid physical device index\n";
            }
        }
        std::cout << '\n';

        state.physical_device = supported_physical_devices[idx];

        glfwShowWindow(state.window);
    }

    void App::create_logical_device()
    {
        auto graphics_present_family_idx =
            state.physical_device->find_first_queue_family_index(
                VK_QUEUE_GRAPHICS_BIT,
                0,
                state.surface
            );

        std::vector<bv::QueueRequest> queue_requests;
        queue_requests.push_back(bv::QueueRequest{
            .flags = 0,
            .queue_family_index = graphics_present_family_idx,
            .num_queues_to_create = 1,
            .priorities = { 1.f }
            });

        bv::PhysicalDeviceFeatures enabled_features{};
        enabled_features.sampler_anisotropy = true;

        state.device = bv::Device::create(
            state.context,
            state.physical_device.value(),
            {
                .queue_requests = queue_requests,
                .extensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME },
                .enabled_features = enabled_features
            }
        );

        state.queue =
            bv::Device::retrieve_queue(
                state.device,
                graphics_present_family_idx,
                0
            );
    }

    void App::create_memory_bank()
    {
        state.mem_bank = bv::MemoryBank::create(state.device);
    }

    void App::create_imgui_descriptor_pool()
    {
        // the example only requires a single combined image sampler descriptor
        // for the font image and only uses one descriptor set (for that). if
        // you wish to load e.g. additional textures you may need to alter pools
        // sizes.

        std::vector<bv::DescriptorPoolSize> pool_sizes
        {
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 }
        };

        state.imgui_descriptor_pool = bv::DescriptorPool::create(
            state.device,
            {
                .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
                .max_sets = 1,
                .pool_sizes = pool_sizes
            }
        );
    }

    void App::init_imgui_vk_window_data()
    {
        state.imgui_vk_window_data.Surface = state.surface->handle();

        // make sure the physical device and the queue family support our window
        // surface
        auto sc_support = state.physical_device->fetch_swapchain_support(
            state.surface
        );
        if (!sc_support.has_value())
        {
            throw std::runtime_error("presentation not supported");
        }

        // choose a surface format
        bool found_surface_format = false;
        for (const auto& sfmt : sc_support->surface_formats)
        {
            if (sfmt.color_space == VK_COLORSPACE_SRGB_NONLINEAR_KHR)
            {
                state.imgui_vk_window_data.SurfaceFormat =
                    bv::SurfaceFormat_to_vk(sfmt);
                found_surface_format = true;
                break;
            }
        }
        if (!found_surface_format)
        {
            throw std::runtime_error("no supported surface format");
        }

        // choose present mode and minimum swapchain image count
        state.imgui_vk_window_data.PresentMode = VK_PRESENT_MODE_FIFO_KHR;
        state.imgui_swapchain_min_image_count =
            ImGui_ImplVulkanH_GetMinImageCountFromPresentMode(
                state.imgui_vk_window_data.PresentMode
            );

        // get window framebuffer size
        int fb_width, fb_height;
        glfwGetFramebufferSize(state.window, &fb_width, &fb_height);

        // create swapchain, render pass, framebuffer, etc.
        ImGui_ImplVulkanH_CreateOrResizeWindow(
            state.context->vk_instance(),
            state.physical_device->handle(),
            state.device->handle(),
            &state.imgui_vk_window_data,
            state.queue->queue_family_index(),
            state.context->vk_allocator_ptr(),
            fb_width,
            fb_height,
            state.imgui_swapchain_min_image_count
        );
    }

    void App::init_imgui()
    {
        // setup Dear ImGui context
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        state.io = &ImGui::GetIO();

        // enable keyboard and gamepad controls
        state.io->ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        state.io->ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;

        // enable docking
        state.io->ConfigFlags |= ImGuiConfigFlags_DockingEnable;

        // setup platform / renderer backends
        ImGui_ImplGlfw_InitForVulkan(state.window, true);
        ImGui_ImplVulkan_InitInfo init_info = {};
        init_info.Instance = state.context->vk_instance();
        init_info.PhysicalDevice = state.physical_device->handle();
        init_info.Device = state.device->handle();
        init_info.QueueFamily = state.queue->queue_family_index();
        init_info.Queue = state.queue->handle();
        init_info.PipelineCache = nullptr;
        init_info.DescriptorPool = state.imgui_descriptor_pool->handle();
        init_info.RenderPass = state.imgui_vk_window_data.RenderPass;
        init_info.Subpass = 0;
        init_info.MinImageCount = state.imgui_swapchain_min_image_count;
        init_info.ImageCount = state.imgui_vk_window_data.ImageCount;
        init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
        init_info.Allocator = state.context->vk_allocator_ptr();
        init_info.CheckVkResultFn = imgui_check_vk_result;
        ImGui_ImplVulkan_Init(&init_info);

        // load UI style and fonts
        update_ui_scale_reload_fonts_and_style();
    }

    void App::layout_image_viewer()
    {
        ImGui::Begin("Image Viewer", 0, ImGuiWindowFlags_HorizontalScrollbar);

        const uint32_t img_width = 3840;
        const uint32_t img_height = 2880;

        // image selector
        // TODO: use a combo
        ImGui::PushFont(state.font_bold);
        ImGui::Text("Base Image");
        ImGui::PopFont();

        // image size
        ImGui::Text("%dx%d", img_width, img_height);

        imgui_horiz_div();

        // zoom
        {
            ImGui::SameLine();
            ImGui::SetNextItemWidth(70.f * state.ui_scale);
            ImGui::DragFloat(
                "##image_zoom",
                &state.image_viewer_zoom,
                .005f,
                .1f,
                2.f,
                "%.2f",
                ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_NoRoundToFormat
            );

            ImGui::SameLine();
            if (ImGui::SmallButton("R##image_viewer"))
            {
                state.image_viewer_zoom = 1.f;
            }
        }

        // image
        if (0)
        {
            ImGui::Image(
                0,
                ImVec2(
                    (float)img_width * state.image_viewer_zoom,
                    (float)img_height * state.image_viewer_zoom
                ),
                { 0, 0 },
                { 1, 1 },
                { 1, 1, 1, 1 },
                COLOR_IMAGE_BORDER
            );
        }

        ImGui::End();
    }

    void App::layout_misc()
    {
        ImGui::Begin("Misc");

        imgui_bold("INTERFACE");

        if (ImGui::InputFloat(
            "Scale##Misc", &state.ui_scale, .125f, .25f, "%.3f"
        ))
        {
            state.ui_scale = std::clamp(state.ui_scale, .75f, 2.f);
            state.ui_scale_updated = true;
        }

        imgui_div();
        imgui_bold("INFO");

        // version
        ImGui::TextWrapped(std::format(
            "{} v{}",
            APP_TITLE,
            APP_VERSION
        ).c_str());

        // GitHub
        if (ImGui::Button("GitHub##Misc"))
        {
            open_url(APP_GITHUB_URL);
        }

        ImGui::End();
    }

    void App::layout_controls()
    {
        // TODO
    }

    void App::setup_imgui_style()
    {
        // img-aligner style from ImThemes
        ImGuiStyle& style = ImGui::GetStyle();

        style.Alpha = 1.0;
        style.DisabledAlpha = 1.0;
        style.WindowPadding = ImVec2(12.0, 12.0);
        style.WindowRounding = 4.0;
        style.WindowBorderSize = 0.0;
        style.WindowMinSize = ImVec2(20.0, 20.0);
        style.WindowTitleAlign = ImVec2(0.5, 0.5);
        style.WindowMenuButtonPosition = ImGuiDir_None;
        style.ChildRounding = 4.0;
        style.ChildBorderSize = 1.0;
        style.PopupRounding = 4.0;
        style.PopupBorderSize = 1.0;
        style.FramePadding = ImVec2(11.0, 6.0);
        style.FrameRounding = 3.0;
        style.FrameBorderSize = 1.0;
        style.ItemSpacing = ImVec2(12.0, 6.0);
        style.ItemInnerSpacing = ImVec2(6.0, 3.0);
        style.CellPadding = ImVec2(12.0, 6.0);
        style.IndentSpacing = 20.0;
        style.ColumnsMinSpacing = 6.0;
        style.ScrollbarSize = 12.0;
        style.ScrollbarRounding = 20.0;
        style.GrabMinSize = 28.0;
        style.GrabRounding = 20.0;
        style.TabRounding = 4.0;
        style.TabBorderSize = 1.0;
        style.TabMinWidthForCloseButton = 0.0;
        style.ColorButtonPosition = ImGuiDir_Right;
        style.ButtonTextAlign = ImVec2(0.5, 0.5);
        style.SelectableTextAlign = ImVec2(0.0, 0.0);

        style.Colors[ImGuiCol_Text] = ImVec4(1.0, 1.0, 1.0, 1.0);
        style.Colors[ImGuiCol_TextDisabled] = ImVec4(0.3625971674919128, 0.3366090059280396, 0.4470588266849518, 1.0);
        style.Colors[ImGuiCol_WindowBg] = ImVec4(0.08998643606901169, 0.0688353031873703, 0.1587982773780823, 1.0);
        style.Colors[ImGuiCol_ChildBg] = ImVec4(0.07058823853731155, 0.05098039284348488, 0.1294117718935013, 1.0);
        style.Colors[ImGuiCol_PopupBg] = ImVec4(0.1274803578853607, 0.1039600074291229, 0.2039215713739395, 1.0);
        style.Colors[ImGuiCol_Border] = ImVec4(1.0, 1.0, 1.0, 0.0313725508749485);
        style.Colors[ImGuiCol_BorderShadow] = ImVec4(0.0784313753247261, 0.08627451211214066, 0.1019607856869698, 0.0);
        style.Colors[ImGuiCol_FrameBg] = ImVec4(0.1323726028203964, 0.1103575527667999, 0.2039215713739395, 1.0);
        style.Colors[ImGuiCol_FrameBgHovered] = ImVec4(0.1686917990446091, 0.1392547339200974, 0.2575107216835022, 1.0);
        style.Colors[ImGuiCol_FrameBgActive] = ImVec4(0.22197425365448, 0.1358246803283691, 0.501960813999176, 1.0);
        style.Colors[ImGuiCol_TitleBg] = ImVec4(0.04083044454455376, 0.03529411926865578, 0.05882352963089943, 1.0);
        style.Colors[ImGuiCol_TitleBgActive] = ImVec4(0.04689064249396324, 0.02564054541289806, 0.1030042767524719, 1.0);
        style.Colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.04100684449076653, 0.03552479669451714, 0.05882352963089943, 1.0);
        style.Colors[ImGuiCol_MenuBarBg] = ImVec4(0.04689064249396324, 0.02564054541289806, 0.1030042767524719, 1.0);
        style.Colors[ImGuiCol_ScrollbarBg] = ImVec4(9.999899930335232e-07, 9.999932899518171e-07, 9.999999974752427e-07, 0.1759656667709351);
        style.Colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.9999899864196777, 0.9999949932098389, 1.0, 0.1072961091995239);
        style.Colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.9999899864196777, 0.9999949932098389, 1.0, 0.1459227204322815);
        style.Colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.9999899864196777, 0.9999949932098389, 1.0, 0.2403433322906494);
        style.Colors[ImGuiCol_CheckMark] = ImVec4(0.4847043752670288, 0.3261802792549133, 1.0, 1.0);
        style.Colors[ImGuiCol_SliderGrab] = ImVec4(0.555158257484436, 0.4077253341674805, 1.0, 0.540772557258606);
        style.Colors[ImGuiCol_SliderGrabActive] = ImVec4(0.7541811466217041, 0.6351931095123291, 1.0, 0.6313725709915161);
        style.Colors[ImGuiCol_Button] = ImVec4(0.1703480333089828, 0.1491580158472061, 0.239215686917305, 1.0);
        style.Colors[ImGuiCol_ButtonHovered] = ImVec4(0.2113431394100189, 0.1846674382686615, 0.2980392277240753, 1.0);
        style.Colors[ImGuiCol_ButtonActive] = ImVec4(0.3330428004264832, 0.2171164900064468, 0.7098039388656616, 1.0);
        style.Colors[ImGuiCol_Header] = ImVec4(0.6881198883056641, 0.5921568870544434, 1.0, 0.0470588244497776);
        style.Colors[ImGuiCol_HeaderHovered] = ImVec4(0.6791232824325562, 0.5803921222686768, 1.0, 0.08627451211214066);
        style.Colors[ImGuiCol_HeaderActive] = ImVec4(0.2269732505083084, 0.1357733607292175, 0.5021458864212036, 1.0);
        style.Colors[ImGuiCol_Separator] = ImVec4(0.1727046072483063, 0.1486197710037231, 0.250980406999588, 1.0);
        style.Colors[ImGuiCol_SeparatorHovered] = ImVec4(0.213542565703392, 0.1777155846357346, 0.2875536680221558, 1.0);
        style.Colors[ImGuiCol_SeparatorActive] = ImVec4(0.213542565703392, 0.1777155846357346, 0.2875536680221558, 1.0);
        style.Colors[ImGuiCol_ResizeGrip] = ImVec4(0.1703480333089828, 0.1491580158472061, 0.239215686917305, 1.0);
        style.Colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.2113431394100189, 0.1846674382686615, 0.2980392277240753, 1.0);
        style.Colors[ImGuiCol_ResizeGripActive] = ImVec4(0.3330428004264832, 0.2171164900064468, 0.7098039388656616, 1.0);
        style.Colors[ImGuiCol_Tab] = ImVec4(0.1514349430799484, 0.1328719705343246, 0.2117647081613541, 1.0);
        style.Colors[ImGuiCol_TabHovered] = ImVec4(0.2113431394100189, 0.1846674382686615, 0.2980392277240753, 1.0);
        style.Colors[ImGuiCol_TabActive] = ImVec4(0.3330428004264832, 0.2171164900064468, 0.7098039388656616, 1.0);
        style.Colors[ImGuiCol_TabUnfocused] = ImVec4(0.1514349430799484, 0.1328719705343246, 0.2117647081613541, 1.0);
        style.Colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.276118129491806, 0.1752556711435318, 0.6039215922355652, 1.0);
        style.Colors[ImGuiCol_PlotLines] = ImVec4(0.4254842102527618, 0.3235217034816742, 0.7568627595901489, 1.0);
        style.Colors[ImGuiCol_PlotLinesHovered] = ImVec4(0.831504225730896, 0.6952790021896362, 1.0, 1.0);
        style.Colors[ImGuiCol_PlotHistogram] = ImVec4(0.3840422034263611, 0.2874279022216797, 0.6980392336845398, 1.0);
        style.Colors[ImGuiCol_PlotHistogramHovered] = ImVec4(0.5856620073318481, 0.4077253341674805, 1.0, 1.0);
        style.Colors[ImGuiCol_TableHeaderBg] = ImVec4(0.5411763191223145, 0.3999999761581421, 1.0, 0.1803921610116959);
        style.Colors[ImGuiCol_TableBorderStrong] = ImVec4(9.999899930335232e-07, 9.999934036386549e-07, 9.999999974752427e-07, 0.1931330561637878);
        style.Colors[ImGuiCol_TableBorderLight] = ImVec4(1.0, 1.0, 1.0, 0.05098039284348488);
        style.Colors[ImGuiCol_TableRowBg] = ImVec4(0.08918920159339905, 0.06957323849201202, 0.1529411822557449, 1.0);
        style.Colors[ImGuiCol_TableRowBgAlt] = ImVec4(0.1216256022453308, 0.1011303439736366, 0.1882352977991104, 1.0);
        style.Colors[ImGuiCol_TextSelectedBg] = ImVec4(0.1882352977991104, 0.1019607856869698, 0.4588235318660736, 0.8627451062202454);
        style.Colors[ImGuiCol_DragDropTarget] = ImVec4(0.3330428004264832, 0.2171164900064468, 0.7098039388656616, 1.0);
        style.Colors[ImGuiCol_NavHighlight] = ImVec4(0.3330428004264832, 0.2171164900064468, 0.7098039388656616, 1.0);
        style.Colors[ImGuiCol_NavWindowingHighlight] = ImVec4(0.3330428004264832, 0.2171164900064468, 0.7098039388656616, 1.0);
        style.Colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.4392156898975372, 0.0, 0.0, 0.3294117748737335);
        style.Colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.0, 0.0, 0.0, 0.5490196347236633);
    }

    void App::update_ui_scale_reload_fonts_and_style()
    {
        // reload fonts
        state.io->Fonts->Clear();
        state.font = state.io->Fonts->AddFontFromFileTTF(
            FONT_PATH,
            FONT_SIZE * state.ui_scale
        );
        state.font_bold = state.io->Fonts->AddFontFromFileTTF(
            FONT_BOLD_PATH,
            FONT_SIZE * state.ui_scale
        );
        if (!state.font || !state.font_bold)
        {
            throw std::runtime_error("failed to load fonts");
        }
        state.io->Fonts->Build();
        ImGui_ImplVulkan_CreateFontsTexture();

        // reload style and apply scale
        setup_imgui_style();
        ImGui::GetStyle().ScaleAllSizes(state.ui_scale);
    }

    void App::imgui_div()
    {
        ImGui::NewLine();
    }

    void App::imgui_horiz_div()
    {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1.f, 1.f, 1.f, .3f), " | ");
    }

    void App::imgui_bold(std::string_view s)
    {
        ImGui::PushFont(state.font_bold);
        ImGui::Text(s.data());
        ImGui::PopFont();
    }

    bool App::imgui_combo(
        const std::string& label,
        const std::vector<std::string>& items,
        int* selected_idx,
        bool full_width
    )
    {
        static std::vector<std::string> active_combo_list;

        active_combo_list = items;
        bool result = false;

        if (full_width)
            ImGui::SetNextItemWidth(-1);

        return ImGui::Combo(
            label.c_str(),
            selected_idx,
            [](void* data, int index, const char** out_text)
            {
                if (index < 0 || index >= active_combo_list.size())
                {
                    return false;
                }

                *out_text = active_combo_list[index].c_str();
                return true;
            },
            nullptr,
            active_combo_list.size()
        );
    }

    void App::render_frame(ImDrawData* draw_data)
    {
        // get reference to the window data to make the code more readable
        auto& window_data = state.imgui_vk_window_data;

        // set background clear color
        window_data.ClearValue.color.float32[0] = COLOR_BG.x;
        window_data.ClearValue.color.float32[1] = COLOR_BG.y;
        window_data.ClearValue.color.float32[2] = COLOR_BG.z;
        window_data.ClearValue.color.float32[3] = COLOR_BG.w;

        // get semaphores

        VkSemaphore image_acquired_semaphore =
            window_data.FrameSemaphores[window_data.SemaphoreIndex]
            .ImageAcquiredSemaphore;

        VkSemaphore render_complete_semaphore =
            window_data.FrameSemaphores[window_data.SemaphoreIndex]
            .RenderCompleteSemaphore;

        // acquire next swapchain image and return if the swapchain needs to be
        // recreated in the next frame.
        VkResult vk_result = vkAcquireNextImageKHR(
            state.device->handle(),
            window_data.Swapchain,
            UINT64_MAX,
            image_acquired_semaphore,
            nullptr,
            &window_data.FrameIndex
        );
        if (vk_result == VK_ERROR_OUT_OF_DATE_KHR
            || vk_result == VK_SUBOPTIMAL_KHR)
        {
            state.imgui_swapchain_rebuild = true;
            return;
        }
        if (vk_result != VK_SUCCESS)
        {
            throw bv::Error(
                "failed to acquire next swapchain image",
                vk_result,
                false
            );
        }

        // get reference to the frame data to make the code more readable
        ImGui_ImplVulkanH_Frame& frame_data =
            window_data.Frames[window_data.FrameIndex];

        // wait for the current frame's fence and reset it
        {
            vk_result = vkWaitForFences(
                state.device->handle(),
                1,
                &frame_data.Fence,
                VK_TRUE,
                UINT64_MAX
            );
            if (vk_result != VK_SUCCESS)
            {
                throw bv::Error(
                    "failed to wait for fence",
                    vk_result,
                    false
                );
            }

            vk_result = vkResetFences(
                state.device->handle(),
                1,
                &frame_data.Fence
            );
            if (vk_result != VK_SUCCESS)
            {
                throw bv::Error(
                    "failed to reset fence",
                    vk_result,
                    false
                );
            }
        }

        // reset command pool
        vk_result = vkResetCommandPool(
            state.device->handle(),
            frame_data.CommandPool,
            0
        );
        if (vk_result != VK_SUCCESS)
        {
            throw bv::Error(
                "failed to reset command pool",
                vk_result,
                false
            );
        }

        // begin recording into the current frame's command buffer
        VkCommandBufferBeginInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vk_result = vkBeginCommandBuffer(frame_data.CommandBuffer, &info);
        if (vk_result != VK_SUCCESS)
        {
            throw bv::Error(
                "failed to begin recording command buffer",
                vk_result,
                false
            );
        }

        // add command to begin the render pass
        {
            VkRenderPassBeginInfo info{};
            info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            info.renderPass = window_data.RenderPass;
            info.framebuffer = frame_data.Framebuffer;
            info.renderArea.extent.width = window_data.Width;
            info.renderArea.extent.height = window_data.Height;
            info.clearValueCount = 1;
            info.pClearValues = &window_data.ClearValue;
            vkCmdBeginRenderPass(
                frame_data.CommandBuffer,
                &info,
                VK_SUBPASS_CONTENTS_INLINE
            );
        }

        // record Dear ImGui primitives into the command buffer
        ImGui_ImplVulkan_RenderDrawData(draw_data, frame_data.CommandBuffer);

        // add command to end the render pass
        vkCmdEndRenderPass(frame_data.CommandBuffer);

        // end recording the command buffer
        vk_result = vkEndCommandBuffer(frame_data.CommandBuffer);
        if (vk_result != VK_SUCCESS)
        {
            throw bv::Error(
                "failed to end recording command buffer",
                vk_result,
                false
            );
        }

        // submit command buffer
        {
            VkPipelineStageFlags wait_stage =
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

            VkSubmitInfo info = {};
            info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            info.waitSemaphoreCount = 1;
            info.pWaitSemaphores = &image_acquired_semaphore;
            info.pWaitDstStageMask = &wait_stage;
            info.commandBufferCount = 1;
            info.pCommandBuffers = &frame_data.CommandBuffer;
            info.signalSemaphoreCount = 1;
            info.pSignalSemaphores = &render_complete_semaphore;

            vk_result = vkQueueSubmit(
                state.queue->handle(),
                1,
                &info,
                frame_data.Fence
            );
            if (vk_result != VK_SUCCESS)
            {
                throw bv::Error(
                    "failed to submit command buffer",
                    vk_result,
                    false
                );
            }
        }
    }

    void App::present_frame()
    {
        if (state.imgui_swapchain_rebuild)
            return;

        // get reference to the window data to make the code more readable
        auto& window_data = state.imgui_vk_window_data;

        VkSemaphore render_complete_semaphore =
            window_data.FrameSemaphores[window_data.SemaphoreIndex]
            .RenderCompleteSemaphore;

        VkPresentInfoKHR info = {};
        info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        info.waitSemaphoreCount = 1;
        info.pWaitSemaphores = &render_complete_semaphore;
        info.swapchainCount = 1;
        info.pSwapchains = &window_data.Swapchain;
        info.pImageIndices = &window_data.FrameIndex;

        VkResult vk_result = vkQueuePresentKHR(state.queue->handle(), &info);
        if (vk_result == VK_ERROR_OUT_OF_DATE_KHR
            || vk_result == VK_SUBOPTIMAL_KHR)
        {
            state.imgui_swapchain_rebuild = true;
            return;
        }
        if (vk_result != VK_SUCCESS)
        {
            throw bv::Error(
                "failed to present frame",
                vk_result,
                false
            );
        }

        window_data.SemaphoreIndex =
            (window_data.SemaphoreIndex + 1) % window_data.SemaphoreCount;
    }

    static void glfw_error_callback(int error, const char* description)
    {
        std::cerr << std::format("GLFW error {}: {}\n", error, description);
    }

    static void imgui_check_vk_result(VkResult err)
    {
        if (err != VK_SUCCESS)
        {
            throw bv::Error("Dear ImGui Vulkan error", err, false);
        }
    }

}
