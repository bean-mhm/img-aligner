#include "app.hpp"

#include "nfd.hpp"

#include <nlohmann/json.hpp>
using json = nlohmann::ordered_json;

namespace img_aligner
{

    static void glfw_error_callback(int error, const char* description);
    static void imgui_check_vk_result(VkResult err);

    float GridWarpOptimizationParams::calc_warp_strength(size_t n_iters)
    {
        // warp strength decay
        float decayed_warp_strength =
            warp_strength
            * std::exp(-warp_strength_decay_rate * (float)n_iters);

        // min warp strength
        decayed_warp_strength = std::max(
            decayed_warp_strength,
            min_warp_strength
        );

        return decayed_warp_strength;
    }

    const char* GridWarpOptimizationStopReason_to_str(
        GridWarpOptimizationStopReason reason
    )
    {
        switch (reason)
        {
        case GridWarpOptimizationStopReason::ManuallyStopped:
            return "ManuallyStopped";
        case GridWarpOptimizationStopReason::LowChangeInCost:
            return "LowChangeInCost";
        case GridWarpOptimizationStopReason::ReachedMaxIters:
            return "ReachedMaxIters";
        case GridWarpOptimizationStopReason::ReachedMaxRuntime:
            return "ReachedMaxRuntime";
        case GridWarpOptimizationStopReason::Error:
            return "Error";
        default:
            return "None";
        }
    }

    const char* GridWarpOptimizationStopReason_to_str_friendly(
        GridWarpOptimizationStopReason reason
    )
    {
        switch (reason)
        {
        case GridWarpOptimizationStopReason::ManuallyStopped:
            return "manually stopped";
        case GridWarpOptimizationStopReason::LowChangeInCost:
            return "low change in cost";
        case GridWarpOptimizationStopReason::ReachedMaxIters:
            return "reached maximum iterations";
        case GridWarpOptimizationStopReason::ReachedMaxRuntime:
            return "reached maximum run time";
        case GridWarpOptimizationStopReason::Error:
            return "an error occurred, check the console";
        default:
            return "none";
        }
    }

    App::App(int argc, char** argv)
        : argc(argc), argv(argv)
    {
        parse_command_line();
    }

    void App::run()
    {
        if (state.cli_mode)
        {
            // this will call init() if necessary
            handle_command_line();
        }
        else
        {
            init();
            main_loop();
        }
        cleanup();
    }

    void App::init()
    {
        init_was_called = true;

        ScopedTimer timer(
            !cli_params.flag_silent,
            "",
            "done initializing ({} s)\n"
        );

        if (state.cli_mode)
        {
            init_context();
            setup_debug_messenger();
            pick_physical_device();
            create_logical_device();
            create_memory_bank();
        }
        else
        {
            NFD_Init();
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
    }

    void App::main_loop()
    {
        // do nothing in command line mode
        if (state.cli_mode)
        {
            return;
        }

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
                    state.queue_main->queue_family_index(),
                    state.context->vk_allocator_ptr(),
                    fb_width,
                    fb_height,
                    state.imgui_swapchain_min_image_count
                );
                state.imgui_vk_window_data.FrameIndex = 0;
                state.imgui_swapchain_rebuild = false;

                // the maximum number of frames in flight could change, so we
                // recreate the UI pass just in case.
                if (ui_pass != nullptr)
                {
                    recreate_ui_pass();
                }
            }

            // sleep if window is iconified
            if (glfwGetWindowAttrib(state.window, GLFW_ICONIFIED) != 0)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            // update UI scale and reload fonts and style if needed
            if (ui_scale_updated)
            {
                ui_scale_updated = false;
                update_ui_scale_reload_fonts_and_style();
            }

            // start the Dear ImGui frame
            ImGui_ImplVulkan_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
            ImGui::DockSpaceOverViewport();

            // UI layout
            // the ordering of these functions matters because layout_controls()
            // might recreate the UI pass and consequently the descriptor sets
            // that are used in ImGui::Image() so layout_image_viewer() should
            // be called after layout_controls().
            ImGui::PushFont(font);
            layout_controls();
            layout_misc();
            layout_image_viewer();
            ImGui::PopFont();

            // update UI pass and make a new copy of the grid vertices when
            // optimizing at the specified interval.
            if (is_optimizing)
            {
                bool interval_reached =
                    elapsed_sec(last_ui_update_when_optimizing_time) >
                    GRID_WARP_OPTIMIZATION_UI_UPDATE_INTERVAL;

                if (interval_reached)
                {
                    need_to_run_ui_pass = true;

                    copy_grid_vertices_for_ui_preview();

                    last_ui_update_when_optimizing_time =
                        std::chrono::high_resolution_clock::now();
                }
            }

            // update UI pass' display image if needed
            if (need_to_run_ui_pass
                && ui_pass != nullptr
                && ui_pass->images().size() > 0
                && selected_image_idx < ui_pass->images().size())
            {
                need_to_run_ui_pass = false;

                if (is_optimizing)
                {
                    need_the_optimization_mutex = true;
                    optimization_mutex.lock_shared();

                    ui_pass->run(
                        ui_pass->images()[selected_image_idx],
                        image_viewer_exposure,
                        image_viewer_use_flim,
                        state.queue_main
                    );

                    optimization_mutex.unlock_shared();
                    need_the_optimization_mutex = false;
                    need_the_optimization_mutex.notify_all();
                }
                else
                {
                    ui_pass->run(
                        ui_pass->images()[selected_image_idx],
                        image_viewer_exposure,
                        image_viewer_use_flim,
                        state.queue_main
                    );
                }
            }

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
        if (!init_was_called)
        {
            return;
        }

        ScopedTimer timer(
            !cli_params.flag_silent,
            "",
            "done cleaning up ({} s)\n"
        );

        if (!state.cli_mode)
        {
            NFD_Quit();
        }

        if (is_optimizing)
        {
            stop_optimization();
        }

        ui_pass = nullptr;
        grid_warper = nullptr;

        base_img = nullptr;
        base_img_mem = nullptr;
        base_imgview = nullptr;

        target_img = nullptr;
        target_img_mem = nullptr;
        target_imgview = nullptr;

        if (state.device != nullptr)
        {
            state.device->wait_idle();
        }

        if (!state.cli_mode)
        {
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
        }

        state.cmd_pools.clear();
        state.transient_cmd_pools.clear();

        state.mem_bank = nullptr;

        state.queue_main = nullptr;
        state.queue_grid_warp_optimize = nullptr;

        state.device = nullptr;
        state.debug_messenger = nullptr;
        state.context = nullptr;

        if (!state.cli_mode)
        {
            glfwDestroyWindow(state.window);
            glfwTerminate();
        }
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
        if (ENABLE_VALIDATION_LAYER)
        {
            layers.push_back("VK_LAYER_KHRONOS_validation");
        }

        std::vector<std::string> extensions;

        // extensions required by GLFW
        if (!state.cli_mode)
        {
            uint32_t glfw_ext_count = 0;
            const char** glfw_exts;
            glfw_exts = glfwGetRequiredInstanceExtensions(&glfw_ext_count);
            for (uint32_t i = 0; i < glfw_ext_count; i++)
            {
                extensions.emplace_back(glfw_exts[i]);
            }
        }

        // debug utils extension
        if (ENABLE_VALIDATION_LAYER)
        {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
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
        if (!ENABLE_VALIDATION_LAYER)
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
                fprintln("Vulkan: {}", message_data.message);
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
            // make sure there's a queue family that supports at least 2 queues
            // with graphics operations and our window surface.
            if (pdev.find_queue_family_indices(
                VK_QUEUE_GRAPHICS_BIT,
                0,
                state.cli_mode ? nullptr : state.surface,
                2
            ).empty())
            {
                continue;
            }

            // make sure the device supports our window surface
            if (!state.cli_mode)
            {
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

            supported_physical_devices.push_back(pdev);
        }

        if (supported_physical_devices.empty())
        {
            throw std::runtime_error("no supported physical devices");
        }

        int32_t actual_pdev_idx = 0;

        // at first, we'll always pick a device automatically but we might
        // change it later. we do this to have an idea of which device
        // *would be* automatically chosen if it was automatic. after this
        // block, actual_pdev_idx will contain the index of the device that
        // would be chosen if the selection was automatic.
        {
            // pick the first device as a fallback
            actual_pdev_idx = 0;

            // pick the first discrete GPU if there's any
            for (int32_t i = 0; i < supported_physical_devices.size(); i++)
            {
                if (supported_physical_devices[i].properties().device_type
                    == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
                {
                    actual_pdev_idx = i;
                    break;
                }
            }
        }

        if (physical_device_idx == PHYSICAL_DEVICE_IDX_AUTO)
        {
            // we've already selected a device automatically so we'll do nothing
            // except print which device we selected.
            if (!cli_params.flag_silent)
            {
                const auto& pdev = supported_physical_devices[actual_pdev_idx];
                fprintln(
                    "automatically selected physical device {}: {} ({})",
                    actual_pdev_idx,
                    pdev.properties().device_name,
                    VkPhysicalDeviceType_to_str(pdev.properties().device_type)
                );
            }
        }
        else if (physical_device_idx == PHYSICAL_DEVICE_IDX_PROMPT)
        {
            println("pick a physical device by entering its index:");
            for (size_t i = 0; i < supported_physical_devices.size(); i++)
            {
                const auto& pdev = supported_physical_devices[i];

                fprint(
                    "{}: {} ({})",
                    i,
                    pdev.properties().device_name,
                    VkPhysicalDeviceType_to_str(pdev.properties().device_type)
                );

                // remember from above, actual_pdev_idx contains the index of
                // the device that *would be* chosen if the selection was
                // automatic.
                if (i == actual_pdev_idx)
                {
                    print(" (default)");
                }

                println();
            }

            while (true)
            {
                std::string s_idx;
                std::getline(std::cin, s_idx);
                try
                {
                    actual_pdev_idx = std::stoi(s_idx);
                    if (actual_pdev_idx < 0
                        || actual_pdev_idx >= supported_physical_devices.size())
                    {
                        throw std::exception();
                    }
                    break;
                }
                catch (const std::exception&)
                {
                    println("enter a valid physical device index");
                }
            }
            println();
        }
        else if (physical_device_idx < 0
            || physical_device_idx >= supported_physical_devices.size())
        {
            throw std::runtime_error("invalid physical device index");
        }
        else
        {
            actual_pdev_idx = physical_device_idx;
        }

        state.physical_device = supported_physical_devices[actual_pdev_idx];

        glfwShowWindow(state.window);
    }

    void App::create_logical_device()
    {
        auto graphics_present_family_idx =
            state.physical_device->find_first_queue_family_index(
                VK_QUEUE_GRAPHICS_BIT,
                0,
                state.cli_mode ? nullptr : state.surface,
                2
            );

        std::vector<bv::QueueRequest> queue_requests;
        queue_requests.push_back(bv::QueueRequest{
            .flags = 0,
            .queue_family_index = graphics_present_family_idx,
            .num_queues_to_create = 2,
            .priorities = { .8f, 1.f }
            });

        bv::PhysicalDeviceFeatures enabled_features{};

        std::vector<std::string> device_extensions;
        if (!state.cli_mode)
        {
            device_extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
        }

        state.device = bv::Device::create(
            state.context,
            state.physical_device.value(),
            {
                .queue_requests = queue_requests,
                .extensions = device_extensions,
                .enabled_features = enabled_features
            }
        );

        state.queue_main = bv::Device::retrieve_queue(
            state.device,
            graphics_present_family_idx,
            0
        );
        state.queue_grid_warp_optimize = bv::Device::retrieve_queue(
            state.device,
            graphics_present_family_idx,
            1
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
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 16 }
        };

        state.imgui_descriptor_pool = bv::DescriptorPool::create(
            state.device,
            {
                .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
                .max_sets = 16,
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
            state.queue_main->queue_family_index(),
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

        // path to imgui.ini. we want the string to live forever so we won't
        // delete it.
        auto ini_path = new std::string((exec_dir() / "imgui.ini").string());
        state.io->IniFilename = ini_path->c_str();

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
        init_info.QueueFamily = state.queue_main->queue_family_index();
        init_info.Queue = state.queue_main->handle();
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

    void App::parse_command_line()
    {
        cli_app = std::make_unique<CLI::App>(
            fmt::format("{} v{} ({})", APP_TITLE, APP_VERSION, APP_GITHUB_URL),
            APP_TITLE
        );
        argv = cli_app->ensure_utf8(argv);

        cli_app->add_flag(
            "--cli",
            state.cli_mode,
            "enable command line mode"
        );

        // CLI11's exception-reliant way of handling version and help is weird
        // so I'm handling it manually.
        cli_app->remove_option(cli_app->get_help_ptr());
        cli_app->add_flag(
            "-h,--help",
            cli_params.flag_help,
            "print this help message and exit"
        );

        // CLI11's exception-reliant way of handling version and help is weird
        // so I'm handling it manually.
        cli_app->remove_option(cli_app->get_version_ptr());
        cli_app->add_flag(
            "-v,--version",
            cli_params.flag_version,
            "print app version and exit"
        );

        cli_app->add_option(
            "-b,--base",
            cli_params.base_img_path,
            "input path to the base image file"
        );

        cli_app->add_option(
            "-t,--target",
            cli_params.target_img_path,
            "input path to the target image file"
        );

        cli_app->add_option(
            "-o,--output",
            cli_params.output_img_path,
            "optional output path to the warped image file"
        );

        cli_app->add_option(
            "-d,--diff0",
            cli_params.difference_img_before_opt_path,
            "optional output path to the difference image file exported before "
            "optimization"
        );

        cli_app->add_option(
            "-D,--diff1",
            cli_params.difference_img_after_opt_path,
            "optional output path to the difference image file exported after "
            "optimization"
        );

        cli_app->add_option(
            "-x,--base-mul",
            grid_warp_params.base_img_mul,
            "base image multiplier"
        )->capture_default_str();

        cli_app->add_option(
            "-y,--target-mul",
            grid_warp_params.target_img_mul,
            "target image multiplier"
        )->capture_default_str();

        cli_add_toggle(
            *cli_app,
            "-U,--undo-base-mul",
            export_warped_img_undo_base_img_mul,
            "toggle whether to apply the inverse of the base image multiplier "
            "when exporting the warped image"
        );

        cli_app->add_option(
            "-g,--grid-res",
            grid_warp_params.grid_res_area,
            "area of the grid resolution"
        )->capture_default_str();

        cli_app->add_option(
            "-p,--grid-padding",
            grid_warp_params.grid_padding,
            "grid padding"
        )->capture_default_str();

        cli_app->add_option(
            "-r,--interm-res",
            grid_warp_params.intermediate_res_area,
            "area of the intermediate resolution"
        )->capture_default_str();

        cli_app->add_option(
            "-c,--cost-res",
            grid_warp_params.cost_res_area,
            "area of the cost resolution"
        )->capture_default_str();

        cli_app->add_option(
            "-s,--seed",
            grid_warp_params.rng_seed,
            "seed number to use for pseudo-random number generators"
        )->capture_default_str();

        cli_app->add_option(
            "-X,--scalex",
            grid_transform.scale.x,
            "grid transform: scale in the X axis"
        );

        cli_app->add_option(
            "-Y,--scaley",
            grid_transform.scale.y,
            "grid transform: scale in the Y axis"
        );

        cli_app->add_option(
            "-R,--rot",
            grid_transform.rotation,
            "grid transform: rotation in degrees"
        );

        cli_app->add_option(
            "-Q,--offsx",
            grid_transform.offset.x,
            "grid transform: offset in the X axis"
        );

        cli_app->add_option(
            "-W,--offsy",
            grid_transform.offset.y,
            "grid transform: offset in the Y axis"
        );

        cli_app->add_option(
            "--scale-jitter",
            optimization_params.scale_jitter,
            "span of the range in which the scale of the current grid "
            "transform will be jittered. if this is 1.25, for example, the "
            "scale of the current grid transform will itself be scaled by a "
            "random value ranging from (1 / 1.25 = 0.8) to 1.25."
        )->capture_default_str();

        cli_app->add_option(
            "--rotation-jitter",
            optimization_params.rotation_jitter,
            "span of the range in which the rotation of the current grid "
            "transform will be jittered. if this is 2, for example, the "
            "rotation of the current grid transform will be offset by a random "
            "value ranging from -2 to 2."
        )->capture_default_str();

        cli_app->add_option(
            "--offset-jitter",
            optimization_params.offset_jitter,
            "span of the range in which the offset of the current grid "
            "transform will be jittered. if this is 0.01, for example, the "
            "offset of the current grid transform will itself be offset by a "
            "random vector in a square with a radius of 0.01 units (in "
            "normalized, zero-centered, aspect-ratio-adjusted UV space)."
        )->capture_default_str();

        cli_app->add_option(
            "--n-transform-iters",
            optimization_params.n_transform_optimization_iters,
            "number of transform optimization iterations. use 0 to disable "
            "transform optimization."
        )->capture_default_str();

        cli_app->add_option(
            "-w,--warp-strength",
            optimization_params.warp_strength
        )->capture_default_str();

        cli_app->add_option(
            "-Z,--warp-strength-decay",
            optimization_params.warp_strength_decay_rate,
            "warp strength will be scaled by e^(-di) where d is the decay rate "
            "and i is the number of iterations."
        )->capture_default_str();

        cli_app->add_option(
            "-C,--min-warp-strength",
            optimization_params.min_warp_strength,
            "minimum warp strength after decaying"
        )->capture_default_str();

        cli_app->add_option(
            "-m,--min-change-in-cost",
            optimization_params.min_change_in_cost_in_last_n_iters,
            fmt::format(
                "stop optimization if the cost decreased by less than this "
                "value in {} iterations",
                grid_warp::N_ITERS_TO_CHECK_CHANGE_IN_COST
            )
        )->capture_default_str();

        cli_app->add_option(
            "-i,--max-iters",
            optimization_params.max_iters,
            "maximum number of iterations (use 0 for no limit)"
        )->capture_default_str();

        cli_app->add_option(
            "-T,--max-runtime",
            optimization_params.max_runtime_sec,
            "maximum run time in seconds (use 0 for no limit)"
        )->capture_default_str();

        cli_app->add_flag(
            "-n,--silent",
            cli_params.flag_silent,
            "don't print log"
        );

        cli_app->add_option(
            "-S,--stat",
            cli_params.optimization_stats_mode,
            "0: don't print optimization statistics. 1: print statistics at "
            "the end. 2: print realtime statistics."
        )->check(CLI::Range(2, "[0 - 2]"))->capture_default_str();

        cli_add_toggle(
            *cli_app,
            "-P,--meta-params",
            metadata_export_options.params_and_res,
            "toggle whether to include parameters and resolutions when "
            "exporting metadata."
        );

        cli_add_toggle(
            *cli_app,
            "-O,--meta-opt",
            metadata_export_options.optimization_info,
            "toggle whether to include optimization parameters and statistics "
            "when exporting metadata."
        );

        cli_add_toggle(
            *cli_app,
            "-V,--meta-vert",
            metadata_export_options.grid_vertices,
            "toggle whether to include grid vertex data when exporting "
            "metadata."
        );

        cli_add_toggle(
            *cli_app,
            "-K,--meta-pretty",
            metadata_export_options.pretty_print,
            "toggle whether to produce pretty printed JSON when exporting "
            "metadata."
        );

        cli_app->add_option(
            "-M,--meta",
            cli_params.metadata_path,
            "optional output path to the metadata file (.json)"
        );

        cli_app->add_option(
            "-G,--gpu",
            physical_device_idx,
            "physical device index. use -1 to prompt the user to pick one or "
            "-2 to select one automatically."
        )->capture_default_str();

        cli_app->parse(argc, argv);
    }

    void App::handle_command_line()
    {
        // do nothing if not in command line mode
        if (!state.cli_mode)
        {
            return;
        }

        if (cli_params.flag_help || argc <= 2)
        {
            println(cli_app->help());
            return;
        }

        if (cli_params.flag_version)
        {
            println(APP_VERSION);
            return;
        }

        TimePoint time_start = std::chrono::high_resolution_clock::now();

        // we can call init() now
        init();

        if (cli_params.base_img_path.empty())
        {
            throw std::invalid_argument("base image path is required");
        }
        if (cli_params.target_img_path.empty())
        {
            throw std::invalid_argument("target image path is required");
        }

        try
        {
            ScopedTimer timer(!cli_params.flag_silent, "loading base image");
            load_image(
                state,
                cli_params.base_img_path,
                base_img,
                base_img_mem,
                base_imgview
            );
        }
        catch (const std::exception& e)
        {
            throw std::runtime_error(fmt::format(
                "failed to load base image: {}",
                e.what()
            ).c_str());
        }

        try
        {
            ScopedTimer timer(!cli_params.flag_silent, "loading target image");
            load_image(
                state,
                cli_params.target_img_path,
                target_img,
                target_img_mem,
                target_imgview
            );
        }
        catch (const std::exception& e)
        {
            throw std::runtime_error(fmt::format(
                "failed to load target image: {}",
                e.what()
            ).c_str());
        }

        try
        {
            ScopedTimer timer(!cli_params.flag_silent, "creating grid warper");
            recreate_grid_warper();
        }
        catch (const std::exception& e)
        {
            throw std::runtime_error(fmt::format(
                "failed to create grid warper: {}",
                e.what()
            ).c_str());
        }

        try
        {
            if (!cli_params.difference_img_before_opt_path.empty())
            {
                ScopedTimer timer(
                    !cli_params.flag_silent,
                    "exporting difference image before optimization"
                );
                save_image(
                    state,
                    grid_warper->get_difference_img(),
                    cli_params.difference_img_before_opt_path
                );
            }
        }
        catch (const std::exception& e)
        {
            throw std::runtime_error(fmt::format(
                "failed to save difference image before optimization: {}",
                e.what()
            ).c_str());
        }

        if (!cli_params.flag_silent)
        {
            println("starting optimization");
        }

        start_optimization();

        TimePoint last_time_print_stats;
        while (is_optimizing)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));

            // print realtime statistics at fixed intervals if enabled
            if (cli_params.optimization_stats_mode ==
                CliGridWarpOptimizationStatsMode::Realtime
                && elapsed_sec(last_time_print_stats) >
                GRID_WARP_OPTIMIZATION_CLI_REALTIME_STATS_INTERVAL)
            {
                print_optimization_statistics(true);
                last_time_print_stats =
                    std::chrono::high_resolution_clock::now();
            }
        }

        if (cli_params.optimization_stats_mode ==
            CliGridWarpOptimizationStatsMode::Realtime)
        {
            print_optimization_statistics(true);
        }
        else if (cli_params.optimization_stats_mode ==
            CliGridWarpOptimizationStatsMode::AtEnd)
        {
            print_optimization_statistics(false);
        }

        if (!cli_params.flag_silent)
        {
            fprintln(
                "done optimizing. stop reason: {}",
                GridWarpOptimizationStopReason_to_str_friendly(
                    optimization_info.stop_reason
                )
            );
        }

        try
        {
            if (!cli_params.output_img_path.empty())
            {
                ScopedTimer timer(
                    !cli_params.flag_silent,
                    "exporting warped image"
                );
                save_image(
                    state,
                    grid_warper->get_warped_hires_img(),
                    cli_params.output_img_path,

                    export_warped_img_undo_base_img_mul
                    ? 1.f / grid_warp_params.base_img_mul
                    : 1.f
                );
            }
        }
        catch (const std::exception& e)
        {
            throw std::runtime_error(fmt::format(
                "failed to save warped image: {}",
                e.what()
            ).c_str());
        }

        try
        {
            if (!cli_params.difference_img_after_opt_path.empty())
            {
                ScopedTimer timer(
                    !cli_params.flag_silent,
                    "exporting difference image after optimization"
                );
                save_image(
                    state,
                    grid_warper->get_difference_img(),
                    cli_params.difference_img_after_opt_path
                );
            }
        }
        catch (const std::exception& e)
        {
            throw std::runtime_error(fmt::format(
                "failed to save difference image after optimization: {}",
                e.what()
            ).c_str());
        }

        try
        {
            if (!cli_params.metadata_path.empty())
            {
                ScopedTimer timer(
                    !cli_params.flag_silent,
                    "exporting metadata"
                );
                export_metadata(cli_params.metadata_path);
            }
        }
        catch (const std::exception& e)
        {
            throw std::runtime_error(fmt::format(
                "failed to export metadata: {}",
                e.what()
            ).c_str());
        }

        if (!cli_params.flag_silent)
        {
            fprintln(
                "everything is done ({} s)",
                to_str(elapsed_sec(time_start))
            );
        }
    }

    void App::export_metadata(
        const std::filesystem::path& path
    )
    {
        if (!grid_warper)
        {
            throw std::logic_error(
                "can't export metadata if there's no grid warper"
            );
        }
        if (is_optimizing)
        {
            throw std::logic_error(
                "can't export metadata during grid warp optimization"
            );
        }

        json j;

        j["exported_from"] = fmt::format(
            "{} v{} ({})",
            APP_TITLE, APP_VERSION, APP_GITHUB_URL
        );

        if (metadata_export_options.params_and_res)
        {
            json j2;

            j2["base_img_mul"] = to_str_hp(grid_warp_params.base_img_mul);
            j2["target_img_mul"] = to_str_hp(grid_warp_params.target_img_mul);

            j2["grid_res_area"] = to_str_hp(grid_warp_params.grid_res_area);
            j2["grid_padding"] = to_str_hp(grid_warp_params.grid_padding);

            j2["intermediate_res_area"] = to_str_hp(
                grid_warp_params.intermediate_res_area
            );
            j2["cost_res_area"] = to_str_hp(grid_warp_params.cost_res_area);

            j2["rng_seed"] = to_str_hp(grid_warp_params.rng_seed);

            j["grid_warp_params"] = j2;
        }

        if (metadata_export_options.params_and_res)
        {
            json j2;

            j2["img_width"] = to_str_hp(grid_warper->get_img_width());
            j2["img_height"] = to_str_hp(grid_warper->get_img_height());

            j2["intermediate_res_x"] = to_str_hp(
                grid_warper->get_intermediate_res_x()
            );
            j2["intermediate_res_y"] = to_str_hp(
                grid_warper->get_intermediate_res_y()
            );

            j2["grid_res_x"] = to_str_hp(grid_warper->get_grid_res_x());
            j2["grid_res_y"] = to_str_hp(grid_warper->get_grid_res_y());

            j2["padded_grid_res_x"] = to_str_hp(
                grid_warper->get_padded_grid_res_x()
            );
            j2["padded_grid_res_y"] = to_str_hp(
                grid_warper->get_padded_grid_res_y()
            );

            j2["cost_res_x"] = to_str_hp(grid_warper->get_cost_res_x());
            j2["cost_res_y"] = to_str_hp(grid_warper->get_cost_res_y());

            j["grid_warp_resolutions"] = j2;
        }

        if (metadata_export_options.params_and_res)
        {
            json j2;

            j2["scale"].push_back(to_str_hp(grid_transform.scale.x));
            j2["scale"].push_back(to_str_hp(grid_transform.scale.y));

            j2["rotation"] = to_str_hp(grid_transform.rotation);

            j2["offset"].push_back(to_str_hp(grid_transform.offset.x));
            j2["offset"].push_back(to_str_hp(grid_transform.offset.y));

            j["grid_transform"] = j2;
        }

        if (metadata_export_options.optimization_info)
        {
            json j2;

            j2["scale_jitter"] = to_str_hp(optimization_params.scale_jitter);
            j2["rotation_jitter"] = to_str_hp(
                optimization_params.rotation_jitter
            );
            j2["offset_jitter"] = to_str_hp(optimization_params.offset_jitter);
            j2["n_transform_optimization_iters"] = to_str_hp(
                optimization_params.n_transform_optimization_iters
            );


            j2["warp_strength"] = to_str_hp(optimization_params.warp_strength);

            j2["min_change_in_cost_in_last_n_iters"] = to_str_hp(
                optimization_params.min_change_in_cost_in_last_n_iters
            );
            j2["N_ITERS_TO_CHECK_CHANGE_IN_COST"] = to_str_hp(
                grid_warp::N_ITERS_TO_CHECK_CHANGE_IN_COST
            );

            j2["max_iters"] = to_str_hp(optimization_params.max_iters);
            j2["max_runtime_sec"] = to_str_hp(
                optimization_params.max_runtime_sec
            );

            j["grid_warp_optimization_params"] = j2;
        }

        if (metadata_export_options.optimization_info)
        {
            json j2;

            if (!grid_warper->get_last_avg_diff())
            {
                j2["last_avg_diff"] = nullptr;
            }
            else
            {
                j2["last_avg_diff"] = to_str_hp(
                    *grid_warper->get_last_avg_diff()
                );
            }

            if (!grid_warper->get_initial_max_local_diff())
            {
                j2["initial_max_local_diff"] = nullptr;
            }
            else
            {
                j2["initial_max_local_diff"] = to_str_hp(
                    *grid_warper->get_initial_max_local_diff()
                );
            }

            j2["n_iters"] = to_str_hp(optimization_info.n_iters);
            j2["n_good_iters"] = to_str_hp(optimization_info.n_good_iters);

            j2["last_jittered_transform"]["scale"].push_back(to_str_hp(
                optimization_info.last_jittered_transform.scale.x
            ));
            j2["last_jittered_transform"]["scale"].push_back(to_str_hp(
                optimization_info.last_jittered_transform.scale.y
            ));

            j2["last_jittered_transform"]["rotation"] = to_str_hp(
                optimization_info.last_jittered_transform.rotation
            );

            j2["last_jittered_transform"]["offset"].push_back(to_str_hp(
                optimization_info.last_jittered_transform.offset.x
            ));
            j2["last_jittered_transform"]["offset"].push_back(to_str_hp(
                optimization_info.last_jittered_transform.offset.y
            ));

            j2["cost_history"] = json::array();
            for (const auto& v : optimization_info.cost_history)
            {
                j2["cost_history"].push_back(to_str_hp(v));
            }

            j2["change_in_cost_in_last_n_iters"] = to_str_hp(
                optimization_info.change_in_cost_in_last_n_iters
            );
            j2["N_ITERS_TO_CHECK_CHANGE_IN_COST"] = to_str_hp(
                grid_warp::N_ITERS_TO_CHECK_CHANGE_IN_COST
            );

            j2["accum_elapsed"] = to_str_hp(optimization_info.accum_elapsed);
            j2["stop_reason"] = GridWarpOptimizationStopReason_to_str(
                optimization_info.stop_reason
            );

            j["grid_warp_optimization_info"] = j2;
        }

        if (metadata_export_options.grid_vertices)
        {
            json j2;

            j2["count_x"] = to_str_hp(
                grid_warper->get_padded_grid_res_x() + 1
            );
            j2["count_y"] = to_str_hp(
                grid_warper->get_padded_grid_res_y() + 1
            );
            j2["count"] = to_str_hp(grid_warper->get_n_vertices());

            const auto vertices = grid_warper->get_vertices();
            for (size_t i = 0; i < grid_warper->get_n_vertices(); i++)
            {
                const auto& vert = vertices[i];

                json jvert;
                jvert["orig_pos"].push_back(to_str_hp(vert.orig_pos.x));
                jvert["orig_pos"].push_back(to_str_hp(vert.orig_pos.y));
                jvert["warped_pos"].push_back(to_str_hp(vert.warped_pos.x));
                jvert["warped_pos"].push_back(to_str_hp(vert.warped_pos.y));

                j2["items"].push_back(jvert);
            }

            j["grid_vertices"] = j2;
        }

        std::ofstream f(
            path,
            std::ofstream::out | std::ofstream::trunc
        );
        if (!f.is_open())
        {
            throw std::runtime_error("failed to open output file");
        }

        f << j.dump(metadata_export_options.pretty_print ? 2 : -1);
        f.flush();
        f.close();
    }

    void App::recreate_grid_warper()
    {
        destroy_grid_warper(false);

        std::optional<std::string> error;
        try
        {
            if (!base_img)
            {
                throw std::string("there's no base image");
            }
            if (!target_img)
            {
                throw std::string("there's no target image");
            }

            uint32_t base_img_width = base_img->config().extent.width;
            uint32_t base_img_height = base_img->config().extent.height;
            uint32_t target_img_width = target_img->config().extent.width;
            uint32_t target_img_height = target_img->config().extent.height;

            if (base_img_width != target_img_width
                || base_img_height != target_img_height)
            {
                throw std::string(
                    "base and target images must have the same resolution"
                );
            }

            grid_warp_params.base_imgview = base_imgview;
            grid_warp_params.target_imgview = target_imgview;

            grid_warper = std::make_unique<grid_warp::GridWarper>(
                state,
                grid_warp_params,
                grid_transform,
                state.queue_main
            );

            grid_warper->run_grid_warp_pass(false, state.queue_main);
            grid_warper->run_difference_and_cost_pass(state.queue_main);
        }
        catch (std::string s)
        {
            error = s;
            grid_warper = nullptr;
        }

        if (!state.cli_mode)
        {
            recreate_ui_pass();

            if (grid_warper != nullptr)
            {
                copy_grid_vertices_for_ui_preview();
            }
        }

        if (error)
        {
            throw std::runtime_error(error->c_str());
        }
    }

    void App::destroy_grid_warper(
        bool recreate_ui_pass_if_destroyed_grid_warper
    )
    {
        optimization_info = GridWarpOptimizationInfo{};
        if (grid_warper != nullptr)
        {
            grid_warper = nullptr;
            if (!state.cli_mode && recreate_ui_pass_if_destroyed_grid_warper)
            {
                recreate_ui_pass();
            }
        }
    }

    void App::start_optimization()
    {
        if (!grid_warper)
        {
            throw std::logic_error(
                "can't do optimization if there's no grid warper"
            );
        }
        if (is_optimizing)
        {
            throw std::logic_error(
                "can't start optimization if it's already running"
            );
        }

        is_optimizing = true;
        optimization_info.last_jittered_transform = grid_transform;
        optimization_info.start_time =
            std::chrono::high_resolution_clock::now();
        optimization_info.stop_reason = GridWarpOptimizationStopReason::None;

        optimization_thread_stop = false;
        optimization_thread = std::make_unique<std::jthread>(
            [this]()
            {
                bool success = false;
                try
                {
                    start_optimization_internal();
                    success = true;
                }
                IMG_ALIGNER_CATCH_ALL;

                if (!success)
                {
                    optimization_info.stop_reason =
                        GridWarpOptimizationStopReason::Error;

                    is_optimizing = false;
                }
            }
        );
    }

    void App::stop_optimization()
    {
        if (!grid_warper)
        {
            throw std::logic_error(
                "can't do optimization if there's no grid warper"
            );
        }
        if (!is_optimizing)
        {
            throw std::logic_error(
                "can't stop optimization if it isn't running"
            );
        }
        if (!optimization_thread)
        {
            throw std::logic_error(
                "can't stop optimization if there's no optimization thread"
            );
        }

        optimization_thread_stop = true;
        optimization_thread->join();
        optimization_thread = nullptr;
    }

    void App::start_optimization_internal()
    {
        while (!optimization_thread_stop)
        {
            optimization_mutex.lock();

            bool cost_decreased = false;
            if (optimization_info.n_iters <
                optimization_params.n_transform_optimization_iters)
            {
                // optimize the transform
                cost_decreased = grid_warper->optimize_transform(
                    (uint32_t)optimization_info.n_iters,
                    grid_transform,
                    optimization_params.scale_jitter,
                    optimization_params.rotation_jitter,
                    optimization_params.offset_jitter,
                    state.queue_grid_warp_optimize,
                    optimization_info.last_jittered_transform
                );
            }
            else
            {
                // optimize by warping
                cost_decreased = grid_warper->optimize_warp(
                    (uint32_t)optimization_info.n_iters,
                    optimization_params.calc_warp_strength(
                        optimization_info.n_iters
                    ),
                    state.queue_grid_warp_optimize
                );
            }

            // update optimization info
            optimization_info_mutex.lock();

            // if transform optimization is enabled but the jitter
            // intensities are effectively 0 then skip it entirely.
            if (optimization_info.n_iters == 0
                && optimization_params.n_transform_optimization_iters >
                0
                && optimization_params.scale_jitter == 1.f
                && optimization_params.rotation_jitter == 0.f
                && optimization_params.offset_jitter == 0.f)
            {
                // update number of iterations
                optimization_info.n_iters =
                    optimization_params.n_transform_optimization_iters;

                // update cost history
                if (grid_warper->get_last_avg_diff().has_value())
                {
                    for (size_t i = 0;
                        i < optimization_params.
                        n_transform_optimization_iters;
                        i++)
                    {
                        optimization_info.cost_history.push_back(
                            *grid_warper->get_last_avg_diff()
                        );
                    }
                }
            }
            else
            {
                // update number of iterations
                optimization_info.n_iters++;
                if (cost_decreased)
                {
                    optimization_info.n_good_iters++;
                }

                // update cost history
                if (grid_warper->get_last_avg_diff().has_value())
                {
                    optimization_info.cost_history.push_back(
                        *grid_warper->get_last_avg_diff()
                    );
                }
            }

            // update min. change in cost in last N iters
            if (optimization_info.cost_history.size() >
                grid_warp::N_ITERS_TO_CHECK_CHANGE_IN_COST)
            {
                optimization_info.change_in_cost_in_last_n_iters =
                    optimization_info.cost_history[
                        optimization_info.cost_history.size() - 1
                            - grid_warp::N_ITERS_TO_CHECK_CHANGE_IN_COST
                    ]
                    - optimization_info.cost_history.back();
            }

            // stop condition: min. change in cost in last N iters. this
            // should only take effect if transform optimization was
            // finished more than
            // grid_warp::N_ITERS_TO_CHECK_CHANGE_IN_COST iterations
            // ago.
            bool transform_opt_finished_long_ago =
                optimization_info.n_iters >= (
                    optimization_params.n_transform_optimization_iters
                    + grid_warp::N_ITERS_TO_CHECK_CHANGE_IN_COST
                    );
            if (transform_opt_finished_long_ago
                && optimization_info.change_in_cost_in_last_n_iters <
                optimization_params.min_change_in_cost_in_last_n_iters)
            {
                optimization_info.stop_reason =
                    GridWarpOptimizationStopReason::LowChangeInCost;
                optimization_thread_stop = true;
            }

            // stop condition: max iters
            if (optimization_params.max_iters > 0
                && optimization_info.n_iters >= optimization_params
                .max_iters)
            {
                optimization_info.stop_reason =
                    GridWarpOptimizationStopReason::ReachedMaxIters;
                optimization_thread_stop = true;
            }

            // stop condition: max run time
            float total_elapsed =
                elapsed_sec(optimization_info.start_time)
                + optimization_info.accum_elapsed;
            if (optimization_params.max_runtime_sec > 0.f
                && total_elapsed >= optimization_params.max_runtime_sec)
            {
                optimization_info.stop_reason =
                    GridWarpOptimizationStopReason::ReachedMaxRuntime;
                optimization_thread_stop = true;
            }

            optimization_info_mutex.unlock();

            // let other threads use the lock if they need to
            optimization_mutex.unlock();
            need_the_optimization_mutex.wait(true);
        }

        // finalize
        {
            std::scoped_lock lock(optimization_mutex);
            std::scoped_lock lock2(optimization_info_mutex);

            if (optimization_info.stop_reason ==
                GridWarpOptimizationStopReason::None)
            {
                optimization_info.stop_reason =
                    GridWarpOptimizationStopReason::ManuallyStopped;
            }

            // update accumulated elapsed time
            optimization_info.accum_elapsed += elapsed_sec(
                optimization_info.start_time
            );

            // run the different passes one last time
            grid_warper->run_grid_warp_pass(
                false,
                state.queue_grid_warp_optimize
            );
            grid_warper->run_grid_warp_pass(
                true,
                state.queue_grid_warp_optimize
            );
            grid_warper->run_difference_and_cost_pass(
                state.queue_grid_warp_optimize
            );

            is_optimizing = false;
        }

        // switch to warped hires image in ui pass
        if (!state.cli_mode && ui_pass != nullptr)
        {
            select_ui_pass_image(grid_warp::WARPED_HIRES_IMAGE_NAME);
            need_to_run_ui_pass = true;
        }
    }

    void App::print_optimization_statistics(bool clear)
    {
        std::scoped_lock lock(optimization_info_mutex);

        if (clear)
        {
            clear_console();
        }

        float total_elapsed = optimization_info.accum_elapsed;
        if (is_optimizing)
        {
            total_elapsed += elapsed_sec(optimization_info.start_time);
        }

        fprint(
            "\nelapsed: {} s\n"
            "total iterations: {}\n"
            "good iterations: {} ({:.1f}%)\n"
            "max local diff.: {}\n"
            "change in cost in {} iters.: {}\n"
            "cost: {}\n\n",

            to_str(total_elapsed),
            optimization_info.n_iters,
            optimization_info.n_good_iters,

            100.f * (float)optimization_info.n_good_iters
            / (float)optimization_info.n_iters,

            to_str(*grid_warper->get_initial_max_local_diff()),
            grid_warp::N_ITERS_TO_CHECK_CHANGE_IN_COST,

            optimization_info.change_in_cost_in_last_n_iters >= FLT_MAX
            ? "-"
            : to_str(optimization_info.change_in_cost_in_last_n_iters),

            optimization_info.cost_history.empty()
            ? "-"
            : to_str(optimization_info.cost_history.back())
        );
    }

    void App::recreate_ui_pass()
    {
        uint32_t max_width = 1;
        uint32_t max_height = 1;
        if (base_img != nullptr)
        {
            max_width = std::max(max_width, base_img->config().extent.width);
            max_height = std::max(max_height, base_img->config().extent.height);
        }
        if (target_img != nullptr)
        {
            max_width = std::max(max_width, target_img->config().extent.width);
            max_height = std::max(
                max_height,
                target_img->config().extent.height
            );
        }

        ui_pass = std::make_unique<UiPass>(
            state,
            max_width,
            max_height,
            state.queue_main
        );

        if (base_img != nullptr)
        {
            ui_pass->add_image(
                base_imgview,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                BASE_IMAGE_NAME,
                base_img->config().extent.width,
                base_img->config().extent.height,
                grid_warp_params.base_img_mul,
                false
            );
        }

        if (target_img != nullptr)
        {
            ui_pass->add_image(
                target_imgview,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                TARGET_IMAGE_NAME,
                target_img->config().extent.width,
                target_img->config().extent.height,
                grid_warp_params.target_img_mul,
                false
            );
        }

        if (grid_warper != nullptr)
        {
            grid_warper->add_images_to_ui_pass(*ui_pass);
        }

        if (selected_image_idx >= ui_pass->images().size())
        {
            selected_image_idx = 0;
        }

        need_to_run_ui_pass = true;
    }

    void App::copy_grid_vertices_for_ui_preview()
    {
        if (!grid_warper)
        {
            throw std::logic_error(
                "can't copy grid vertices if there's no grid warper"
            );
        }

        // copying this boolean to avoid synchronization headaches
        bool should_lock = is_optimizing;
        if (should_lock)
        {
            need_the_optimization_mutex = true;
            optimization_mutex.lock_shared();
        }

        grid_vertices_copy_for_ui_preview.resize(grid_warper->get_n_vertices());
        std::copy(
            grid_warper->get_vertices(),
            grid_warper->get_vertices() + grid_warper->get_n_vertices(),
            grid_vertices_copy_for_ui_preview.data()
        );

        if (should_lock)
        {
            optimization_mutex.unlock_shared();
            need_the_optimization_mutex = false;
            need_the_optimization_mutex.notify_all();
        }
    }

    void App::layout_controls()
    {
        // these are used to know if we should update certain stuff when related
        // parameters are modified.
        bool grid_transform_changed = false;
        bool should_update_warp_strength_plot = false;

        ImGui::Begin("Controls");

        ImGui::BeginDisabled(is_optimizing);

        imgui_bold("IMAGES");

        // load base image
        if (imgui_button_full_width("Load Base Image##Controls"))
        {
            if (browse_and_load_image(
                base_img,
                base_img_mem,
                base_imgview
            ))
            {
                destroy_grid_warper(false);

                recreate_ui_pass();
                select_ui_pass_image(BASE_IMAGE_NAME);
            }
        }

        // load target image
        if (imgui_button_full_width("Load Target Image##Controls"))
        {
            if (browse_and_load_image(
                target_img,
                target_img_mem,
                target_imgview
            ))
            {
                destroy_grid_warper(false);

                recreate_ui_pass();
                select_ui_pass_image(TARGET_IMAGE_NAME);
            }
        }

        // base image multiplier
        imgui_small_div();
        if (imgui_slider_or_drag(
            "Base Image Multiplier",
            "##base_img_mul",
            "Scale the RGB values of the base image",
            &grid_warp_params.base_img_mul,
            0.f,
            10.f,
            .1f,
            -1,
            true
        ))
        {
            grid_warp_params.base_img_mul = std::max(
                grid_warp_params.base_img_mul,
                0.f
            );

            destroy_grid_warper(true);

            // update multiplier and switch to the image
            if (ui_pass != nullptr)
            {
                for (size_t i = 0; i < ui_pass->images().size(); i++)
                {
                    if (ui_pass->images()[i].name != BASE_IMAGE_NAME)
                    {
                        continue;
                    }

                    ui_pass->images()[i].mul = grid_warp_params.base_img_mul;
                    selected_image_idx = i;
                    break;
                }
                need_to_run_ui_pass = true;
            }
        }

        // target image multiplier
        imgui_small_div();
        if (imgui_slider_or_drag(
            "Target Image Multiplier",
            "##target_img_mul",
            "Scale the RGB values of the target image",
            &grid_warp_params.target_img_mul,
            0.f,
            10.f,
            .1f,
            -1,
            true
        ))
        {
            grid_warp_params.target_img_mul = std::max(
                grid_warp_params.target_img_mul,
                0.f
            );

            destroy_grid_warper(true);

            // update multiplier and switch to the image
            if (ui_pass != nullptr)
            {
                for (size_t i = 0; i < ui_pass->images().size(); i++)
                {
                    if (ui_pass->images()[i].name != TARGET_IMAGE_NAME)
                    {
                        continue;
                    }

                    ui_pass->images()[i].mul = grid_warp_params.target_img_mul;
                    selected_image_idx = i;
                    break;
                }
                need_to_run_ui_pass = true;
            }
        }

        imgui_div();
        imgui_bold("GRID WARPER");

        // grid resolution
        imgui_small_div();
        if (imgui_slider_or_drag(
            "Grid Resolution",
            "##grid_res",
            "Area of the warping grid resolution",
            &grid_warp_params.grid_res_area,
            (uint32_t)1,
            (uint32_t)(8192u * 8192u),
            .8f
        ))
        {
            destroy_grid_warper(true);
        }

        // grid padding
        imgui_small_div();
        if (imgui_slider_or_drag(
            "Grid Padding",
            "##grid_padding",
            "The actual grid used for warping has extra added borders to "
            "prevent black empty spaces when the edges get warped. This value "
            "controls the amount of that padding proportional to the grid "
            "resolution.",
            &grid_warp_params.grid_padding,
            0.f,
            1.f,
            .004f
        ))
        {
            destroy_grid_warper(true);
        }

        // intermediate resolution
        imgui_small_div();
        if (imgui_slider_or_drag(
            "Intermediate Resolution",
            "##intermediate_res",
            "The images are temporarily downsampled throughout the "
            "optimization process to improve computation speed. This value "
            "defines the area of the intermediate image resolution.",
            &grid_warp_params.intermediate_res_area,
            (uint32_t)1,
            (uint32_t)(16384u * 16384u),
            2000.f
        ))
        {
            destroy_grid_warper(true);
        }

        // cost resolution
        imgui_small_div();
        if (imgui_slider_or_drag(
            "Cost Resolution",
            "##cost_res",
            "The difference image is smoothly downscaled to the (normally "
            "tiny) cost resolution after which we find the maximum and average "
            "of its pixels for optimization. This value defines the area of "
            "the cost resolution.",
            &grid_warp_params.cost_res_area,
            (uint32_t)1,
            (uint32_t)(256u * 256u),
            1.f
        ))
        {
            destroy_grid_warper(true);
        }

        // RNG seed
        imgui_small_div();
        ImGui::TextWrapped("Seed");
        imgui_tooltip(
            "Seed number to use for pseudo-random number generators"
        );
        ImGui::SetNextItemWidth(-FLT_MIN);
        if (ImGui::InputScalar(
            "##rng_seed",
            ImGuiDataType_U32,
            &grid_warp_params.rng_seed
        ))
        {
            destroy_grid_warper(true);
        }

        // create grid warper
        imgui_small_div();
        if (!grid_warper && imgui_button_full_width("Recreate Grid Warper"))
        {
            try
            {
                recreate_grid_warper();

                // switch to the difference image
                if (ui_pass != nullptr)
                {
                    select_ui_pass_image(grid_warp::DIFFERENCE_IMAGE_NAME);
                    need_to_run_ui_pass = true;
                }
            }
            catch (const std::exception& e)
            {
                current_errors.push_back(fmt::format(
                    "Failed to recreate grid warper: {}",
                    e.what()
                ));
                ImGui::OpenPopup(ERROR_DIALOG_TITLE);
            }
        }

        imgui_div();
        imgui_bold("TRANSFORM");

        imgui_small_div();
        ImGui::TextWrapped(
            "Modifying the transform will reset any previous warping or "
            "optimization."
        );

        imgui_small_div();
        if (imgui_slider_or_drag(
            "Scale",
            "##transform_scale",
            "",
            (float*)&grid_transform.scale.x,
            0.f,
            2.f,
            -1.f,
            4,
            false,
            2
        ))
        {
            if (glfwGetKey(state.window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS
                || glfwGetKey(state.window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS)
            {
                float avg = glm::dot(grid_transform.scale, glm::vec2(.5f));
                grid_transform.scale = glm::vec2(avg);
            }

            grid_transform_changed = true;
        }
        imgui_tooltip("Hold [Shift] to lock");

        imgui_small_div();
        if (imgui_slider_or_drag(
            "Rotation",
            "##transform_rotation",
            "",
            &grid_transform.rotation,
            0.f,
            360.f,
            .1f,
            4,
            false,
            1,
            false,
            true
        ))
        {
            grid_transform_changed = true;
        }

        imgui_small_div();
        if (imgui_slider_or_drag(
            "Offset",
            "##transform_offset",
            "",
            (float*)&grid_transform.offset.x,
            -1.f,
            1.f,
            .001f,
            4,
            false,
            2
        ))
        {
            grid_transform_changed = true;
        }

        imgui_small_div();
        if (imgui_button_full_width("Reset"))
        {
            grid_transform = Transform2d();
            grid_transform_changed = true;
        }

        // if the grid transform was modified, regenerate grid vertices, run
        // grid warp, difference, and cost passes, and switch to the difference
        // image.
        if (grid_transform_changed)
        {
            // reset optimization
            optimization_info = GridWarpOptimizationInfo{};

            if (grid_warper != nullptr)
            {
                grid_warper->regenerate_grid_vertices(grid_transform);

                grid_warper->run_grid_warp_pass(false, state.queue_main);
                grid_warper->run_difference_and_cost_pass(state.queue_main);

                if (ui_pass != nullptr)
                {
                    select_ui_pass_image(grid_warp::DIFFERENCE_IMAGE_NAME);
                    need_to_run_ui_pass = true;
                }

                copy_grid_vertices_for_ui_preview();
            }
        }

        imgui_div();
        imgui_bold("TRANSFORM OPTIMIZATION");

        // scale jitter
        imgui_small_div();
        imgui_slider_or_drag(
            "Scale Jitter",
            "##scale_jitter",
            "Span of the range in which the scale of the current grid "
            "transform will be jittered. If this is 1.25, for example, the "
            "scale of the current grid transform will itself be scaled by a "
            "random value ranging from (1 / 1.25 = 0.8) to 1.25.",
            &optimization_params.scale_jitter,
            1.f,
            1.25f,
            .01f,
            4
        );

        // rotation jitter
        imgui_small_div();
        imgui_slider_or_drag(
            "Rotation Jitter",
            "##rotation_jitter",
            "Span of the range in which the rotation of the current grid "
            "transform will be jittered. If this is 2, for example, the "
            "rotation of the current grid transform will be offset by a random "
            "value ranging from -2 to 2.",
            &optimization_params.rotation_jitter,
            0.f,
            10.f,
            .1f,
            4
        );

        // offset jitter
        imgui_small_div();
        imgui_slider_or_drag(
            "Offset Jitter",
            "##offset_jitter",
            "Span of the range in which the offset of the current grid "
            "transform will be jittered. If this is 0.01, for example, the "
            "offset of the current grid transform will itself be offset by a "
            "random vector in a square with a radius of 0.01 units (in "
            "normalized, zero-centered, aspect-ratio-adjusted UV space).",
            &optimization_params.offset_jitter,
            0.f,
            .1f,
            .001f,
            4
        );

        // number of transform optimization iterations
        imgui_small_div();
        imgui_slider_or_drag(
            "Number of Iterations",
            "##n_transform_optimization_iters",
            "Use 0 to disable transform optimization.",
            &optimization_params.n_transform_optimization_iters,
            (uint32_t)0,
            (uint32_t)10000,
            2.f
        );

        imgui_div();
        imgui_bold("WARP OPTIMIZATION");

        // warp strength
        imgui_small_div();
        if (imgui_slider_or_drag(
            "Warp Strength",
            "##warp_strength",
            "",
            &optimization_params.warp_strength,
            .000001f,
            .1f,
            -1.f,
            6,
            true
        ))
        {
            should_update_warp_strength_plot = true;
        }

        // warp strength decay rate
        imgui_small_div();
        if (imgui_slider_or_drag(
            "Warp Strength Decay Rate",
            "##warp_strength_decay_rate",
            "Warp strength will be scaled by e^(-di) where d is the decay rate "
            "and i is the number of iterations.",
            &optimization_params.warp_strength_decay_rate,
            0.f,
            .05f,
            -1.f,
            6,
            true
        ))
        {
            should_update_warp_strength_plot = true;
        }

        // min warp strength
        imgui_small_div();
        if (imgui_slider_or_drag(
            "Min Warp Strength",
            "##min_warp_strength",
            "Lower limit of warp strength",
            &optimization_params.min_warp_strength,
            0.f,
            .01f,
            -1.f,
            6,
            true
        ))
        {
            should_update_warp_strength_plot = true;
        }

        // update warp strength plot if needed
        if (should_update_warp_strength_plot || warp_strength_plot.empty())
        {
            warp_strength_plot.resize(
                GRID_WARP_OPTIMIZATION_WARP_STRENGTH_PLOT_N_ITERS
            );
            for (size_t i = 0; i < warp_strength_plot.size(); i++)
            {
                warp_strength_plot[i] =
                    optimization_params.calc_warp_strength(i);
            }
        }

        // draw warp strength plot
        imgui_small_div();
        {
            ImGui::TextWrapped(fmt::format(
                "Warp Strength in {} Iterations",
                GRID_WARP_OPTIMIZATION_WARP_STRENGTH_PLOT_N_ITERS
            ).c_str());

            ImGui::PlotLines(
                "##warp_strength_plot",
                warp_strength_plot.data(),
                (int)warp_strength_plot.size(),
                0,
                "##",
                0.f,
                optimization_params.warp_strength,
                ImVec2{
                    -FLT_MIN,
                    120.f * ui_scale
                }
            );
        }

        imgui_div();
        imgui_bold("STOP IF");

        // min change in cost in N iters
        imgui_small_div();
        imgui_slider_or_drag(
            fmt::format(
                "In {} iterations, the cost decreased by less than",
                grid_warp::N_ITERS_TO_CHECK_CHANGE_IN_COST
            ),
            "##min_change_in_cost_in_last_n_iters",
            "This only takes affect after transform optimization is finished.",
            &optimization_params.min_change_in_cost_in_last_n_iters,
            0.f,
            .001f,
            -1.f,
            6,
            true
        );

        // max iterations
        imgui_small_div();
        imgui_slider_or_drag(
            "The total number of iterations exceeds",
            "##max_iters",
            "This includes transform optimization iterations as well. Use 0 "
            "for unlimited number of iterations.",
            &optimization_params.max_iters,
            (uint32_t)0,
            (uint32_t)4000000000,
            30.f
        );

        // max runtime
        imgui_small_div();
        imgui_slider_or_drag(
            "Run time exceeds (seconds)",
            "##max_runtime",
            "Use 0 for unlimited run time",
            &optimization_params.max_runtime_sec,
            0.f,
            3000000.f,
            300.f,
            -1,
            true
        );

        ImGui::EndDisabled();

        // start alignin' / stop
        imgui_small_div();
        if (is_optimizing)
        {
            if (imgui_button_full_width("Stop##Controls"))
            {
                stop_optimization();
            }
        }
        else
        {
            ImGui::BeginDisabled(!grid_warper);

            const char* button_label =
                (optimization_info.n_iters > 0)
                ? "Continue Alignin'##Controls"
                : "Start Alignin'##Controls";
            if (imgui_button_full_width(button_label)
                && grid_warper != nullptr)
            {
                start_optimization();
                select_ui_pass_image(grid_warp::DIFFERENCE_IMAGE_NAME);
            }

            ImGui::EndDisabled();
        }

        if (grid_warper != nullptr && optimization_info.n_iters > 0)
        {
            std::scoped_lock lock(optimization_info_mutex);

            imgui_div();
            imgui_bold("STATS");

            if (optimization_info.stop_reason
                != GridWarpOptimizationStopReason::None)
            {
                ImGui::TextWrapped(
                    "Stop reason: %s",
                    GridWarpOptimizationStopReason_to_str_friendly(
                        optimization_info.stop_reason
                    )
                );
            }

            float total_elapsed = optimization_info.accum_elapsed;
            if (is_optimizing)
            {
                total_elapsed += elapsed_sec(optimization_info.start_time);
            }

            ImGui::TextWrapped(fmt::format(
                "Elapsed: {} s",
                to_str(total_elapsed, 3, 0, 2)
            ).c_str());

            ImGui::TextWrapped(fmt::format(
                "Total Iterations: {}",
                optimization_info.n_iters
            ).c_str());

            ImGui::TextWrapped(fmt::format(
                "Good Iterations: {} ({:.1f}%%)",
                optimization_info.n_good_iters,
                100.f * (float)optimization_info.n_good_iters
                / (float)optimization_info.n_iters
            ).c_str());
            imgui_tooltip(
                "The number of iterations where the cost decreased (instead of "
                "staying still)"
            );

            if (grid_warper->get_initial_max_local_diff().has_value())
            {
                ImGui::TextWrapped(fmt::format(
                    "Max Local Diff.: {}",
                    to_str(*grid_warper->get_initial_max_local_diff())
                ).c_str());
            }
            imgui_tooltip("Maximum value in the pixels of the cost image.");

            if (optimization_info.change_in_cost_in_last_n_iters >= FLT_MAX)
            {
                ImGui::TextWrapped("Change in Cost: -");
            }
            else
            {
                ImGui::TextWrapped(fmt::format(
                    "Change in Cost: {}",
                    to_str(optimization_info.change_in_cost_in_last_n_iters)
                ).c_str());
            }
            imgui_tooltip(fmt::format(
                "How much the cost decreased in the last {} iterations",
                grid_warp::N_ITERS_TO_CHECK_CHANGE_IN_COST
            ).c_str());

            if (!optimization_info.cost_history.empty())
            {
                ImGui::TextWrapped(fmt::format(
                    "Cost: {}",
                    to_str(optimization_info.cost_history.back())
                ).c_str());

                ImGui::PlotLines(
                    "##cost_history_plot",
                    optimization_info.cost_history.data(),
                    (int)optimization_info.cost_history.size(),
                    0,
                    "##",
                    FLT_MAX,
                    FLT_MAX,
                    ImVec2{
                        -FLT_MIN,
                        200.f * ui_scale
                    }
                );
            }
        }

        imgui_div();

        ImGui::BeginDisabled(!grid_warper || is_optimizing);

        imgui_bold("EXPORT IMAGES");

        ImGui::BeginDisabled(optimization_info.n_iters < 1);

        // undo base image multiplier when exporting warped image
        ImGui::Checkbox(
            "Undo Base Image Multiplier",
            &export_warped_img_undo_base_img_mul
        );
        imgui_tooltip(
            "Apply the inverse of the base image multiplier when exporting the "
            "warped image"
        );

        // export warped image
        if (imgui_button_full_width("Export Warped Image"))
        {
            browse_and_save_image(
                grid_warper->get_warped_hires_img(),

                export_warped_img_undo_base_img_mul
                ? 1.f / grid_warp_params.base_img_mul
                : 1.f
            );
        }
        imgui_tooltip("Export the warped image at full resolution");

        ImGui::EndDisabled();

        // export difference image
        if (imgui_button_full_width("Export Difference"))
        {
            browse_and_save_image(grid_warper->get_difference_img());
        }
        imgui_tooltip(
            "Export the difference image at the intermediate resolution"
        );

        ImGui::EndDisabled();

        imgui_div();

        ImGui::BeginDisabled(
            !grid_warper || is_optimizing
        );

        imgui_bold("EXPORT METADATA");

        // metadata export options

        ImGui::Checkbox(
            "Parameters & Resolutions",
            &metadata_export_options.params_and_res
        );
        imgui_tooltip(
            "Include grid warp parameters and internal resolutions"
        );

        ImGui::Checkbox(
            "Optimization Info",
            &metadata_export_options.optimization_info
        );
        imgui_tooltip(
            "Include grid warp optimization parameters and statistics"
        );

        ImGui::Checkbox(
            "Grid Vertices",
            &metadata_export_options.grid_vertices
        );
        imgui_tooltip(
            "Include grid vertex data"
        );

        ImGui::Checkbox(
            "Pretty Print",
            &metadata_export_options.pretty_print
        );
        imgui_tooltip(
            "Produce pretty printed JSON"
        );

        // export metadata
        if (imgui_button_full_width("Export Metadata")
            && grid_warper != nullptr)
        {
            browse_and_export_metadata();
        }

        ImGui::EndDisabled();

        imgui_dialogs();

        imgui_div();
        ImGui::End();
    }

    void App::layout_misc()
    {
        ImGui::Begin("Misc");

        imgui_bold("INTERFACE");

        if (ImGui::InputFloat(
            "Scale##Misc", &ui_scale, .125f, .25f, "%.3f"
        ))
        {
            ui_scale = std::clamp(ui_scale, .75f, 2.f);
            ui_scale_updated = true;
        }

        imgui_div();
        imgui_bold("INFO");

        // version
        ImGui::TextWrapped(fmt::format(
            "{} v{}",
            APP_TITLE,
            APP_VERSION
        ).c_str());

        // GitHub
        if (imgui_button_full_width("GitHub##Misc"))
        {
            open_url(APP_GITHUB_URL);
        }

        imgui_div();
        ImGui::End();
    }

    void App::layout_image_viewer()
    {
        ImGui::Begin("Image Viewer", 0, ImGuiWindowFlags_HorizontalScrollbar);

        if (!ui_pass || ui_pass->images().size() < 1)
        {
            ImGui::End();
            return;
        }

        // make sure selected_image_idx is valid
        if (selected_image_idx < 0
            || selected_image_idx >= ui_pass->images().size())
        {
            selected_image_idx = 0;
            need_to_run_ui_pass = true;
        }

        // image selector
        std::vector<std::string> image_names;
        for (const auto& ui_image_info : ui_pass->images())
        {
            image_names.push_back(ui_image_info.name);
        }
        if (imgui_combo(
            "##image_selector",
            image_names,
            &selected_image_idx,
            false
        ))
        {
            need_to_run_ui_pass = true;
        }

        const auto& sel_img_info =
            ui_pass->images()[selected_image_idx];

        // this is a hack thanks to ImGui. I'm not even gonna explain what it's
        // fixing. disable it and see for yourself.
        {
            float original_padding = ImGui::GetWindowContentRegionMin().x;
            original_padding = std::max(original_padding, FLT_MIN);

            float old_alpha = ImGui::GetStyle().Alpha;
            ImGui::GetStyle().Alpha = 0.f;
            {
                bool unused;
                ImGui::Checkbox("##", &unused);
            }
            ImGui::GetStyle().Alpha = old_alpha;

            ImGui::SameLine(original_padding);
        }

        // image size
        ImGui::Text("%ux%u", sel_img_info.width, sel_img_info.height);

        imgui_horiz_div();

        // zoom
        {
            ImGui::SameLine();
            ImGui::Text("Zoom");

            ImGui::SameLine();
            ImGui::SetNextItemWidth(70.f * ui_scale);
            ImGui::DragFloat(
                "##image_zoom",
                &image_viewer_zoom,
                .005f,
                .1f,
                3.f,
                "%.2f",
                ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_NoRoundToFormat
            );

            ImGui::SameLine();
            if (ImGui::Button("R##image_zoom_reset"))
            {
                image_viewer_zoom = 1.f;
            }
        }

        // fit
        ImGui::SameLine();
        if (ImGui::Checkbox("Fit", &image_viewer_fit))
        {
            if (image_viewer_fit)
            {
                image_viewer_zoom = 1.f;
            }
        }

        imgui_horiz_div();

        // exposure
        {
            ImGui::SameLine();
            ImGui::Text("Exposure");

            ImGui::SameLine();
            ImGui::SetNextItemWidth(70.f * ui_scale);
            if (ImGui::DragFloat(
                "##image_exposure",
                &image_viewer_exposure,
                .05f,
                -10.f,
                10.f,
                "%.2f",
                ImGuiSliderFlags_NoRoundToFormat
            ))
            {
                need_to_run_ui_pass = true;
            }

            ImGui::SameLine();
            if (ImGui::Button("R##image_exposure_reset"))
            {
                image_viewer_exposure = 0.f;
                need_to_run_ui_pass = true;
            }
        }

        imgui_horiz_div();

        // use flim
        ImGui::SameLine();
        if (ImGui::Checkbox("flim", &image_viewer_use_flim))
        {
            need_to_run_ui_pass = true;
        }
        imgui_tooltip(
            "Use flim color transform (https://github.com/bean-mhm/flim)"
        );

        imgui_horiz_div();

        // preview grid
        ImGui::SameLine();
        ImGui::BeginDisabled(!grid_warper);
        ImGui::Checkbox("Preview Grid", &preview_grid);
        imgui_tooltip(
            "Preview the grid lines (including padding). This will only work "
            "if the base and target images are loaded and have identical "
            "resolutions."
        );
        ImGui::EndDisabled();

        // sub-child for the image and the grid preview
        ImGui::BeginChild(
            "##image",
            ImGui::GetContentRegionAvail(),
            0,
            ImGuiWindowFlags_NoBackground
            | ImGuiWindowFlags_NoCollapse
            | ImGuiWindowFlags_NoSavedSettings
            | ImGuiWindowFlags_HorizontalScrollbar
        );
        {
            // image
            float image_scale = image_viewer_zoom;
            if (image_viewer_fit)
            {
                auto parent_size = ImGui::GetWindowSize();
                image_scale *= .97f * (float)std::min(
                    parent_size.x / sel_img_info.width,
                    parent_size.y / sel_img_info.height
                );
            }
            ui_pass->draw_imgui_image(
                ui_pass->images()[selected_image_idx],
                image_scale
            );

            // get the 4 corners of the last item which is the image
            // (bl = bottom left, tr = top right, etc.). make sure to remove the
            // offset caused by the image border.
            glm::vec2 image_tl = imvec_to_glm(ImGui::GetItemRectMin()) + 1.f;
            glm::vec2 image_br = imvec_to_glm(ImGui::GetItemRectMax()) - 2.f;
            glm::vec2 image_tr{ image_br.x, image_tl.y };
            glm::vec2 image_bl{ image_tl.x, image_br.y };
            glm::vec2 image_span = image_tr - image_bl;

            // preview grid lines
            if (preview_grid && grid_warper != nullptr)
            {
                // update the copy of the vertices if it has the wrong size
                bool size_mismatch =
                    grid_vertices_copy_for_ui_preview.size()
                    != grid_warper->get_n_vertices();
                if (size_mismatch)
                {
                    copy_grid_vertices_for_ui_preview();
                }

                ImDrawList* draw_list = ImGui::GetWindowDrawList();

                uint32_t padded_res_x = grid_warper->get_padded_grid_res_x();
                uint32_t padded_res_y = grid_warper->get_padded_grid_res_y();
                const grid_warp::GridVertex* vertices =
                    grid_vertices_copy_for_ui_preview.data();

                const ImU32 line_col = ImGui::ColorConvertFloat4ToU32(
                    { .65f, .65f, .65f, .7f }
                );
                const ImU32 line_col_outside = ImGui::ColorConvertFloat4ToU32(
                    { .8f, .8f, .8f, .05f }
                );
                float line_thickness = 1.f;

                // remember that the number of vertices is
                // (padded_res_x + 1) * (padded_res_y + 1) to cover all edges.
                uint32_t stride_y = padded_res_x + 1;
                for (uint32_t y = 0; y <= padded_res_y; y++)
                {
                    for (uint32_t x = 0; x <= padded_res_x; x++)
                    {
                        const auto& vert = vertices[x + y * stride_y];
                        if (x < padded_res_x)
                        {
                            auto& vert_right = vertices[(x + 1) + y * stride_y];

                            bool is_outside =
                                vec2_is_outside_01(vert.orig_pos)
                                || vec2_is_outside_01(vert_right.orig_pos);

                            draw_list->AddLine(
                                imvec_from_glm(
                                    image_bl + image_span * vert.warped_pos
                                ),
                                imvec_from_glm(
                                    image_bl
                                    + image_span * vert_right.warped_pos
                                ),
                                is_outside ? line_col_outside : line_col,
                                line_thickness
                            );
                        }
                        if (y < padded_res_y)
                        {
                            auto& vert_up = vertices[x + (y + 1) * stride_y];

                            bool is_outside =
                                vec2_is_outside_01(vert.orig_pos)
                                || vec2_is_outside_01(vert_up.orig_pos);

                            draw_list->AddLine(
                                imvec_from_glm(
                                    image_bl + image_span * vert.warped_pos
                                ),
                                imvec_from_glm(
                                    image_bl + image_span * vert_up.warped_pos
                                ),
                                is_outside ? line_col_outside : line_col,
                                line_thickness
                            );
                        }
                    }
                }
            }
        }
        ImGui::EndChild();

        ImGui::End();
    }

    void App::setup_imgui_style()
    {
        // img-aligner 2 style from ImThemes
        ImGuiStyle& style = ImGui::GetStyle();

        style.Alpha = 1.0;
        style.DisabledAlpha = 0.5;
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
        style.ItemSpacing = ImVec2(7.0, 5.0);
        style.ItemInnerSpacing = ImVec2(6.0, 3.0);
        style.CellPadding = ImVec2(8.0, 5.0);
        style.IndentSpacing = 20.0;
        style.ColumnsMinSpacing = 6.0;
        style.ScrollbarSize = 12.0;
        style.ScrollbarRounding = 100.0;
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
        style.Colors[ImGuiCol_WindowBg] = ImVec4(0.09609425067901611, 0.08342945575714111, 0.1372549086809158, 1.0);
        style.Colors[ImGuiCol_ChildBg] = ImVec4(0.08734812587499619, 0.07234141230583191, 0.125490203499794, 1.0);
        style.Colors[ImGuiCol_PopupBg] = ImVec4(0.1503584384918213, 0.1302575916051865, 0.2156862765550613, 1.0);
        style.Colors[ImGuiCol_Border] = ImVec4(1.0, 1.0, 1.0, 0.0313725508749485);
        style.Colors[ImGuiCol_BorderShadow] = ImVec4(0.0784313753247261, 0.08627451211214066, 0.1019607856869698, 0.0);
        style.Colors[ImGuiCol_FrameBg] = ImVec4(0.1421570628881454, 0.1231526285409927, 0.2039215713739395, 1.0);
        style.Colors[ImGuiCol_FrameBgHovered] = ImVec4(0.1805853843688965, 0.1532641500234604, 0.2588235437870026, 1.0);
        style.Colors[ImGuiCol_FrameBgActive] = ImVec4(0.22197425365448, 0.1358246803283691, 0.501960813999176, 1.0);
        style.Colors[ImGuiCol_TitleBg] = ImVec4(0.04083044454455376, 0.03529411926865578, 0.05882352963089943, 1.0);
        style.Colors[ImGuiCol_TitleBgActive] = ImVec4(0.06465748697519302, 0.0499807745218277, 0.1019607856869698, 1.0);
        style.Colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.04100684449076653, 0.03552479669451714, 0.05882352963089943, 1.0);
        style.Colors[ImGuiCol_MenuBarBg] = ImVec4(0.06465748697519302, 0.0499807745218277, 0.1019607856869698, 1.0);
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
        style.Colors[ImGuiCol_SeparatorHovered] = ImVec4(0.2416060566902161, 0.2003998607397079, 0.3254902064800262, 1.0);
        style.Colors[ImGuiCol_SeparatorActive] = ImVec4(0.2416060566902161, 0.2003998607397079, 0.3254902064800262, 1.0);
        style.Colors[ImGuiCol_ResizeGrip] = ImVec4(0.1703480333089828, 0.1491580158472061, 0.239215686917305, 1.0);
        style.Colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.2113431394100189, 0.1846674382686615, 0.2980392277240753, 1.0);
        style.Colors[ImGuiCol_ResizeGripActive] = ImVec4(0.3330428004264832, 0.2171164900064468, 0.7098039388656616, 1.0);
        style.Colors[ImGuiCol_Tab] = ImVec4(0.1514349430799484, 0.1328719705343246, 0.2117647081613541, 1.0);
        style.Colors[ImGuiCol_TabHovered] = ImVec4(0.2113431394100189, 0.1846674382686615, 0.2980392277240753, 1.0);
        style.Colors[ImGuiCol_TabActive] = ImVec4(0.3330428004264832, 0.2171164900064468, 0.7098039388656616, 1.0);
        style.Colors[ImGuiCol_TabUnfocused] = ImVec4(0.1514349430799484, 0.1328719705343246, 0.2117647081613541, 1.0);
        style.Colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.276118129491806, 0.1752556711435318, 0.6039215922355652, 1.0);
        style.Colors[ImGuiCol_PlotLines] = ImVec4(0.4413722157478333, 0.3442983329296112, 0.7568627595901489, 1.0);
        style.Colors[ImGuiCol_PlotLinesHovered] = ImVec4(0.831504225730896, 0.6952790021896362, 1.0, 1.0);
        style.Colors[ImGuiCol_PlotHistogram] = ImVec4(0.4208443462848663, 0.3162475824356079, 0.7607843279838562, 1.0);
        style.Colors[ImGuiCol_PlotHistogramHovered] = ImVec4(0.5916955471038818, 0.4117646813392639, 1.0, 1.0);
        style.Colors[ImGuiCol_TableHeaderBg] = ImVec4(0.5414839386940002, 0.4209150075912476, 0.9333333373069763, 0.1803921610116959);
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
        font = state.io->Fonts->AddFontFromFileTTF(
            (exec_dir() / FONT_PATH).string().c_str(),
            FONT_SIZE * ui_scale
        );
        font_bold = state.io->Fonts->AddFontFromFileTTF(
            (exec_dir() / FONT_BOLD_PATH).string().c_str(),
            FONT_SIZE * ui_scale
        );
        if (!font || !font_bold)
        {
            throw std::runtime_error("failed to load fonts");
        }
        state.io->Fonts->Build();
        ImGui_ImplVulkan_CreateFontsTexture();

        // reload style and apply scale
        setup_imgui_style();
        ImGui::GetStyle().ScaleAllSizes(ui_scale);

        ImGui::GetStyle().HoverDelayNormal = .65f;
        ImGui::GetStyle().HoverStationaryDelay = .2f;
    }

    void App::imgui_div()
    {
        ImGui::Dummy({ 1.f, 26.f * ui_scale });
    }

    void App::imgui_small_div()
    {
        ImGui::Dummy({ 1.f, 5.f * ui_scale });
    }

    void App::imgui_horiz_div()
    {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1.f, 1.f, 1.f, .3f), " | ");
    }

    void App::imgui_bold(std::string_view s)
    {
        ImGui::PushFont(font_bold);
        ImGui::TextWrapped(s.data());
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

    bool App::imgui_button_full_width(const char* label)
    {
        return ImGui::Button(label, { -FLT_MIN, 35.f * ui_scale });
    }

    void App::imgui_tooltip(std::string_view s)
    {
        if (s.empty()
            || !ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)
            || !ImGui::BeginTooltip())
        {
            return;
        }

        ImGui::PushTextWrapPos(400.f * ui_scale);
        ImGui::TextWrapped(s.data());
        ImGui::PopTextWrapPos();

        ImGui::EndTooltip();
    }

    void App::imgui_dialogs()
    {
        ImGui::SetNextWindowPos(
            {
                .5f * (float)state.imgui_vk_window_data.Width,
                .5f * (float)state.imgui_vk_window_data.Height
            },
            0,
            { .5f, .5f }
        );

        // error dialog
        if (ImGui::BeginPopupModal(
            ERROR_DIALOG_TITLE,
            nullptr,
            ImGuiWindowFlags_AlwaysAutoResize
        ))
        {
            std::string s;
            for (size_t i = 0; i < current_errors.size(); i++)
            {
                if (i != 0)
                {
                    s += '\n';
                }
                s += current_errors[i];
            }

            ImGui::TextWrapped(s.c_str());
            ImGui::NewLine();
            if (ImGui::Button("OK##error_dialog", dialog_button_size()))
            {
                current_errors.clear();
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }
    }

    ImVec2 App::dialog_button_size()
    {
        return { 350.f * ui_scale, 35.f * ui_scale };
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
                state.queue_main->handle(),
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

        VkResult vk_result = vkQueuePresentKHR(
            state.queue_main->handle(),
            &info
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
                "failed to present frame",
                vk_result,
                false
            );
        }

        window_data.SemaphoreIndex =
            (window_data.SemaphoreIndex + 1) % window_data.SemaphoreCount;
    }

    bool App::browse_and_load_image(
        bv::ImagePtr& img,
        bv::MemoryChunkPtr& img_mem,
        bv::ImageViewPtr& imgview
    )
    {
        nfdu8char_t* nfd_filename;
        nfdopendialogu8args_t args{ 0 };
        nfdu8filteritem_t filters[]{
            { "Images", "exr,png,jpg,jpeg" },
            { "OpenEXR", "exr" },
            { "PNG", "png" },
            { "JPEG", "jpg,jpeg" }
        };
        args.filterList = filters;
        args.filterCount = sizeof(filters) / sizeof(nfdu8filteritem_t);

        nfdresult_t result = NFD_OpenDialogU8_With(&nfd_filename, &args);
        if (result == NFD_OKAY)
        {
            std::string filename(nfd_filename);
            NFD_FreePathU8(nfd_filename);

            try
            {
                load_image(state, filename, img, img_mem, imgview);
                return true;
            }
            catch (const std::exception& e)
            {
                current_errors.push_back(fmt::format(
                    "Failed to load image from file \"{}\": {}",
                    filename,
                    e.what()
                ));
                ImGui::OpenPopup(ERROR_DIALOG_TITLE);
            }
        }
        else if (result == NFD_CANCEL)
        {
            // user pressed cancel
        }
        else if (result == NFD_ERROR)
        {
            current_errors.push_back(fmt::format(
                "Native File Dialog: {}",
                NFD_GetError()
            ));
            ImGui::OpenPopup(ERROR_DIALOG_TITLE);
        }

        return false;
    }

    void App::browse_and_save_image(const bv::ImagePtr& img, float mul)
    {
        nfdu8char_t* nfd_filename;
        nfdsavedialogu8args_t args{ 0 };
        nfdu8filteritem_t filters[]{
            { "OpenEXR", "exr" },
            { "PNG", "png" },
            { "JPEG", "jpg,jpeg" }
        };
        args.filterList = filters;
        args.filterCount = sizeof(filters) / sizeof(nfdu8filteritem_t);

        nfdresult_t result = NFD_SaveDialogU8_With(&nfd_filename, &args);
        if (result == NFD_OKAY)
        {
            std::string filename(nfd_filename);
            NFD_FreePathU8(nfd_filename);

            try
            {
                save_image(state, img, filename, mul);
            }
            catch (const std::exception& e)
            {
                current_errors.push_back(fmt::format(
                    "Failed to save image to file \"{}\": {}",
                    filename,
                    e.what()
                ));
                ImGui::OpenPopup(ERROR_DIALOG_TITLE);
            }
        }
        else if (result == NFD_CANCEL)
        {
            // user pressed cancel
        }
        else if (result == NFD_ERROR)
        {
            current_errors.push_back(fmt::format(
                "Native File Dialog: {}",
                NFD_GetError()
            ));
            ImGui::OpenPopup(ERROR_DIALOG_TITLE);
        }
    }

    void App::browse_and_export_metadata()
    {
        nfdu8char_t* nfd_filename;
        nfdsavedialogu8args_t args{ 0 };
        nfdu8filteritem_t filters[1] = { { "JSON", "json" } };
        args.filterList = filters;
        args.filterCount = sizeof(filters) / sizeof(nfdu8filteritem_t);
        nfdresult_t result = NFD_SaveDialogU8_With(&nfd_filename, &args);
        if (result == NFD_OKAY)
        {
            std::string filename(nfd_filename);
            NFD_FreePathU8(nfd_filename);

            try
            {
                export_metadata(filename);
            }
            catch (const std::exception& e)
            {
                current_errors.push_back(fmt::format(
                    "Failed to export metadata to file \"{}\": {}",
                    filename,
                    e.what()
                ));
                ImGui::OpenPopup(ERROR_DIALOG_TITLE);
            }
        }
        else if (result == NFD_CANCEL)
        {
            // user pressed cancel
        }
        else if (result == NFD_ERROR)
        {
            current_errors.push_back(fmt::format(
                "Native File Dialog: {}",
                NFD_GetError()
            ));
            ImGui::OpenPopup(ERROR_DIALOG_TITLE);
        }
    }

    void App::select_ui_pass_image(std::string_view name)
    {
        if (!ui_pass)
        {
            throw std::logic_error(
                "can't select UI pass image if there's no UI pass"
            );
        }

        for (size_t i = 0; i < ui_pass->images().size(); i++)
        {
            if (ui_pass->images()[i].name != name)
            {
                continue;
            }
            selected_image_idx = i;
            break;
        }
    }

    static void glfw_error_callback(int error, const char* description)
    {
        std::cerr << fmt::format("GLFW error {}: {}\n", error, description);
    }

    static void imgui_check_vk_result(VkResult err)
    {
        if (err != VK_SUCCESS)
        {
            throw bv::Error("Dear ImGui Vulkan error", err, false);
        }
    }

}
