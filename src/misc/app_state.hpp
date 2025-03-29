#pragma once

#include "common.hpp"

namespace img_aligner
{

    struct AppState
    {
        // true means command line mode is enabled and the GUI is disabled
        bool cli_mode = false;

        GLFWwindow* window = nullptr;
        ImGuiIO* io = nullptr;

        bv::ContextPtr context = nullptr;
        bv::DebugMessengerPtr debug_messenger = nullptr;
        bv::SurfacePtr surface = nullptr;
        std::optional<bv::PhysicalDevice> physical_device;
        bv::DevicePtr device = nullptr;

        bv::QueuePtr queue_main;
        bv::QueuePtr queue_grid_warp_optimize;

        bv::MemoryBankPtr mem_bank = nullptr;

        // command pools for every thread
        std::unordered_map<std::thread::id, bv::CommandPoolPtr> cmd_pools;
        std::unordered_map<std::thread::id, bv::CommandPoolPtr>
            transient_cmd_pools;

        // lazy initialize the command pools so they are created on the right
        // thread based on std::this_thread::get_id().
        const bv::CommandPoolPtr& cmd_pool(bool transient);

        bv::DescriptorPoolPtr imgui_descriptor_pool = nullptr;
        uint32_t imgui_swapchain_min_image_count = 0;
        ImGui_ImplVulkanH_Window imgui_vk_window_data;
        bool imgui_swapchain_rebuild = false;
    };

}
