#pragma once

#include <iostream>
#include <fstream>
#include <filesystem>
#include <format>
#include <string>
#include <vector>
#include <array>
#include <span>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <set>
#include <limits>
#include <algorithm>
#include <optional>
#include <chrono>
#include <stdexcept>
#include <cmath>
#include <cstdlib>
#include <cstdint>

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_vulkan.h"

#include "vulkan/vulkan.h"
#include "GLFW/glfw3.h"

#include "beva/beva.hpp"

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

// access element in 2D array with row-major ordering
#define ACCESS_2D(arr, ix, iy, res_x) ((arr)[(ix) + (iy) * (res_x)])
#define INDEX_2D(ix, iy, res_x) ((ix) + (iy) * (res_x))

namespace img_aligner
{

    static constexpr auto APP_TITLE = "img-aligner";
    static constexpr auto APP_VERSION = "0.1.0-dev";
    static constexpr auto APP_GITHUB_URL =
        "https://github.com/bean-mhm/img-aligner";

    static constexpr uint32_t INITIAL_WIDTH = 960;
    static constexpr uint32_t INITIAL_HEIGHT = 720;

    static constexpr ImVec4 COLOR_BG{ .04f, .03f, .08f, 1.f };
    static constexpr ImVec4 COLOR_IMAGE_BORDER{ .05f, .05f, .05f, 1.f };
    static constexpr ImVec4 COLOR_INFO_TEXT{ .33f, .74f, .91f, 1.f };
    static constexpr ImVec4 COLOR_WARNING_TEXT{ .94f, .58f, .28f, 1.f };
    static constexpr ImVec4 COLOR_ERROR_TEXT{ .95f, .3f, .23f, 1.f };

    static constexpr float FONT_SIZE = 20.f;
    static constexpr auto FONT_PATH = "./fonts/Outfit-Regular.ttf";
    static constexpr auto FONT_BOLD_PATH = "./fonts/Outfit-Bold.ttf";

    static constexpr VkFormat RGBA_FORMAT = VK_FORMAT_R32G32B32A32_SFLOAT;
    static constexpr VkFormat R_FORMAT = VK_FORMAT_R32_SFLOAT;
    static constexpr VkFormat UI_DISPLAY_IMG_FORMAT =
        VK_FORMAT_R16G16B16A16_SFLOAT;

    static constexpr bool DEBUG_MODE = true;

    // total number of threads that create or submit command buffers
    static constexpr size_t N_THREADS = 2;

    struct AppState
    {
        GLFWwindow* window = nullptr;
        ImGuiIO* io = nullptr;

        bv::ContextPtr context = nullptr;
        bv::DebugMessengerPtr debug_messenger = nullptr;
        bv::SurfacePtr surface = nullptr;
        std::optional<bv::PhysicalDevice> physical_device;
        bv::DevicePtr device = nullptr;

        bv::QueuePtr queue = nullptr;
        std::mutex queue_mutex;

        bv::MemoryBankPtr mem_bank = nullptr;

        // command pools for each thread (size: N_THREADS)
        std::vector<bv::CommandPoolPtr> cmd_pools;
        std::vector<bv::CommandPoolPtr> transient_cmd_pools;

        bv::DescriptorPoolPtr imgui_descriptor_pool = nullptr;
        uint32_t imgui_swapchain_min_image_count = 0;
        ImGui_ImplVulkanH_Window imgui_vk_window_data;
        bool imgui_swapchain_rebuild = false;
    };

    double elapsed_sec(const std::chrono::steady_clock::time_point& t);
    double elapsed_sec(
        const std::optional<std::chrono::steady_clock::time_point>& t
    );

    std::vector<uint8_t> read_file(const std::string& filename);

    void open_url(std::string_view url);

    // closest upper power of 2 to an integer.
    // examples: 11 -> 16, 3000 -> 4096, 256 -> 256
    uint32_t upper_power_of_2(uint32_t n);

    // how many times an integer can be divided by 2 (or bit-shifted to the
    // right) until it reaches 0.
    uint32_t round_log2(uint32_t n);

    // if use_transfer_pool is true, the command buffer will be allocated
    // from transfer_cmd_pool instead of cmd_pool. transfer_cmd_pool has the
    // VK_COMMAND_POOL_CREATE_TRANSIENT_BIT flag enabled which might be of
    // interest.
    bv::CommandBufferPtr begin_single_time_commands(
        AppState& state,
        bool use_transient_pool,
        size_t thread_idx
    );

    // end and submit one-time command buffer. if no fence is provided,
    // Queue::wait_idle() will be used. if a fence is provided you'll be in
    // charge of synchronization (like waiting on the fence).
    void end_single_time_commands(
        AppState& state,
        bv::CommandBufferPtr& cmd_buf,
        bool lock_queue_mutex,
        const bv::FencePtr fence = nullptr
    );

    uint32_t find_memory_type_idx(
        AppState& state,
        uint32_t supported_type_bits,
        VkMemoryPropertyFlags required_properties
    );

    VkSampleCountFlagBits find_max_sample_count(AppState& state);

    void create_image(
        AppState& state,
        uint32_t width,
        uint32_t height,
        uint32_t mip_levels,
        VkSampleCountFlagBits num_samples,
        VkFormat format,
        VkImageTiling tiling,
        VkImageUsageFlags usage,
        VkMemoryPropertyFlags memory_properties,
        bv::ImagePtr& out_image,
        bv::MemoryChunkPtr& out_memory_chunk
    );

    void transition_image_layout(
        const bv::CommandBufferPtr& cmd_buf,
        const bv::ImagePtr& image,
        VkImageLayout old_layout,
        VkImageLayout new_layout,
        uint32_t mip_levels
    );

    void copy_buffer_to_image(
        const bv::CommandBufferPtr& cmd_buf,
        const bv::BufferPtr& buffer,
        const bv::ImagePtr& image,
        VkDeviceSize buffer_offset = 0
    );

    // if use_general_layout is true, the image is expected to be in
    // VK_IMAGE_LAYOUT_GENERAL and no layout transitions will happen. otherwise,
    // the image must be in VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL and it will be
    // transitioned to VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL at the end.
    //
    // next_stage_mask defines the upcoming pipeline stages that should wait for
    // the mipmap operation to finish.
    //
    // next_stage_access_mask defines what operation in the upcoming stage will
    // wait for the mipmap operation to finish.
    void generate_mipmaps(
        AppState& state,
        bv::CommandBufferPtr& cmd_buf,
        const bv::ImagePtr& image,
        bool use_general_layout,
        VkPipelineStageFlags next_stage_mask,
        VkAccessFlags next_stage_access_mask
    );

    bv::ImageViewPtr create_image_view(
        AppState& state,
        const bv::ImagePtr& image,
        VkFormat format,
        VkImageAspectFlags aspect_flags,
        uint32_t mip_levels
    );

    void create_buffer(
        AppState& state,
        VkDeviceSize size,
        VkBufferUsageFlags usage,
        VkMemoryPropertyFlags memory_properties,
        bv::BufferPtr& out_buffer,
        bv::MemoryChunkPtr& out_memory_chunk
    );

    void copy_buffer(
        const bv::CommandBufferPtr& cmd_buf,
        bv::BufferPtr src,
        bv::BufferPtr dst,
        VkDeviceSize size
    );

    void create_texture(
        AppState& state,
        uint32_t width,
        uint32_t height,
        VkFormat format,
        void* pixels,
        size_t size_bytes,
        bool mipmapped,
        size_t thread_idx,
        bv::ImagePtr& out_img,
        bv::MemoryChunkPtr& out_img_mem,
        bv::ImageViewPtr& out_imgview
    );

}
