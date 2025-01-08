#pragma once

#include <iostream>
#include <fstream>
#include <format>
#include <string>
#include <vector>
#include <array>
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

namespace img_aligner
{

    static constexpr auto WINDOW_TITLE = "img-aligner - bean-mhm";
    static constexpr uint32_t INITIAL_WIDTH = 960;
    static constexpr uint32_t INITIAL_HEIGHT = 720;
    static constexpr ImVec4 COLOR_BG{ .04f, .03f, .08f, 1.f };

    static constexpr float FONT_SIZE = 24.f;
    static constexpr auto FONT_PATH = "./fonts/Outfit-Regular.ttf";
    static constexpr auto FONT_BOLD_PATH = "./fonts/Outfit-Bold.ttf";

    static constexpr bool DEBUG_MODE = true;
    static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;
    static constexpr VkSampleCountFlagBits MSAA_LEVEL = VK_SAMPLE_COUNT_8_BIT;

    struct AppState
    {
        GLFWwindow* window = nullptr;
        ImGuiIO* io = nullptr;
        ImFont* font = nullptr;
        ImFont* font_bold = nullptr;

        bv::ContextPtr context = nullptr;
        bv::DebugMessengerPtr debug_messenger = nullptr;
        bv::SurfacePtr surface = nullptr;
        std::optional<bv::PhysicalDevice> physical_device;
        bv::DevicePtr device = nullptr;
        bv::QueuePtr queue = nullptr;
        bv::MemoryBankPtr mem_bank = nullptr;

        bv::CommandPoolPtr cmd_pool = nullptr;
        bv::CommandPoolPtr transient_cmd_pool = nullptr;

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

    // if use_transfer_pool is true, the command buffer will be allocated
        // from transfer_cmd_pool instead of cmd_pool. transfer_cmd_pool has the
        // VK_COMMAND_POOL_CREATE_TRANSIENT_BIT flag enabled which might be of
        // interest.
    bv::CommandBufferPtr begin_single_time_commands(
        AppState& state,
        bool use_transient_pool
    );

    // end and submit one-time command buffer. if no fence is provided,
    // Queue::wait_idle() will be used. if a fence is provided you'll be in
    // charge of synchronization (like waiting on the fence).
    void end_single_time_commands(
        AppState& state,
        bv::CommandBufferPtr& cmd_buf,
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

    // when new_layout is VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, the
    // vertex_shader argument defines whether dstStageMask should be set to
    // VK_PIPELINE_STAGE_VERTEX_SHADER_BIT or
    // VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, otherwise it's ignored.
    void transition_image_layout(
        const bv::CommandBufferPtr& cmd_buf,
        const bv::ImagePtr& image,
        VkImageLayout old_layout,
        VkImageLayout new_layout,
        uint32_t mip_levels,
        bool vertex_shader = false
    );

    void copy_buffer_to_image(
        const bv::CommandBufferPtr& cmd_buf,
        const bv::BufferPtr& buffer,
        const bv::ImagePtr& image,
        uint32_t width,
        uint32_t height
    );

    // if vertex_shader == true then dstStageMask will be set to
    // VK_PIPELINE_STAGE_VERTEX_SHADER_BIT instead of
    // VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT in vkCmdPipelineBarrier().
    // this means that the vertex shader will wait for the image to be ready
    // because the image will be used in the vertex shader.
    void generate_mipmaps(
        AppState& state,
        const bv::CommandBufferPtr& cmd_buf,
        const bv::ImagePtr& image,
        int32_t width,
        int32_t height,
        uint32_t mip_levels,
        bool vertex_shader = false
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

}
