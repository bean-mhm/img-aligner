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
#include <shared_mutex>
#include <thread>
#include <atomic>
#include <unordered_map>
#include <set>
#include <limits>
#include <algorithm>
#include <optional>
#include <chrono>
#include <random>
#include <type_traits>
#include <stdexcept>
#include <cmath>
#include <cstdlib>
#include <cstdint>

#include "CLI11/CLI11.hpp"

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_vulkan.h"

#include "vulkan/vulkan.h"
#include "vulkan/vk_enum_string_helper.h"
#include "GLFW/glfw3.h"

#include "beva/beva.hpp"

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

#include "circular_buffer.hpp"
#include "transform2d.hpp"

// access element in 2D array with row-major ordering
#define ACCESS_2D(arr, ix, iy, res_x) ((arr)[(ix) + (iy) * (res_x)])
#define INDEX_2D(ix, iy, res_x) ((ix) + (iy) * (res_x))

namespace img_aligner
{

    static constexpr auto APP_TITLE = "img-aligner";
    static constexpr auto APP_VERSION = "0.1.0-dev";
    static constexpr auto APP_GITHUB_URL =
        "https://github.com/bean-mhm/img-aligner";

    static constexpr uint32_t INITIAL_WIDTH = 1024;
    static constexpr uint32_t INITIAL_HEIGHT = 720;

    static constexpr ImVec4 COLOR_BG{ .04f, .03f, .08f, 1.f };
    static constexpr ImVec4 COLOR_IMAGE_BORDER{ .05f, .05f, .05f, 1.f };
    static constexpr ImVec4 COLOR_INFO_TEXT{ .33f, .74f, .91f, 1.f };
    static constexpr ImVec4 COLOR_WARNING_TEXT{ .94f, .58f, .28f, 1.f };
    static constexpr ImVec4 COLOR_ERROR_TEXT{ .95f, .3f, .23f, 1.f };

    static constexpr float FONT_SIZE = 20.f;
    static constexpr auto FONT_PATH = "fonts/Outfit-Regular.ttf";
    static constexpr auto FONT_BOLD_PATH = "fonts/Outfit-Bold.ttf";

    static constexpr VkFormat RGBA_FORMAT = VK_FORMAT_R32G32B32A32_SFLOAT;
    static constexpr VkFormat R_FORMAT = VK_FORMAT_R32_SFLOAT;
    static constexpr VkFormat UI_DISPLAY_IMG_FORMAT =
        VK_FORMAT_R16G16B16A16_SFLOAT;

    // enable Vulkan validation layer and debug messages. this is usually only
    // available if the Vulkan SDK is installed on the user's machine, which is
    // rarely the case for a regular user, so disable this for final releases.
    static constexpr bool ENABLE_VALIDATION_LAYER = false;

    // the interval at which to run the UI pass and make a new copy of the grid
    // vertices for previewing in the UI, when grid warp optimization is
    // running.
    static constexpr float GRID_WARP_OPTIMIZATION_UI_UPDATE_INTERVAL = .7f;

    static constexpr float GRID_WARP_OPTIMIZATION_CLI_REALTIME_STATS_INTERVAL =
        .3f;

    static constexpr size_t GRID_WARP_OPTIMIZATION_WARP_STRENGTH_PLOT_N_ITERS
        = 5000;

    using TimePoint = std::chrono::high_resolution_clock::time_point;

    class ScopedTimer
    {
    public:
        ScopedTimer(
            bool should_print = true,
            std::string start_message = "processing",
            std::string end_message = " ({} s)\n"
        );
        ~ScopedTimer();

    private:
        TimePoint start_time;

        bool should_print;
        std::string end_message;

    };

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

    double elapsed_sec(const TimePoint& t);
    double elapsed_sec(const std::optional<TimePoint>& t);

    const std::filesystem::path& exec_dir(
        const std::filesystem::path& new_value = {}
    );
    std::vector<uint8_t> read_file(const std::filesystem::path& path);
    void open_url(std::string_view url);
    void clear_console();

    // closest upper power of 2 to an integer.
    // examples: 11 -> 16, 3000 -> 4096, 256 -> 256
    uint32_t upper_power_of_2(uint32_t n);

    // how many times an integer can be divided by 2 (or bit-shifted to the
    // right) until it reaches 0.
    uint32_t round_log2(uint32_t n);

    template<std::floating_point T>
    int32_t determine_precision(
        T v,
        int32_t max_significant_digits = 5, // can only reduce decimal digits
        int32_t min_precision = 1,
        int32_t max_precision = 11
    )
    {
        // this may be negative and it's intentional
        int32_t n_integral_digits = (int32_t)std::floor(std::log10(v)) + 1;

        min_precision = std::max(min_precision, 0);
        max_precision = std::max(max_precision, min_precision);

        return std::clamp(
            max_significant_digits - n_integral_digits,
            min_precision,
            max_precision
        );
    }

    template<std::integral T>
    std::string to_str(T v)
    {
        return std::to_string(v);
    }

    template<std::floating_point T>
    std::string to_str(
        T v,
        int32_t max_significant_digits = 5, // can only reduce decimal digits
        int32_t min_precision = 1,
        int32_t max_precision = 11,
        int32_t* out_precision = nullptr,
        int32_t* out_n_trailing_zeros = nullptr
    )
    {
        int32_t precision = determine_precision(
            v,
            max_significant_digits,
            min_precision,
            max_precision
        );

        std::string s = std::format("{0:.{1}f}", v, precision);

        // remove redundant trailing zeros after decimal point
        int32_t n_trailing_zeros = 0;
        if (precision > 0)
        {
            while (s.ends_with('0'))
            {
                s = s.substr(0, s.length() - 1);
                n_trailing_zeros++;
            }
            if (s.ends_with('.'))
            {
                s = s.substr(0, s.length() - 1);
            }
        }

        if (out_precision)
        {
            *out_precision = precision;
        }
        if (out_n_trailing_zeros)
        {
            *out_n_trailing_zeros = n_trailing_zeros;
        }

        if (s == "-0")
        {
            s = "0";
        }

        return s;
    }

    // use higher precision for floats
    template<std::floating_point T>
    std::string to_str_hp(T v)
    {
        return to_str(v, 15, 0, 20);
    }

    // use highest precision for floats
    template<std::floating_point T>
    std::string to_str_hhp(T v)
    {
        std::string s = std::format(
            std::is_same_v<T, float> ? "{0:.50f}" : "{0:.326f}",
            v
        );

        // remove redundant zeros after decimal point
        while (s.ends_with('0'))
        {
            s = s.substr(0, s.length() - 1);
        }
        if (s.ends_with('.'))
        {
            s = s.substr(0, s.length() - 1);
        }

        if (s == "-0")
        {
            s = "0";
        }

        return s;
    }

    // no difference for non-float types
    template<std::integral T>
    std::string to_str_hp(T v)
    {
        return std::to_string(v);
    }

    // no difference for non-float types
    template<std::integral T>
    std::string to_str_hhp(T v)
    {
        return std::to_string(v);
    }

    // unused because ImGui is buggy
    template<std::floating_point T>
    std::string determine_precision_for_imgui(
        T v,
        int32_t max_significant_digits = 5, // can only reduce decimal digits
        int32_t min_precision = 1,
        int32_t max_precision = 11
    )
    {
        int32_t precision = 0;
        int32_t n_trailing_zeros = 0;
        std::string s = to_str(
            v,
            max_significant_digits,
            min_precision,
            max_precision,
            &precision,
            &n_trailing_zeros
        );

        // don't remove literally all trailing zeros for ImGui because it bugs
        // out sometimes.
        return std::format(
            "%.{}f",
            std::max(precision - n_trailing_zeros, min_precision)
        );
    }

    // linear interpolation
    template<typename V, std::floating_point T>
    T lerp(const V& a, const V& b, T t)
    {
        return a + t * (b - a);
    }

    template<typename T, T(*exp_fn)(T) = std::exp>
    T unnormalized_gaussian(T standard_deviation, T x)
    {
        T a = x / standard_deviation;
        return exp_fn((T)(-.5) * a * a);
    }

    constexpr ImVec2 imvec_from_glm(const glm::vec2& v)
    {
        return { v.x, v.y };
    }

    constexpr glm::vec2 imvec_to_glm(const ImVec2& v)
    {
        return { v.x, v.y };
    }

    constexpr bool vec2_is_outside_01(const glm::vec2& v)
    {
        return v.x < 0.f || v.y < 0.f || v.x > 1.f || v.y > 1.f;
    }

    // zero out a vector's capacity to actually free the memory
    template<typename T>
    void clear_vec(std::vector<T>& vec)
    {
        vec.clear();
        std::vector<T>().swap(vec);
    }

    static void cli_add_toggle(
        CLI::App& cli_app,
        const std::string& flag_name,
        bool& flag_result,
        const std::string& flag_description = "toggle flag"
    )
    {
        cli_app.add_flag_callback(
            flag_name,
            [&flag_result]()
            {
                flag_result = !flag_result;
            },
            std::format("{} (default: {})", flag_description, flag_result)
        );
    }

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
        bv::CommandBufferPtr& cmd_buf,
        const bv::QueuePtr& queue,
        const bv::FencePtr& fence = nullptr
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

    void copy_image_to_buffer(
        const bv::CommandBufferPtr& cmd_buf,
        const bv::ImagePtr& image,
        const bv::BufferPtr& buffer,
        VkDeviceSize buffer_offset = 0
    );

    // returns pixels in the RGBA F32 format (performs conversions if needed)
    std::vector<float> read_back_image_rgbaf32(
        AppState& state,
        const bv::ImagePtr& image,
        const bv::QueuePtr& queue,
        bool vflip // flip vertically
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
        const bv::QueuePtr& queue,
        uint32_t width,
        uint32_t height,
        VkFormat format,
        void* pixels,
        size_t size_bytes,
        bool mipmapped,
        bv::ImagePtr& out_img,
        bv::MemoryChunkPtr& out_img_mem,
        bv::ImageViewPtr& out_imgview
    );

    const char* VkPhysicalDeviceType_to_str(VkPhysicalDeviceType v);

}
