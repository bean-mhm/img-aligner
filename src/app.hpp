#pragma once

#include <string>
#include <vector>
#include <array>
#include <unordered_map>
#include <cstdint>

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

    struct UniformBufferObject
    {
        alignas(16) glm::mat4 model;
        alignas(16) glm::mat4 view;
        alignas(16) glm::mat4 proj;
    };

    struct Vertex
    {
        glm::vec3 pos;
        glm::vec2 texcoord;

        static const bv::VertexInputBindingDescription binding;
        bool operator==(const Vertex& other) const;
    };

    class App {
    public:
        App() = default;
        void run();

    private:
        static constexpr const char* TITLE =
            "img-aligner";
        static constexpr int INITIAL_WIDTH = 960;
        static constexpr int INITIAL_HEIGHT = 720;

        static constexpr bool DEBUG_MODE = true;
        static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;

        void init();
        void main_loop();
        void cleanup();

    private:
        GLFWwindow* window = nullptr;
        bv::ContextPtr context = nullptr;
        bv::DebugMessengerPtr debug_messenger = nullptr;
        bv::SurfacePtr surface = nullptr;
        std::optional<bv::PhysicalDevice> physical_device;
        bv::DevicePtr device = nullptr;
        bv::QueuePtr graphics_present_queue = nullptr;
        bv::MemoryBankPtr mem_bank = nullptr;
        bv::SwapchainPtr swapchain = nullptr;
        std::vector<bv::ImageViewPtr> swapchain_imgviews;

        bv::RenderPassPtr render_pass = nullptr;
        bv::DescriptorSetLayoutPtr descriptor_set_layout = nullptr;
        bv::PipelineLayoutPtr pipeline_layout = nullptr;
        bv::GraphicsPipelinePtr graphics_pipeline = nullptr;

        bv::CommandPoolPtr cmd_pool = nullptr;
        bv::CommandPoolPtr transient_cmd_pool = nullptr;

        bv::ImagePtr color_img = nullptr;
        bv::MemoryChunkPtr color_img_mem = nullptr;
        bv::ImageViewPtr color_imgview = nullptr;

        std::vector<bv::FramebufferPtr> swapchain_framebufs;

        uint32_t texture_mip_levels = 1;
        bv::ImagePtr texture_img = nullptr;
        bv::MemoryChunkPtr texture_img_mem = nullptr;
        bv::ImageViewPtr texture_imgview = nullptr;
        bv::SamplerPtr texture_sampler = nullptr;

        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices;

        bv::BufferPtr vertex_buf = nullptr;
        bv::MemoryChunkPtr vertex_buf_mem = nullptr;

        bv::BufferPtr index_buf = nullptr;
        bv::MemoryChunkPtr index_buf_mem = nullptr;

        std::vector<bv::BufferPtr> uniform_bufs;
        std::vector<bv::MemoryChunkPtr> uniform_bufs_mem;
        std::vector<void*> uniform_bufs_mapped;

        bv::DescriptorPoolPtr descriptor_pool = nullptr;
        std::vector<bv::DescriptorSetPtr> descriptor_sets;

        // "per frame" stuff (as in frames in flight)
        std::vector<bv::CommandBufferPtr> cmd_bufs;
        std::vector<bv::SemaphorePtr> semaphs_image_available;
        std::vector<bv::SemaphorePtr> semaphs_render_finished;
        std::vector<bv::FencePtr> fences_in_flight;

        uint32_t graphics_present_family_idx = 0;

        bool framebuf_resized = false;
        uint32_t frame_idx = 0;

        VkSampleCountFlagBits msaa_samples = VK_SAMPLE_COUNT_1_BIT;

        std::chrono::steady_clock::time_point start_time;

        void init_window();
        void init_context();
        void setup_debug_messenger();
        void create_surface();
        void pick_physical_device();
        void create_logical_device();
        void create_memory_bank();
        void create_swapchain();
        void create_render_pass();
        void create_descriptor_set_layout();
        void create_graphics_pipeline();
        void create_command_pools();
        void create_color_resources();
        void create_swapchain_framebuffers();
        void create_texture_image();
        void create_texture_sampler();
        void load_model();
        void create_vertex_buffer();
        void create_index_buffer();
        void create_uniform_buffers();
        void create_descriptor_pool();
        void create_descriptor_sets();
        void create_command_buffers();
        void create_sync_objects();

        void draw_frame();

        void cleanup_swapchain();
        void recreate_swapchain();

        // if use_transfer_pool is true, the command buffer will be allocated
        // from transfer_cmd_pool instead of cmd_pool. transfer_cmd_pool has the
        // VK_COMMAND_POOL_CREATE_TRANSIENT_BIT flag enabled which might be of
        // interest.
        bv::CommandBufferPtr begin_single_time_commands(
            bool use_transient_pool
        );

        // end and submit one-time command buffer. if no fence is provided,
        // Queue::wait_idle() will be used. if a fence is provided you'll be in
        // charge of synchronization (like waiting on the fence).
        void end_single_time_commands(
            bv::CommandBufferPtr& cmd_buf,
            const bv::FencePtr fence = nullptr
        );

        uint32_t find_memory_type_idx(
            uint32_t supported_type_bits,
            VkMemoryPropertyFlags required_properties
        );

        VkSampleCountFlagBits find_max_sample_count();

        void create_image(
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
            const bv::CommandBufferPtr& cmd_buf,
            const bv::ImagePtr& image,
            int32_t width,
            int32_t height,
            uint32_t mip_levels,
            bool vertex_shader = false
        );

        bv::ImageViewPtr create_image_view(
            const bv::ImagePtr& image,
            VkFormat format,
            VkImageAspectFlags aspect_flags,
            uint32_t mip_levels
        );

        void create_buffer(
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

        void record_command_buffer(
            const bv::CommandBufferPtr& cmd_buf,
            uint32_t img_idx,
            float elapsed
        );

        void update_uniform_buffer(uint32_t frame_idx, float elapsed);

        friend void glfw_framebuf_resize_callback(
            GLFWwindow* window,
            int width,
            int height
        );

    };

}
