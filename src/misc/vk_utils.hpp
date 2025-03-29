#pragma once

#include "common.hpp"
#include "app_state.hpp"
#include "numbers.hpp"

namespace img_aligner
{

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
