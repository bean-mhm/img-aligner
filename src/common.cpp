#include "common.hpp"

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#include <Windows.h>
#endif

namespace img_aligner
{

    double elapsed_sec(const std::chrono::steady_clock::time_point& t)
    {
        const auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(now - t).count();
    }

    double elapsed_sec(
        const std::optional<std::chrono::steady_clock::time_point>& t
    )
    {
        if (!t.has_value())
            return 0.;

        const auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(now - *t).count();
    }

    std::vector<uint8_t> read_file(const std::string& filename)
    {
        std::ifstream f(filename, std::ios::ate | std::ios::binary);
        if (!f.is_open())
        {
            throw std::runtime_error(std::format(
                "failed to read file \"{}\"",
                filename
            ).c_str());
        }

        size_t size_in_chars = (size_t)f.tellg();
        size_t size_in_bytes = size_in_chars * sizeof(char);

        std::vector<uint8_t> buf(size_in_bytes);
        f.seekg(0);
        f.read(reinterpret_cast<char*>(buf.data()), size_in_chars);
        f.close();

        return buf;
    }

    void open_url(std::string_view url)
    {
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
        ShellExecuteA(NULL, "open", url.data(), NULL, NULL, SW_SHOWNORMAL);
#elif __APPLE__
        std::system(std::format("open \"{}\"", url).c_str());
#elif __linux__
        std::system(std::format("xdg-open \"{}\"", url).c_str());
#else
        throw std::exception(std::format(
            "{} is not implemented for this platform",
            __FUNCTION__
        ).c_str());
#endif
    }

    uint32_t upper_power_of_2(uint32_t n)
    {
        if (n != 0)
        {
            n -= 1;
        }

        uint32_t i = 0;
        while (n > 0)
        {
            n >>= 1;
            i++;
        }
        return (uint32_t)1 << i;
    }

    uint32_t round_log2(uint32_t n)
    {
        uint32_t i = 0;
        while (n > 0)
        {
            n >>= 1;
            i++;
        }
        return i;
    }

    bv::CommandBufferPtr begin_single_time_commands(
        AppState& state,
        bool use_transient_pool
    )
    {
        auto cmd_buf = bv::CommandPool::allocate_buffer(
            use_transient_pool ? state.transient_cmd_pool : state.cmd_pool,
            VK_COMMAND_BUFFER_LEVEL_PRIMARY
        );
        cmd_buf->begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
        return cmd_buf;
    }

    void end_single_time_commands(
        AppState& state,
        bv::CommandBufferPtr& cmd_buf,
        const bv::FencePtr fence
    )
    {
        cmd_buf->end();
        state.queue->submit({}, {}, { cmd_buf }, {}, fence);
        if (fence == nullptr)
        {
            state.queue->wait_idle();
        }
        cmd_buf = nullptr;
    }

    uint32_t find_memory_type_idx(
        AppState& state,
        uint32_t supported_type_bits,
        VkMemoryPropertyFlags required_properties
    )
    {
        const auto& mem_props = state.physical_device->memory_properties();
        for (uint32_t i = 0; i < mem_props.memory_types.size(); i++)
        {
            bool has_required_properties =
                (required_properties & mem_props.memory_types[i].property_flags)
                == required_properties;

            if ((supported_type_bits & (1 << i)) && has_required_properties)
            {
                return i;
            }
        }
        throw std::runtime_error("failed to find a suitable memory type");
    }

    VkSampleCountFlagBits find_max_sample_count(AppState& state)
    {
        const auto& limits = state.physical_device.value().properties().limits;
        VkSampleCountFlags counts =
            limits.framebuffer_color_sample_counts
            & limits.framebuffer_depth_sample_counts;

        // 64 samples is insanely high
        //if (counts & VK_SAMPLE_COUNT_64_BIT) return VK_SAMPLE_COUNT_64_BIT;
        //if (counts & VK_SAMPLE_COUNT_32_BIT) return VK_SAMPLE_COUNT_32_BIT;
        if (counts & VK_SAMPLE_COUNT_16_BIT) return VK_SAMPLE_COUNT_16_BIT;
        if (counts & VK_SAMPLE_COUNT_8_BIT) return VK_SAMPLE_COUNT_8_BIT;
        if (counts & VK_SAMPLE_COUNT_4_BIT) return VK_SAMPLE_COUNT_4_BIT;
        if (counts & VK_SAMPLE_COUNT_2_BIT) return VK_SAMPLE_COUNT_2_BIT;
        return VK_SAMPLE_COUNT_1_BIT;
    }

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
    )
    {
        bv::Extent3d extent{
            .width = width,
            .height = height,
            .depth = 1
        };
        out_image = bv::Image::create(
            state.device,
            {
                .flags = 0,
                .image_type = VK_IMAGE_TYPE_2D,
                .format = format,
                .extent = extent,
                .mip_levels = mip_levels,
                .array_layers = 1,
                .samples = num_samples,
                .tiling = tiling,
                .usage = usage,
                .sharing_mode = VK_SHARING_MODE_EXCLUSIVE,
                .queue_family_indices = {},
                .initial_layout = VK_IMAGE_LAYOUT_UNDEFINED
            }
        );

        out_memory_chunk = state.mem_bank->allocate(
            out_image->memory_requirements(),
            memory_properties
        );
        out_memory_chunk->bind(out_image);
    }

    void transition_image_layout(
        const bv::CommandBufferPtr& cmd_buf,
        const bv::ImagePtr& image,
        VkImageLayout old_layout,
        VkImageLayout new_layout,
        uint32_t mip_levels
    )
    {
        VkAccessFlags src_access_mask = 0;
        VkAccessFlags dst_access_mask = 0;
        VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

        if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED
            && new_layout == VK_IMAGE_LAYOUT_GENERAL)
        {
            src_access_mask = 0;
            dst_access_mask = 0;

            src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            dst_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        }
        else if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED
            && new_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
        {
            src_access_mask = 0;
            dst_access_mask = VK_ACCESS_TRANSFER_WRITE_BIT;

            src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        }
        else if (old_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
            && new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        {
            src_access_mask = VK_ACCESS_TRANSFER_WRITE_BIT;
            dst_access_mask = VK_ACCESS_SHADER_READ_BIT;

            src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            dst_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }
        else if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED
            && new_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        {
            src_access_mask = 0;
            dst_access_mask =
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT
                | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

            src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            dst_stage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        }
        else
        {
            throw std::invalid_argument("unsupported image layout transition");
        }

        VkImageAspectFlags subresource_aspect_mask = VK_IMAGE_ASPECT_COLOR_BIT;
        if (new_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        {
            subresource_aspect_mask = VK_IMAGE_ASPECT_DEPTH_BIT;
            if (bv::format_has_stencil_component(image->config().format))
            {
                subresource_aspect_mask |= VK_IMAGE_ASPECT_STENCIL_BIT;
            }
        }

        VkImageMemoryBarrier barrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = src_access_mask,
            .dstAccessMask = dst_access_mask,
            .oldLayout = old_layout,
            .newLayout = new_layout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image->handle(),
            .subresourceRange = VkImageSubresourceRange{
                .aspectMask = subresource_aspect_mask,
                .baseMipLevel = 0,
                .levelCount = mip_levels,
                .baseArrayLayer = 0,
                .layerCount = 1
        }
        };

        vkCmdPipelineBarrier(
            cmd_buf->handle(),
            src_stage,
            dst_stage,
            0,
            0, nullptr,
            0, nullptr,
            1, &barrier
        );
    }

    void copy_buffer_to_image(
        const bv::CommandBufferPtr& cmd_buf,
        const bv::BufferPtr& buffer,
        const bv::ImagePtr& image,
        VkDeviceSize buffer_offset
    )
    {
        VkBufferImageCopy region{
            .bufferOffset = buffer_offset,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource = VkImageSubresourceLayers{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1
        },
            .imageOffset = { 0, 0, 0 },
            .imageExtent = bv::Extent3d_to_vk(image->config().extent)
        };

        vkCmdCopyBufferToImage(
            cmd_buf->handle(),
            buffer->handle(),
            image->handle(),
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &region
        );
    }

    void generate_mipmaps(
        AppState& state,
        const bv::CommandBufferPtr& cmd_buf,
        const bv::ImagePtr& image
    )
    {
        // check if the image format supports linear blitting
        auto format_props = state.physical_device->fetch_format_properties(
            image->config().format
        );
        if (!(format_props.optimal_tiling_features
            & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT))
        {
            throw std::runtime_error(
                "image format does not support linear blitting"
            );
        }

        VkImageMemoryBarrier barrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = 0, // will be changed below
            .dstAccessMask = 0, // will be changed below
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, // will be changed below
            .newLayout = VK_IMAGE_LAYOUT_UNDEFINED, // will be changed below
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image->handle(),
            .subresourceRange = VkImageSubresourceRange{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0, // will be changed below
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
        }
        };

        uint32_t mip_levels = image->config().mip_levels;

        int32_t mip_width = (int32_t)image->config().extent.width;
        int32_t mip_height = (int32_t)image->config().extent.height;

        for (uint32_t i = 1; i < mip_levels; i++)
        {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.subresourceRange.baseMipLevel = i - 1;
            vkCmdPipelineBarrier(
                cmd_buf->handle(),
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                0, nullptr,
                0, nullptr,
                1, &barrier
            );

            VkImageBlit blit{};
            blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            blit.srcSubresource.mipLevel = i - 1;
            blit.srcSubresource.baseArrayLayer = 0;
            blit.srcSubresource.layerCount = 1;
            blit.srcOffsets[0] = { 0, 0, 0 };
            blit.srcOffsets[1] = { mip_width, mip_height, 1 };
            blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            blit.dstSubresource.mipLevel = i;
            blit.dstSubresource.baseArrayLayer = 0;
            blit.dstSubresource.layerCount = 1;
            blit.dstOffsets[0] = { 0, 0, 0 };
            blit.dstOffsets[1] = {
                mip_width > 1 ? mip_width / 2 : 1,
                mip_height > 1 ? mip_height / 2 : 1,
                1
            };
            vkCmdBlitImage(
                cmd_buf->handle(),
                image->handle(),
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                image->handle(),
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1,
                &blit,
                VK_FILTER_LINEAR
            );

            barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.subresourceRange.baseMipLevel = i - 1;
            vkCmdPipelineBarrier(
                cmd_buf->handle(),
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                0,
                0, nullptr,
                0, nullptr,
                1, &barrier
            );

            if (mip_width > 1) mip_width /= 2;
            if (mip_height > 1) mip_height /= 2;
        }

        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.subresourceRange.baseMipLevel = mip_levels - 1;
        vkCmdPipelineBarrier(
            cmd_buf->handle(),
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &barrier
        );
    }

    bv::ImageViewPtr create_image_view(
        AppState& state,
        const bv::ImagePtr& image,
        VkFormat format,
        VkImageAspectFlags aspect_flags,
        uint32_t mip_levels
    )
    {
        bv::ImageSubresourceRange subresource_range{
            .aspect_mask = aspect_flags,
            .base_mip_level = 0,
            .level_count = mip_levels,
            .base_array_layer = 0,
            .layer_count = 1
        };

        return bv::ImageView::create(
            state.device,
            image,
            {
                .flags = 0,
                .view_type = VK_IMAGE_VIEW_TYPE_2D,
                .format = format,
                .components = {},
                .subresource_range = subresource_range
            }
        );
    }

    void create_buffer(
        AppState& state,
        VkDeviceSize size,
        VkBufferUsageFlags usage,
        VkMemoryPropertyFlags memory_properties,
        bv::BufferPtr& out_buffer,
        bv::MemoryChunkPtr& out_memory_chunk
    )
    {
        out_buffer = bv::Buffer::create(
            state.device,
            {
                .flags = 0,
                .size = size,
                .usage = usage,
                .sharing_mode = VK_SHARING_MODE_EXCLUSIVE,
                .queue_family_indices = {}
            }
        );

        out_memory_chunk = state.mem_bank->allocate(
            out_buffer->memory_requirements(),
            memory_properties
        );
        out_memory_chunk->bind(out_buffer);
    }

    void copy_buffer(
        const bv::CommandBufferPtr& cmd_buf,
        bv::BufferPtr src,
        bv::BufferPtr dst,
        VkDeviceSize size
    )
    {
        VkBufferCopy copy_region{
            .srcOffset = 0,
            .dstOffset = 0,
            .size = size
        };
        vkCmdCopyBuffer(
            cmd_buf->handle(),
            src->handle(),
            dst->handle(),
            1,
            &copy_region
        );
    }

}
