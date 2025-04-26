#include "vk_utils.hpp"

namespace img_aligner
{

    bv::CommandBufferPtr begin_single_time_commands(
        AppState& state,
        bool use_transient_pool
    )
    {
        auto cmd_buf = bv::CommandPool::allocate_buffer(
            state.cmd_pool(use_transient_pool),
            VK_COMMAND_BUFFER_LEVEL_PRIMARY
        );
        cmd_buf->begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
        return cmd_buf;
    }

    void end_single_time_commands(
        bv::CommandBufferPtr& cmd_buf,
        const bv::QueuePtr& queue,
        const bv::FencePtr& fence
    )
    {
        cmd_buf->end();

        queue->submit({}, {}, { cmd_buf }, {}, fence);
        if (fence == nullptr)
        {
            queue->wait_idle();
            cmd_buf = nullptr;
        }
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

    void copy_image_to_buffer(
        const bv::CommandBufferPtr& cmd_buf,
        const bv::ImagePtr& image,
        const bv::BufferPtr& buffer,
        VkDeviceSize buffer_offset
    )
    {
        VkBufferImageCopy copy_region{
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

        vkCmdCopyImageToBuffer(
            cmd_buf->handle(),
            image->handle(),
            VK_IMAGE_LAYOUT_GENERAL,
            buffer->handle(),
            1, &copy_region
        );
    }

    std::vector<float> read_back_image_rgbaf32(
        AppState& state,
        const bv::ImagePtr& image,
        const bv::QueuePtr& queue,
        bool vflip
    )
    {
        size_t width = image->config().extent.width;
        size_t height = image->config().extent.height;

        // verify format and figure out channel count and bit depth
        size_t n_channels = 0;
        size_t n_bytes_per_channel = 0;
        switch (image->config().format)
        {
        case VK_FORMAT_R32G32B32A32_SFLOAT:
            n_channels = 4;
            n_bytes_per_channel = sizeof(float);
            break;

        case VK_FORMAT_R32_SFLOAT:
            n_channels = 1;
            n_bytes_per_channel = sizeof(float);
            break;

        default:
            throw std::invalid_argument(fmt::format(
                "image format ({}) not supported for read back",
                string_VkFormat(image->config().format)
            ).c_str());
        }

        // size of the image in bytes
        VkDeviceSize size_bytes =
            width * height * n_channels * n_bytes_per_channel;

        // buffer to copy to
        bv::BufferPtr buf = nullptr;
        bv::MemoryChunkPtr buf_mem = nullptr;
        create_buffer(
            state,
            size_bytes,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,

            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,

            buf,
            buf_mem
        );
        float* buf_mapped = (float*)buf_mem->mapped();

        // copy to buffer
        auto fence = bv::Fence::create(state.device, 0);
        auto cmd_buf = begin_single_time_commands(state, true);
        copy_image_to_buffer(cmd_buf, image, buf);
        end_single_time_commands(cmd_buf, queue, fence);
        fence->wait();

        // convert to RGBA F32 and flip if needed, then store in a vector
        std::vector<float> pixels_rgbaf32(width * height * 4);
        if (image->config().format == VK_FORMAT_R32G32B32A32_SFLOAT)
        {
            if (vflip)
            {
                // copy row by row in reverse order
                for (size_t y = 0; y < height; y++)
                {
                    size_t src_red_idx = ((height - y - 1) * width) * 4;
                    size_t dst_red_idx = (y * width) * 4;

                    // copy an entire row at once
                    std::copy(
                        buf_mapped + src_red_idx,
                        buf_mapped + src_red_idx + (width * 4),
                        pixels_rgbaf32.data() + dst_red_idx
                    );
                }
            }
            else
            {
                std::copy(
                    buf_mapped,
                    buf_mapped + pixels_rgbaf32.size(),
                    pixels_rgbaf32.data()
                );
            }
        }
        else if (image->config().format == VK_FORMAT_R32_SFLOAT)
        {
            if (vflip)
            {
                for (size_t y = 0; y < height; y++)
                {
                    for (size_t x = 0; x < width; x++)
                    {
                        size_t src_pixel_idx = x + ((height - y - 1) * width);
                        size_t dst_red_idx = (x + (y * width)) * 4;

                        float v = buf_mapped[src_pixel_idx];
                        pixels_rgbaf32[dst_red_idx + 0] = v;
                        pixels_rgbaf32[dst_red_idx + 1] = v;
                        pixels_rgbaf32[dst_red_idx + 2] = v;
                        pixels_rgbaf32[dst_red_idx + 3] = 1.f;
                    }
                }
            }
            else
            {
                for (size_t y = 0; y < height; y++)
                {
                    for (size_t x = 0; x < width; x++)
                    {
                        size_t src_pixel_idx = x + (y * width);
                        size_t dst_red_idx = src_pixel_idx * 4;

                        float v = buf_mapped[src_pixel_idx];
                        pixels_rgbaf32[dst_red_idx + 0] = v;
                        pixels_rgbaf32[dst_red_idx + 1] = v;
                        pixels_rgbaf32[dst_red_idx + 2] = v;
                        pixels_rgbaf32[dst_red_idx + 3] = 1.f;
                    }
                }
            }
        }
        else
        {
            throw std::logic_error(
                "this should never happen because we already checked for the "
                "format above."
            );
        }

        return pixels_rgbaf32;
    }

    void generate_mipmaps(
        AppState& state,
        bv::CommandBufferPtr& cmd_buf,
        const bv::ImagePtr& image,
        bool use_general_layout,
        VkPipelineStageFlags next_stage_mask,
        VkAccessFlags next_stage_access_mask
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
            barrier.oldLayout = use_general_layout
                ? VK_IMAGE_LAYOUT_GENERAL
                : VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = use_general_layout
                ? VK_IMAGE_LAYOUT_GENERAL
                : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
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

                use_general_layout
                ? VK_IMAGE_LAYOUT_GENERAL
                : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,

                image->handle(),

                use_general_layout
                ? VK_IMAGE_LAYOUT_GENERAL
                : VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,

                1,
                &blit,
                VK_FILTER_LINEAR
            );

            barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier.dstAccessMask = next_stage_access_mask;
            barrier.oldLayout = use_general_layout
                ? VK_IMAGE_LAYOUT_GENERAL
                : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.newLayout = use_general_layout
                ? VK_IMAGE_LAYOUT_GENERAL
                : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.subresourceRange.baseMipLevel = i - 1;
            vkCmdPipelineBarrier(
                cmd_buf->handle(),
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                next_stage_mask,
                0,
                0, nullptr,
                0, nullptr,
                1, &barrier
            );

            if (mip_width > 1) mip_width /= 2;
            if (mip_height > 1) mip_height /= 2;
        }

        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = next_stage_access_mask;
        barrier.oldLayout = use_general_layout
            ? VK_IMAGE_LAYOUT_GENERAL
            : VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = use_general_layout
            ? VK_IMAGE_LAYOUT_GENERAL
            : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.subresourceRange.baseMipLevel = mip_levels - 1;
        vkCmdPipelineBarrier(
            cmd_buf->handle(),
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            next_stage_mask,
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
    )
    {
        if (width < 1 || height < 1)
        {
            throw std::invalid_argument(
                "texture size must be at least 1 in each dimension"
            );
        }
        if (size_bytes < 1)
        {
            throw std::invalid_argument(
                "texture pixel data size must be at least 1 byte"
            );
        }

        // create staging buffer and upload pixel data to it
        bv::BufferPtr staging_buf;
        bv::MemoryChunkPtr staging_buf_mem;
        create_buffer(
            state,
            size_bytes,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,

            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,

            staging_buf,
            staging_buf_mem
        );
        std::copy(
            (uint8_t*)pixels,
            (uint8_t*)pixels + size_bytes,
            (uint8_t*)staging_buf_mem->mapped()
        );
        staging_buf_mem->flush();

        // base and target images use the original resolution with mipmapping

        uint32_t mip_levels = 1;
        if (mipmapped)
        {
            mip_levels = round_log2(std::max(width, height));
        };

        create_image(
            state,
            width,
            height,
            mip_levels,
            VK_SAMPLE_COUNT_1_BIT,
            format,
            VK_IMAGE_TILING_OPTIMAL,

            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
            | VK_IMAGE_USAGE_SAMPLED_BIT,

            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            out_img,
            out_img_mem
        );
        out_imgview = create_image_view(
            state,
            out_img,
            format,
            VK_IMAGE_ASPECT_COLOR_BIT,
            mip_levels
        );

        auto cmd_buf = begin_single_time_commands(state, true);

        transition_image_layout(
            cmd_buf,
            out_img,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            mip_levels
        );

        copy_buffer_to_image(
            cmd_buf,
            staging_buf,
            out_img,
            0
        );

        if (mipmapped)
        {
            // generate mipmaps which will also transitions the image to
            // VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL.
            generate_mipmaps(
                state,
                cmd_buf,
                out_img,
                false,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                VK_ACCESS_SHADER_READ_BIT
            );
        }
        else
        {
            transition_image_layout(
                cmd_buf,
                out_img,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                mip_levels
            );
        }

        end_single_time_commands(cmd_buf, queue);

        staging_buf = nullptr;
        staging_buf_mem = nullptr;
    }

    const char* VkPhysicalDeviceType_to_str(VkPhysicalDeviceType v)
    {
        switch (v)
        {
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
            return "integrated GPU";
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
            return "discrete GPU";
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
            return "virtual GPU";
        case VK_PHYSICAL_DEVICE_TYPE_CPU:
            return "CPU";
        default:
            return "unknown device type";
        }
    }

}
