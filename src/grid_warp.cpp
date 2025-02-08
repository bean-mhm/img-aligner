#include "grid_warp.hpp"

namespace img_aligner::grid_warp
{

    const bv::VertexInputBindingDescription GridVertex::binding{
            .binding = 0,
            .stride = sizeof(GridVertex),
            .input_rate = VK_VERTEX_INPUT_RATE_VERTEX
    };

    bool GridVertex::operator==(const GridVertex& other) const
    {
        return
            warped_pos == other.warped_pos
            && orig_pos == other.orig_pos;
    }

    static const std::vector<bv::VertexInputAttributeDescription>
        attributes
    {
        bv::VertexInputAttributeDescription{
            .location = 0,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32_SFLOAT,
            .offset = offsetof(GridVertex, warped_pos)
    },
        bv::VertexInputAttributeDescription{
            .location = 1,
            .binding = 0,
            .format = VK_FORMAT_R32G32_SFLOAT,
            .offset = offsetof(GridVertex, orig_pos)
    }
    };

    GridWarper::GridWarper(
        AppState& state,
        const Params& params
    )
        : state(state),
        img_width(params.img_width),
        img_height(params.img_height)
    {
        if (img_width < 1 || img_height < 1)
        {
            throw std::invalid_argument(
                "image resolution must be at least 1 in every axis"
            );
        }

        uint32_t image_res_min = std::min(img_width, img_height);
        uint32_t image_res_max = std::max(img_width, img_height);
        double aspect_ratio = (double)image_res_max / (double)image_res_min;

        // figure out the intermediate resolution
        uint32_t actual_intermediate_res = std::clamp(
            params.intermediate_res_smallest_axis,
            (uint32_t)1,
            image_res_min
        );
        if (img_width < img_height)
        {
            intermediate_res_x = actual_intermediate_res;
            intermediate_res_y = (uint32_t)std::ceil(
                (double)actual_intermediate_res * aspect_ratio
            );
        }
        else
        {
            intermediate_res_y = actual_intermediate_res;
            intermediate_res_x = (uint32_t)std::ceil(
                (double)actual_intermediate_res * aspect_ratio
            );
        }

        // figure out the grid resolution
        if (params.grid_res_smallest_axis < 1)
        {
            grid_res_x = 1;
            grid_res_y = 1;
        }
        else if (img_width < img_height)
        {
            grid_res_x = params.grid_res_smallest_axis;
            grid_res_y = (uint32_t)std::ceil(
                (double)params.grid_res_smallest_axis * aspect_ratio
            );
        }
        else
        {
            grid_res_y = params.grid_res_smallest_axis;
            grid_res_x = (uint32_t)std::ceil(
                (double)params.grid_res_smallest_axis * aspect_ratio
            );
        }

        // figure out the padded grid resolution. basically we add a border on
        // the outside for padding. without the bordering cells, we might get
        // black empty spaces in the warped image.
        double actual_grid_padding = (double)std::max(params.grid_padding, 0.f);
        uint32_t grid_res_max = std::max(grid_res_x, grid_res_y);
        padded_grid_res_x = (uint32_t)std::ceil(
            (double)grid_res_x
            + (2. * actual_grid_padding * (double)grid_res_max)
        );
        padded_grid_res_y = (uint32_t)std::ceil(
            (double)grid_res_y
            + (2. * actual_grid_padding * (double)grid_res_max)
        );

        // padded resolution must be original resolution plus an even number
        if ((padded_grid_res_x - grid_res_x) % 2 != 0)
            padded_grid_res_x++;
        if ((padded_grid_res_y - grid_res_y) % 2 != 0)
            padded_grid_res_y++;

        create_vertex_and_index_buffer_and_generate_vertices();
        create_sampler_and_images(
            params.base_img_pixels_rgba,
            params.target_img_pixels_rgba,
            params.will_display_images_in_ui
        );
    }

    void GridWarper::create_vertex_and_index_buffer_and_generate_vertices()
    {
        // please keep in mind that the 2D resolution of the vertex array is
        // (padded_grid_res_x + 1) by (padded_grid_res_y + 1) to account for
        // vertices at the edges. for example, for a 2x2 grid we would need 3x3
        // vertices.
        uint32_t n_vertices = (padded_grid_res_x + 1) * (padded_grid_res_y + 1);
        uint32_t vertices_size_bytes = n_vertices * sizeof(GridVertex);

        // create vertex buffer and map its memory
        create_buffer(
            state,
            vertices_size_bytes,
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,

            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,

            vertex_buf,
            vertex_buf_mem
        );
        vertex_buf_mapped = (GridVertex*)vertex_buf_mem->mapped();

        // create index buffer and upload triangle indices
        {
            uint32_t n_cells = padded_grid_res_x * padded_grid_res_y;
            uint32_t n_triangles = n_cells * 2;
            uint32_t n_triangle_vertices = n_triangles * 3;

            std::vector<uint32_t> indices;
            indices.reserve(n_triangle_vertices);
            for (int32_t y = 0; y < padded_grid_res_y; y++)
            {
                for (int32_t x = 0; x < padded_grid_res_x; x++)
                {
                    // index of bottom left, bottom right, top left and top
                    // right vertices
                    uint32_t bl_idx = INDEX_2D(x, y, padded_grid_res_x + 1);
                    uint32_t br_idx = INDEX_2D(x + 1, y, padded_grid_res_x + 1);
                    uint32_t tl_idx = INDEX_2D(x, y + 1, padded_grid_res_x + 1);
                    uint32_t tr_idx =
                        INDEX_2D(x + 1, y + 1, padded_grid_res_x + 1);

                    // push 2 triangles that fill the quad which is the current
                    // cell

                    indices.push_back(bl_idx);
                    indices.push_back(br_idx);
                    indices.push_back(tr_idx);

                    indices.push_back(bl_idx);
                    indices.push_back(tr_idx);
                    indices.push_back(tl_idx);
                }
            }

            VkDeviceSize indices_size_bytes =
                sizeof(indices[0]) * indices.size();

            bv::BufferPtr staging_buf;
            bv::MemoryChunkPtr staging_buf_mem;
            create_buffer(
                state,
                indices_size_bytes,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,

                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,

                staging_buf,
                staging_buf_mem
            );

            void* mapped = staging_buf_mem->mapped();
            std::copy(
                indices.data(),
                indices.data() + indices.size(),
                (uint32_t*)mapped
            );
            staging_buf_mem->flush();

            create_buffer(
                state,
                indices_size_bytes,

                VK_BUFFER_USAGE_TRANSFER_DST_BIT
                | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,

                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                index_buf,
                index_buf_mem
            );

            auto cmd_buf = begin_single_time_commands(state, true);
            copy_buffer(cmd_buf, staging_buf, index_buf, indices_size_bytes);
            end_single_time_commands(state, cmd_buf);

            staging_buf = nullptr;
            staging_buf_mem = nullptr;
        }

        // calculate and set the initial positions of the vertices

        float cell_width = 1.f / (float)grid_res_x;
        float cell_height = 1.f / (float)grid_res_y;

        int32_t horizontal_pad = (int32_t)(padded_grid_res_x - grid_res_x) / 2;
        int32_t vertical_pad = (int32_t)(padded_grid_res_y - grid_res_y) / 2;

        for (int32_t y = 0; y <= padded_grid_res_y; y++)
        {
            for (int32_t x = 0; x <= padded_grid_res_x; x++)
            {
                // remove the offset caused by padding
                int32_t ax = x - horizontal_pad;
                int32_t ay = y - vertical_pad;

                glm::vec2 p{ (float)ax * cell_width, (float)ay * cell_height };

                ACCESS_2D(vertex_buf_mapped, x, y, padded_grid_res_x + 1) = {
                    .warped_pos = p,
                    .orig_pos = p
                };
            }
        }
        vertex_buf_mem->flush();
    }

    void GridWarper::create_sampler_and_images(
        std::span<float> base_img_pixels_rgba,
        std::span<float> target_img_pixels_rgba,
        bool will_display_images_in_ui
    )
    {
        // calculate the size of each of the base and target images in bytes
        static constexpr uint32_t n_channels = 4;
        static constexpr uint32_t n_bytes_per_channel = sizeof(float);
        VkDeviceSize img_size_bytes =
            img_width * img_height
            * n_channels * n_bytes_per_channel;

        // create staging buffer and upload pixel data to it

        bv::BufferPtr staging_buf;
        bv::MemoryChunkPtr staging_buf_mem;

        if (base_img_pixels_rgba.size_bytes() != img_size_bytes
            || target_img_pixels_rgba.size_bytes() != img_size_bytes)
        {
            throw std::invalid_argument(std::format(
                "provided pixel data doesn't have the expected size of {} "
                "bytes",
                img_size_bytes
            ).c_str());
        }

        create_buffer(
            state,
            img_size_bytes * 2, // * 2 cuz we store base and target images
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,

            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,

            staging_buf,
            staging_buf_mem
        );

        void* mapped = staging_buf_mem->mapped();
        std::copy(
            (uint8_t*)base_img_pixels_rgba.data(),
            (uint8_t*)base_img_pixels_rgba.data() + img_size_bytes,
            (uint8_t*)mapped
        );
        std::copy(
            (uint8_t*)target_img_pixels_rgba.data(),
            (uint8_t*)target_img_pixels_rgba.data() + img_size_bytes,
            (uint8_t*)mapped + img_size_bytes
        );
        staging_buf_mem->flush();

        // create sampler
        sampler = bv::Sampler::create(
            state.device,
            {
                .flags = 0,
                .mag_filter = VK_FILTER_LINEAR,
                .min_filter = VK_FILTER_LINEAR,
                .mipmap_mode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
                .address_mode_u = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,
                .address_mode_v = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,
                .address_mode_w = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,
                .mip_lod_bias = 0.,
                .anisotropy_enable = false,
                .max_anisotropy = 0.,
                .compare_enable = false,
                .compare_op = VK_COMPARE_OP_ALWAYS,
                .min_lod = 0.,
                .max_lod = VK_LOD_CLAMP_NONE,
                .border_color = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
                .unnormalized_coordinates = false
            }
        );

        // command buffer for image layout transitions and generating mipmaps
        auto cmd_buf = begin_single_time_commands(state, true);

        // create images and image views, and transition layouts

        // base and target images use the original resolution with mipmapping,
        // they are transfer destinations (when copying from staging buffer to
        // them) and sources (when generating mipmaps), and they are sampled in
        // shaders and the UI.

        uint32_t mip_levels = round_log2(std::max(img_width, img_height));

        create_image(
            state,
            img_width,
            img_height,
            mip_levels,
            VK_SAMPLE_COUNT_1_BIT,
            RGBA_FORMAT,
            VK_IMAGE_TILING_OPTIMAL,

            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
            | VK_IMAGE_USAGE_SAMPLED_BIT,

            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            base_img,
            base_img_mem
        );
        base_imgview = create_image_view(
            state,
            base_img,
            RGBA_FORMAT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            mip_levels
        );
        transition_image_layout(
            cmd_buf,
            base_img,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            mip_levels
        );

        create_image(
            state,
            img_width,
            img_height,
            mip_levels,
            VK_SAMPLE_COUNT_1_BIT,
            RGBA_FORMAT,
            VK_IMAGE_TILING_OPTIMAL,

            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
            | VK_IMAGE_USAGE_SAMPLED_BIT,

            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            target_img,
            target_img_mem
        );
        target_imgview = create_image_view(
            state,
            target_img,
            RGBA_FORMAT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            mip_levels
        );
        transition_image_layout(
            cmd_buf,
            target_img,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            mip_levels
        );

        // warped image uses intermediate resolution with no mipmapping. it's
        // a color attachment but also sampled in the difference pass and the
        // UI.

        create_image(
            state,
            intermediate_res_x,
            intermediate_res_y,
            1,
            VK_SAMPLE_COUNT_1_BIT,
            RGBA_FORMAT,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            warped_img,
            warped_img_mem
        );
        warped_imgview = create_image_view(
            state,
            warped_img,
            RGBA_FORMAT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            1
        );
        transition_image_layout(
            cmd_buf,
            warped_img,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL,
            1
        );

        // final warped image uses the original resolution with no mipmapping.
        // it's a color attachment but also sampled in the UI. it can also be
        // a transfer source when we want to read back the image for saving.
        create_image(
            state,
            img_width,
            img_height,
            1,
            VK_SAMPLE_COUNT_1_BIT,
            RGBA_FORMAT,
            VK_IMAGE_TILING_OPTIMAL,

            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
            | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,

            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            final_img,
            final_img_mem
        );
        final_imgview = create_image_view(
            state,
            final_img,
            RGBA_FORMAT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            1
        );
        transition_image_layout(
            cmd_buf,
            final_img,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL,
            1
        );

        // difference image uses a square resolution of the closest upper power
        // of 2 of the intermediate resolution. it uses mipmapping and is a
        // color attachment, but also sampled in the UI. it can also be a
        // transfer source when we want to read back the average difference
        // (error) while optimizing, or when we generate mipmaps.
        difference_res_square = upper_power_of_2(std::max(
            intermediate_res_x,
            intermediate_res_y
        ));
        uint32_t difference_mip_levels = round_log2(difference_res_square);
        create_image(
            state,
            difference_res_square,
            difference_res_square,
            difference_mip_levels,
            VK_SAMPLE_COUNT_1_BIT,
            R_FORMAT,
            VK_IMAGE_TILING_OPTIMAL,

            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
            | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,

            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            difference_img,
            difference_img_mem
        );
        difference_imgview = create_image_view(
            state,
            difference_img,
            R_FORMAT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            difference_mip_levels
        );
        transition_image_layout(
            cmd_buf,
            difference_img,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL,
            difference_mip_levels
        );

        // copy image data from the staging buffer to the base and target images
        copy_buffer_to_image(
            cmd_buf,
            staging_buf,
            base_img,
            0
        );
        copy_buffer_to_image(
            cmd_buf,
            staging_buf,
            target_img,
            img_size_bytes
        );

        // generate mipmaps for base and target images which will also
        // transition them to VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL.
        generate_mipmaps(
            state,
            cmd_buf,
            base_img
        );
        generate_mipmaps(
            state,
            cmd_buf,
            target_img
        );

        // end, submit, and wait for the command buffer
        end_single_time_commands(state, cmd_buf);

        staging_buf = nullptr;
        staging_buf_mem = nullptr;

        // create descriptor sets for ImGUI's Vulkan implementation to display
        // images in the UI.
        if (will_display_images_in_ui)
        {
            base_img_ds_imgui = ImGui_ImplVulkan_AddTexture(
                sampler->handle(),
                base_imgview->handle(),
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
            );
            target_img_ds_imgui = ImGui_ImplVulkan_AddTexture(
                sampler->handle(),
                target_imgview->handle(),
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
            );
            warped_img_ds_imgui = ImGui_ImplVulkan_AddTexture(
                sampler->handle(),
                warped_imgview->handle(),
                VK_IMAGE_LAYOUT_GENERAL
            );
            difference_img_ds_imgui = ImGui_ImplVulkan_AddTexture(
                sampler->handle(),
                difference_imgview->handle(),
                VK_IMAGE_LAYOUT_GENERAL
            );
            final_img_ds_imgui = ImGui_ImplVulkan_AddTexture(
                sampler->handle(),
                final_imgview->handle(),
                VK_IMAGE_LAYOUT_GENERAL
            );
        }
    }

}
