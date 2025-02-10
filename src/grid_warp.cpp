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
        gwp_vertex_attributes
    {
        bv::VertexInputAttributeDescription{
            .location = 0,
            .binding = 0,
            .format = VK_FORMAT_R32G32_SFLOAT,
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
        const Params& params,
        size_t thread_idx
    )
        : state(state),
        img_width(params.img_width),
        img_height(params.img_height),
        will_display_images_in_ui(params.will_display_images_in_ui)
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

        create_vertex_and_index_buffer_and_generate_vertices(thread_idx);
        create_sampler_and_images(
            params.base_img_pixels_rgba,
            params.target_img_pixels_rgba,
            thread_idx
        );
        create_avg_difference_buffer();
        create_passes();
    }

    GridWarper::~GridWarper()
    {
        if (_ui_img_ds_for_imgui != nullptr)
            ImGui_ImplVulkan_RemoveTexture(_ui_img_ds_for_imgui);

        _ui_image_infos.clear();

        uip_fence = nullptr;
        uip_graphics_pipeline = nullptr;
        uip_pipeline_layout = nullptr;
        uip_framebuf = nullptr;
        uip_render_pass = nullptr;

        uip_descriptor_pool = nullptr;
        uip_descriptor_set_layout = nullptr;

        dfp_fence = nullptr;
        dfp_graphics_pipeline = nullptr;
        dfp_pipeline_layout = nullptr;
        dfp_framebuf = nullptr;
        dfp_render_pass = nullptr;

        dfp_descriptor_set = nullptr;
        dfp_descriptor_pool = nullptr;
        dfp_descriptor_set_layout = nullptr;

        gwp_fence = nullptr;
        gwp_graphics_pipeline = nullptr;
        gwp_pipeline_layout = nullptr;
        gwp_framebuf = nullptr;
        gwp_framebuf_hires = nullptr;
        gwp_render_pass = nullptr;

        gwp_descriptor_set = nullptr;
        gwp_descriptor_pool = nullptr;
        gwp_descriptor_set_layout = nullptr;

        vertex_buf = nullptr;
        vertex_buf_mem = nullptr;

        index_buf = nullptr;
        index_buf_mem = nullptr;

        base_img = nullptr;
        base_img_mem = nullptr;
        base_imgview = nullptr;

        target_img = nullptr;
        target_img_mem = nullptr;
        target_imgview = nullptr;

        warped_img = nullptr;
        warped_img_mem = nullptr;
        warped_imgview = nullptr;

        warped_hires_img = nullptr;
        warped_hires_img_mem = nullptr;
        warped_hires_imgview = nullptr;

        difference_img = nullptr;
        difference_img_mem = nullptr;
        difference_imgview = nullptr;
        difference_imgview_first_mip = nullptr;

        ui_img = nullptr;
        ui_img_mem = nullptr;
        ui_imgview = nullptr;

        sampler = nullptr;

        avg_difference_buf = nullptr;
        avg_difference_buf_mem = nullptr;
    }

    void GridWarper::run_grid_warp_pass(bool hires, size_t thread_idx)
    {
        auto cmd_buf = create_grid_warp_pass_cmd_buf(hires, thread_idx);
        {
            std::scoped_lock lock(state.queue_mutex);
            state.queue->submit({}, {}, { cmd_buf }, {}, gwp_fence);
        }
        gwp_fence->wait();
        gwp_fence->reset();
    }

    float GridWarper::run_difference_pass(size_t thread_idx)
    {
        auto cmd_buf = create_difference_pass_cmd_buf(thread_idx);
        {
            std::scoped_lock lock(state.queue_mutex);
            state.queue->submit({}, {}, { cmd_buf }, {}, dfp_fence);
        }
        dfp_fence->wait();
        dfp_fence->reset();

        return *avg_difference_buf_mapped;
    }

    void GridWarper::run_ui_pass(
        size_t ui_image_info_idx,
        float exposure,
        bool use_flim,
        size_t thread_idx
    )
    {
        if (!will_display_images_in_ui)
        {
            throw std::logic_error("can't run UI pass when it's disabled");
        }

        if (ui_image_info_idx >= _ui_image_infos.size())
        {
            throw std::invalid_argument("invalid index for the UI image info");
        }

        const auto& ui_image_info = _ui_image_infos[ui_image_info_idx];

        uip_frag_push_constants.single_channel =
            ui_image_info.single_channel ? 1 : 0;

        uip_frag_push_constants.img_mul = std::exp2(exposure);
        uip_frag_push_constants.use_flim = use_flim ? 1 : 0;

        auto cmd_buf = create_ui_pass_cmd_buf(
            ui_image_info.ui_pass_ds,
            thread_idx
        );
        {
            std::scoped_lock lock(state.queue_mutex);
            state.queue->submit({}, {}, { cmd_buf }, {}, uip_fence);
        }
        uip_fence->wait();
        uip_fence->reset();
    }

    void GridWarper::create_vertex_and_index_buffer_and_generate_vertices(
        size_t thread_idx
    )
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
            n_triangle_vertices = n_triangles * 3;

            std::vector<uint32_t> indices;
            indices.reserve(n_triangle_vertices);
            for (int32_t y = 0; y < (int32_t)padded_grid_res_y; y++)
            {
                for (int32_t x = 0; x < (int32_t)padded_grid_res_x; x++)
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

            auto cmd_buf = begin_single_time_commands(state, true, thread_idx);
            copy_buffer(cmd_buf, staging_buf, index_buf, indices_size_bytes);
            end_single_time_commands(state, cmd_buf, true);

            staging_buf = nullptr;
            staging_buf_mem = nullptr;
        }

        // calculate and set the initial positions of the vertices

        float cell_width = 1.f / (float)grid_res_x;
        float cell_height = 1.f / (float)grid_res_y;

        int32_t horizontal_pad = (int32_t)(padded_grid_res_x - grid_res_x) / 2;
        int32_t vertical_pad = (int32_t)(padded_grid_res_y - grid_res_y) / 2;

        for (int32_t y = 0; y <= (int32_t)padded_grid_res_y; y++)
        {
            for (int32_t x = 0; x <= (int32_t)padded_grid_res_x; x++)
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
        size_t thread_idx
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
        auto cmd_buf = begin_single_time_commands(state, true, thread_idx);

        // create images and image views, and transition layouts

        // base and target images use the original resolution with mipmapping

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
        // a color attachment but also sampled in the difference pass.

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

        // high-resolution warped image uses the original resolution with no
        // mipmapping
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
            warped_hires_img,
            warped_hires_img_mem
        );
        warped_hires_imgview = create_image_view(
            state,
            warped_hires_img,
            RGBA_FORMAT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            1
        );
        transition_image_layout(
            cmd_buf,
            warped_hires_img,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL,
            1
        );

        // difference image uses a square resolution of the closest upper power
        // of 2 of the intermediate resolution. it uses mipmapping to get the
        // average difference (error) while optimizing.
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
            | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,

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
        difference_imgview_first_mip = create_image_view(
            state,
            difference_img,
            R_FORMAT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            1
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
            base_img,
            false,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_ACCESS_SHADER_READ_BIT
        );
        generate_mipmaps(
            state,
            cmd_buf,
            target_img,
            false,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_ACCESS_SHADER_READ_BIT
        );

        // UI-specific
        if (will_display_images_in_ui)
        {
            // create the UI image which uses the original resolution with no
            // mipmapping. images with different resolutions will just be
            // rendered to the bottom-left corner at which we'll crop when
            // providing UV coordinates to ImGUI.
            create_image(
                state,
                img_width,
                img_height,
                1,
                VK_SAMPLE_COUNT_1_BIT,
                UI_FORMAT,
                VK_IMAGE_TILING_OPTIMAL,

                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                | VK_IMAGE_USAGE_SAMPLED_BIT,

                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                ui_img,
                ui_img_mem
            );
            ui_imgview = create_image_view(
                state,
                ui_img,
                UI_FORMAT,
                VK_IMAGE_ASPECT_COLOR_BIT,
                1
            );
            transition_image_layout(
                cmd_buf,
                ui_img,
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_GENERAL,
                1
            );

            // create descriptor set for ImGUI's Vulkan implementation to
            // display the UI image
            _ui_img_ds_for_imgui = ImGui_ImplVulkan_AddTexture(
                sampler->handle(),
                ui_imgview->handle(),
                VK_IMAGE_LAYOUT_GENERAL
            );
        }

        // end, submit, and wait for the command buffer
        end_single_time_commands(state, cmd_buf, true);

        staging_buf = nullptr;
        staging_buf_mem = nullptr;
    }

    void GridWarper::create_avg_difference_buffer()
    {
        create_buffer(
            state,
            sizeof(float),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,

            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,

            avg_difference_buf,
            avg_difference_buf_mem
        );
        avg_difference_buf_mapped = (float*)avg_difference_buf_mem->mapped();
    }

    void GridWarper::create_passes()
    {
        // shaders

        bv::ShaderModulePtr fullscreen_quad_vert_shader_module = nullptr;
        bv::ShaderStage fullscreen_quad_vert_shader_stage{};

        bv::ShaderModulePtr gwp_vert_shader_module = nullptr;
        bv::ShaderStage gwp_vert_shader_stage{};

        bv::ShaderModulePtr gwp_frag_shader_module = nullptr;
        bv::ShaderStage gwp_frag_shader_stage{};

        bv::ShaderModulePtr dfp_frag_shader_module = nullptr;
        bv::ShaderStage dfp_frag_shader_stage{};

        bv::ShaderModulePtr uip_frag_shader_module = nullptr;
        bv::ShaderStage uip_frag_shader_stage{};

        {
            std::vector<uint8_t> shader_code = read_file(
                "./shaders/fullscreen_quad_vert.spv"
            );
            fullscreen_quad_vert_shader_module = bv::ShaderModule::create(
                state.device,
                std::move(shader_code)
            );
            fullscreen_quad_vert_shader_stage = bv::ShaderStage{
                .flags = {},
                .stage = VK_SHADER_STAGE_VERTEX_BIT,
                .module = fullscreen_quad_vert_shader_module,
                .entry_point = "main",
                .specialization_info = std::nullopt
            };

            shader_code = read_file(
                "./shaders/grid_warp_pass_vert.spv"
            );
            gwp_vert_shader_module = bv::ShaderModule::create(
                state.device,
                std::move(shader_code)
            );
            gwp_vert_shader_stage = bv::ShaderStage{
                .flags = {},
                .stage = VK_SHADER_STAGE_VERTEX_BIT,
                .module = gwp_vert_shader_module,
                .entry_point = "main",
                .specialization_info = std::nullopt
            };

            shader_code = read_file(
                "./shaders/grid_warp_pass_frag.spv"
            );
            gwp_frag_shader_module = bv::ShaderModule::create(
                state.device,
                std::move(shader_code)
            );
            gwp_frag_shader_stage = bv::ShaderStage{
                .flags = {},
                .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                .module = gwp_frag_shader_module,
                .entry_point = "main",
                .specialization_info = std::nullopt
            };

            shader_code = read_file(
                "./shaders/difference_pass_frag.spv"
            );
            dfp_frag_shader_module = bv::ShaderModule::create(
                state.device,
                std::move(shader_code)
            );
            dfp_frag_shader_stage = bv::ShaderStage{
                .flags = {},
                .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                .module = dfp_frag_shader_module,
                .entry_point = "main",
                .specialization_info = std::nullopt
            };

            shader_code = read_file(
                "./shaders/ui_pass_frag.spv"
            );
            uip_frag_shader_module = bv::ShaderModule::create(
                state.device,
                std::move(shader_code)
            );
            uip_frag_shader_stage = bv::ShaderStage{
                .flags = {},
                .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                .module = uip_frag_shader_module,
                .entry_point = "main",
                .specialization_info = std::nullopt
            };
        }

        // grid warp pass: descriptor set layout
        {
            bv::DescriptorSetLayoutBinding binding_base_img{
                .binding = 0,
                .descriptor_type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptor_count = 1,
                .stage_flags = VK_SHADER_STAGE_FRAGMENT_BIT,
                .immutable_samplers = { sampler }
            };

            gwp_descriptor_set_layout = bv::DescriptorSetLayout::create(
                state.device,
                {
                    .flags = 0,
                    .bindings = { binding_base_img }
                }
            );
        }

        // grid warp pass: descriptor pool
        {
            // 1 image in every descriptor set * 1 set in total
            bv::DescriptorPoolSize image_pool_size{
                .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptor_count = 1
            };

            gwp_descriptor_pool = bv::DescriptorPool::create(
                state.device,
                {
                    .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
                    .max_sets = 1,
                    .pool_sizes = { image_pool_size }
                }
            );
        }

        // grid warp pass: descriptor set
        {
            gwp_descriptor_set = bv::DescriptorPool::allocate_set(
                gwp_descriptor_pool,
                gwp_descriptor_set_layout
            );

            bv::DescriptorImageInfo base_img_info{
                .sampler = sampler,
                .image_view = base_imgview,
                .image_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
            };

            std::vector<bv::WriteDescriptorSet> descriptor_writes;

            descriptor_writes.push_back({
                .dst_set = gwp_descriptor_set,
                .dst_binding = 0,
                .dst_array_element = 0,
                .descriptor_count = 1,
                .descriptor_type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .image_infos = { base_img_info },
                .buffer_infos = {},
                .texel_buffer_views = {}
                });

            bv::DescriptorSet::update_sets(state.device, descriptor_writes, {});
        }

        // grid warp pass: render pass
        {
            bv::Attachment color_attachment{
                .flags = 0,
                .format = RGBA_FORMAT,
                .samples = VK_SAMPLE_COUNT_1_BIT,
                .load_op = VK_ATTACHMENT_LOAD_OP_CLEAR,
                .store_op = VK_ATTACHMENT_STORE_OP_STORE,
                .stencil_load_op = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                .stencil_store_op = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                .initial_layout = VK_IMAGE_LAYOUT_UNDEFINED,
                .final_layout = VK_IMAGE_LAYOUT_GENERAL
            };

            bv::AttachmentReference color_attachment_ref{
                .attachment = 0,
                .layout = VK_IMAGE_LAYOUT_GENERAL
            };

            bv::Subpass subpass{
                .flags = 0,
                .pipeline_bind_point = VK_PIPELINE_BIND_POINT_GRAPHICS,
                .input_attachments = {},
                .color_attachments = { color_attachment_ref },
                .resolve_attachments = {},
                .depth_stencil_attachment = std::nullopt,
                .preserve_attachment_indices = {}
            };

            bv::SubpassDependency dependency{
                .src_subpass = VK_SUBPASS_EXTERNAL,
                .dst_subpass = 0,
                .src_stage_mask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                .dst_stage_mask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                .src_access_mask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                .dst_access_mask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                .dependency_flags = 0
            };

            gwp_render_pass = bv::RenderPass::create(
                state.device,
                bv::RenderPassConfig{
                    .flags = 0,
                    .attachments = { color_attachment },
                    .subpasses = { subpass },
                    .dependencies = { dependency }
                }
            );
        }

        // grid warp pass: framebuffers
        {
            gwp_framebuf = bv::Framebuffer::create(
                state.device,
                bv::FramebufferConfig{
                    .flags = 0,
                    .render_pass = gwp_render_pass,
                    .attachments = { warped_imgview },
                    .width = warped_img->config().extent.width,
                    .height = warped_img->config().extent.height,
                    .layers = 1
                }
            );

            gwp_framebuf_hires = bv::Framebuffer::create(
                state.device,
                bv::FramebufferConfig{
                    .flags = 0,
                    .render_pass = gwp_render_pass,
                    .attachments = { warped_hires_imgview },
                    .width = warped_hires_img->config().extent.width,
                    .height = warped_hires_img->config().extent.height,
                    .layers = 1
                }
            );
        }

        // grid warp pass: pipeline layout
        {
            bv::PushConstantRange frag_push_constant_range{
                .stage_flags = VK_SHADER_STAGE_FRAGMENT_BIT,
                .offset = 0,
                .size = sizeof(GridWarpPassFragPushConstants)
            };

            gwp_pipeline_layout = bv::PipelineLayout::create(
                state.device,
                bv::PipelineLayoutConfig{
                    .flags = 0,
                    .set_layouts = { gwp_descriptor_set_layout },
                    .push_constant_ranges = { frag_push_constant_range }
                }
            );
        }

        // grid warp pass: graphics pipeline
        {
            // not using intermediate res here because we'll use dynamic
            // viewport and scissor states anyway.
            bv::Viewport viewport{
                .x = 0.f,
                .y = 0.f,
                .width = (float)img_width,
                .height = (float)img_height,
                .min_depth = 0.f,
                .max_depth = 1.f
            };

            bv::Rect2d scissor{
                .offset = { 0, 0 },
                .extent = { img_width, img_height }
            };

            bv::ViewportState viewport_state{
                .viewports = { viewport },
                .scissors = { scissor }
            };

            bv::RasterizationState rasterization_state{
                .depth_clamp_enable = false,
                .rasterizer_discard_enable = false,
                .polygon_mode = VK_POLYGON_MODE_FILL,
                .cull_mode = VK_CULL_MODE_BACK_BIT,
                .front_face = VK_FRONT_FACE_CLOCKWISE,
                .depth_bias_enable = false,
                .depth_bias_constant_factor = 0.f,
                .depth_bias_clamp = 0.f,
                .depth_bias_slope_factor = 0.f,
                .line_width = 1.f
            };

            bv::MultisampleState multisample_state{
                .rasterization_samples = VK_SAMPLE_COUNT_1_BIT,
                .sample_shading_enable = false,
                .min_sample_shading = 1.f,
                .sample_mask = {},
                .alpha_to_coverage_enable = false,
                .alpha_to_one_enable = false
            };

            bv::DepthStencilState depth_stencil_state{
                .flags = 0,
                .depth_test_enable = false,
                .depth_write_enable = false,
                .depth_compare_op = VK_COMPARE_OP_LESS,
                .depth_bounds_test_enable = false,
                .stencil_test_enable = false,
                .front = {},
                .back = {},
                .min_depth_bounds = 0.f,
                .max_depth_bounds = 1.f
            };

            bv::ColorBlendAttachment color_blend_attachment{
                .blend_enable = false,
                .src_color_blend_factor = VK_BLEND_FACTOR_ONE,
                .dst_color_blend_factor = VK_BLEND_FACTOR_ZERO,
                .color_blend_op = VK_BLEND_OP_ADD,
                .src_alpha_blend_factor = VK_BLEND_FACTOR_ONE,
                .dst_alpha_blend_factor = VK_BLEND_FACTOR_ZERO,
                .alpha_blend_op = VK_BLEND_OP_ADD,
                .color_write_mask =
                VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT
            };

            bv::ColorBlendState color_blend_state{
                .flags = 0,
                .logic_op_enable = false,
                .logic_op = VK_LOGIC_OP_COPY,
                .attachments = { color_blend_attachment },
                .blend_constants = { 0.f, 0.f, 0.f, 0.f }
            };

            gwp_graphics_pipeline = bv::GraphicsPipeline::create(
                state.device,
                bv::GraphicsPipelineConfig{
                    .flags = 0,
                    .stages = { gwp_vert_shader_stage, gwp_frag_shader_stage },
                    .vertex_input_state = bv::VertexInputState{
                        .binding_descriptions = { GridVertex::binding },
                        .attribute_descriptions = { gwp_vertex_attributes }
                    },
                    .input_assembly_state = bv::InputAssemblyState{
                        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                        .primitive_restart_enable = false
                    },
                    .tessellation_state = std::nullopt,
                    .viewport_state = viewport_state,
                    .rasterization_state = rasterization_state,
                    .multisample_state = multisample_state,
                    .depth_stencil_state = depth_stencil_state,
                    .color_blend_state = color_blend_state,
                    .dynamic_states = {
                        VK_DYNAMIC_STATE_VIEWPORT,
                        VK_DYNAMIC_STATE_SCISSOR
                    },
                    .layout = gwp_pipeline_layout,
                    .render_pass = gwp_render_pass,
                    .subpass_index = 0,
                    .base_pipeline = std::nullopt
                }
            );
        }

        // grid warp pass: fence
        gwp_fence = bv::Fence::create(state.device, 0);

        // difference pass: descriptor set layout
        {
            bv::DescriptorSetLayoutBinding binding_warped_img{
                .binding = 0,
                .descriptor_type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptor_count = 1,
                .stage_flags = VK_SHADER_STAGE_FRAGMENT_BIT,
                .immutable_samplers = { sampler }
            };

            bv::DescriptorSetLayoutBinding binding_target_img{
                .binding = 1,
                .descriptor_type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptor_count = 1,
                .stage_flags = VK_SHADER_STAGE_FRAGMENT_BIT,
                .immutable_samplers = { sampler }
            };

            dfp_descriptor_set_layout = bv::DescriptorSetLayout::create(
                state.device,
                {
                    .flags = 0,
                    .bindings = { binding_warped_img, binding_target_img }
                }
            );
        }

        // difference pass: descriptor pool
        {
            // 2 images in every descriptor set * 1 set in total
            bv::DescriptorPoolSize image_pool_size{
                .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptor_count = 2
            };

            dfp_descriptor_pool = bv::DescriptorPool::create(
                state.device,
                {
                    .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
                    .max_sets = 1,
                    .pool_sizes = { image_pool_size }
                }
            );
        }

        // difference pass: descriptor set
        {
            dfp_descriptor_set = bv::DescriptorPool::allocate_set(
                dfp_descriptor_pool,
                dfp_descriptor_set_layout
            );

            bv::DescriptorImageInfo warped_img_info{
                .sampler = sampler,
                .image_view = warped_imgview,
                .image_layout = VK_IMAGE_LAYOUT_GENERAL
            };

            bv::DescriptorImageInfo target_img_info{
                .sampler = sampler,
                .image_view = target_imgview,
                .image_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
            };

            std::vector<bv::WriteDescriptorSet> descriptor_writes;

            descriptor_writes.push_back({
                .dst_set = dfp_descriptor_set,
                .dst_binding = 0,
                .dst_array_element = 0,
                .descriptor_count = 1,
                .descriptor_type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .image_infos = { warped_img_info },
                .buffer_infos = {},
                .texel_buffer_views = {}
                });

            descriptor_writes.push_back({
                .dst_set = dfp_descriptor_set,
                .dst_binding = 1,
                .dst_array_element = 0,
                .descriptor_count = 1,
                .descriptor_type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .image_infos = { target_img_info },
                .buffer_infos = {},
                .texel_buffer_views = {}
                });

            bv::DescriptorSet::update_sets(state.device, descriptor_writes, {});
        }

        // difference pass: render pass
        {
            bv::Attachment color_attachment{
                .flags = 0,
                .format = R_FORMAT,
                .samples = VK_SAMPLE_COUNT_1_BIT,
                .load_op = VK_ATTACHMENT_LOAD_OP_CLEAR,
                .store_op = VK_ATTACHMENT_STORE_OP_STORE,
                .stencil_load_op = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                .stencil_store_op = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                .initial_layout = VK_IMAGE_LAYOUT_UNDEFINED,
                .final_layout = VK_IMAGE_LAYOUT_GENERAL
            };

            bv::AttachmentReference color_attachment_ref{
                .attachment = 0,
                .layout = VK_IMAGE_LAYOUT_GENERAL
            };

            bv::Subpass subpass{
                .flags = 0,
                .pipeline_bind_point = VK_PIPELINE_BIND_POINT_GRAPHICS,
                .input_attachments = {},
                .color_attachments = { color_attachment_ref },
                .resolve_attachments = {},
                .depth_stencil_attachment = std::nullopt,
                .preserve_attachment_indices = {}
            };

            bv::SubpassDependency dependency{
                .src_subpass = VK_SUBPASS_EXTERNAL,
                .dst_subpass = 0,
                .src_stage_mask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                .dst_stage_mask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                .src_access_mask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                .dst_access_mask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                .dependency_flags = 0
            };

            dfp_render_pass = bv::RenderPass::create(
                state.device,
                bv::RenderPassConfig{
                    .flags = 0,
                    .attachments = { color_attachment },
                    .subpasses = { subpass },
                    .dependencies = { dependency }
                }
            );
        }

        // difference pass: framebuffer
        dfp_framebuf = bv::Framebuffer::create(
            state.device,
            bv::FramebufferConfig{
                .flags = 0,
                .render_pass = dfp_render_pass,
                .attachments = { difference_imgview_first_mip },
                .width = difference_img->config().extent.width,
                .height = difference_img->config().extent.height,
                .layers = 1
            }
        );

        // difference pass: pipeline layout
        {
            bv::PushConstantRange frag_push_constant_range{
                .stage_flags = VK_SHADER_STAGE_FRAGMENT_BIT,
                .offset = 0,
                .size = sizeof(DifferencePassFragPushConstants)
            };

            dfp_pipeline_layout = bv::PipelineLayout::create(
                state.device,
                bv::PipelineLayoutConfig{
                    .flags = 0,
                    .set_layouts = { dfp_descriptor_set_layout },
                    .push_constant_ranges = { frag_push_constant_range }
                }
            );
        }

        // difference pass: graphics pipeline
        {
            bv::Viewport viewport{
                .x = 0.f,
                .y = 0.f,
                .width = (float)dfp_framebuf->config().width,
                .height = (float)dfp_framebuf->config().height,
                .min_depth = 0.f,
                .max_depth = 1.f
            };

            bv::Rect2d scissor{
                .offset = { 0, 0 },
                .extent = {
                    dfp_framebuf->config().width,
                    dfp_framebuf->config().height
                }
            };

            bv::ViewportState viewport_state{
                .viewports = { viewport },
                .scissors = { scissor }
            };

            bv::RasterizationState rasterization_state{
                .depth_clamp_enable = false,
                .rasterizer_discard_enable = false,
                .polygon_mode = VK_POLYGON_MODE_FILL,
                .cull_mode = VK_CULL_MODE_BACK_BIT,
                .front_face = VK_FRONT_FACE_CLOCKWISE,
                .depth_bias_enable = false,
                .depth_bias_constant_factor = 0.f,
                .depth_bias_clamp = 0.f,
                .depth_bias_slope_factor = 0.f,
                .line_width = 1.f
            };

            bv::MultisampleState multisample_state{
                .rasterization_samples = VK_SAMPLE_COUNT_1_BIT,
                .sample_shading_enable = false,
                .min_sample_shading = 1.f,
                .sample_mask = {},
                .alpha_to_coverage_enable = false,
                .alpha_to_one_enable = false
            };

            bv::DepthStencilState depth_stencil_state{
                .flags = 0,
                .depth_test_enable = false,
                .depth_write_enable = false,
                .depth_compare_op = VK_COMPARE_OP_LESS,
                .depth_bounds_test_enable = false,
                .stencil_test_enable = false,
                .front = {},
                .back = {},
                .min_depth_bounds = 0.f,
                .max_depth_bounds = 1.f
            };

            bv::ColorBlendAttachment color_blend_attachment{
                .blend_enable = false,
                .src_color_blend_factor = VK_BLEND_FACTOR_ONE,
                .dst_color_blend_factor = VK_BLEND_FACTOR_ZERO,
                .color_blend_op = VK_BLEND_OP_ADD,
                .src_alpha_blend_factor = VK_BLEND_FACTOR_ONE,
                .dst_alpha_blend_factor = VK_BLEND_FACTOR_ZERO,
                .alpha_blend_op = VK_BLEND_OP_ADD,
                .color_write_mask =
                VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT
            };

            bv::ColorBlendState color_blend_state{
                .flags = 0,
                .logic_op_enable = false,
                .logic_op = VK_LOGIC_OP_COPY,
                .attachments = { color_blend_attachment },
                .blend_constants = { 0.f, 0.f, 0.f, 0.f }
            };

            dfp_graphics_pipeline = bv::GraphicsPipeline::create(
                state.device,
                bv::GraphicsPipelineConfig{
                    .flags = 0,
                    .stages = {
                        fullscreen_quad_vert_shader_stage,
                        dfp_frag_shader_stage
                    },
                    .vertex_input_state = bv::VertexInputState{
                        .binding_descriptions = {},
                        .attribute_descriptions = {}
                    },
                    .input_assembly_state = bv::InputAssemblyState{
                        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                        .primitive_restart_enable = false
                    },
                    .tessellation_state = std::nullopt,
                    .viewport_state = viewport_state,
                    .rasterization_state = rasterization_state,
                    .multisample_state = multisample_state,
                    .depth_stencil_state = depth_stencil_state,
                    .color_blend_state = color_blend_state,
                    .dynamic_states = {},
                    .layout = dfp_pipeline_layout,
                    .render_pass = dfp_render_pass,
                    .subpass_index = 0,
                    .base_pipeline = std::nullopt
                }
            );
        }

        // difference pass: fence
        dfp_fence = bv::Fence::create(state.device, 0);

        // UI pass: descriptor set layout
        if (will_display_images_in_ui)
        {
            bv::DescriptorSetLayoutBinding binding_img{
                .binding = 0,
                .descriptor_type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptor_count = 1,
                .stage_flags = VK_SHADER_STAGE_FRAGMENT_BIT,
                .immutable_samplers = { sampler }
            };

            uip_descriptor_set_layout = bv::DescriptorSetLayout::create(
                state.device,
                {
                    .flags = 0,
                    .bindings = { binding_img }
                }
            );
        }

        // UI pass: descriptor pool
        if (will_display_images_in_ui)
        {
            // 1 image in every descriptor set * less than 16 sets in total
            bv::DescriptorPoolSize image_pool_size{
                .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptor_count = 16
            };

            uip_descriptor_pool = bv::DescriptorPool::create(
                state.device,
                {
                    .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
                    .max_sets = 16,
                    .pool_sizes = { image_pool_size }
                }
            );
        }

        // UI pass: descriptor sets
        if (will_display_images_in_ui)
        {
            _ui_image_infos.push_back({
                "Base Image",
                base_img->config().extent.width,
                base_img->config().extent.height,
                false,
                nullptr
                });
            create_ui_pass_descriptor_set(
                base_imgview,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                _ui_image_infos.back().ui_pass_ds
            );

            _ui_image_infos.push_back({
                "Target Image",
                target_img->config().extent.width,
                target_img->config().extent.height,
                false,
                nullptr
                });
            create_ui_pass_descriptor_set(
                target_imgview,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                _ui_image_infos.back().ui_pass_ds
            );

            _ui_image_infos.push_back({
                "Warped Image (Intermediate Resolution)",
                warped_img->config().extent.width,
                warped_img->config().extent.height,
                false,
                nullptr
                });
            create_ui_pass_descriptor_set(
                warped_imgview,
                VK_IMAGE_LAYOUT_GENERAL,
                _ui_image_infos.back().ui_pass_ds
            );

            _ui_image_infos.push_back({
                "Warped Image (Original Resolution)",
                warped_hires_img->config().extent.width,
                warped_hires_img->config().extent.height,
                false,
                nullptr
                });
            create_ui_pass_descriptor_set(
                warped_hires_imgview,
                VK_IMAGE_LAYOUT_GENERAL,
                _ui_image_infos.back().ui_pass_ds
            );

            // we only need the bottom left corner of the difference image which
            // is a rectangle of the intermediate resolution.
            _ui_image_infos.push_back({
                "Difference Image",
                warped_img->config().extent.width,
                warped_img->config().extent.height,
                true,
                nullptr
                });
            create_ui_pass_descriptor_set(
                difference_imgview,
                VK_IMAGE_LAYOUT_GENERAL,
                _ui_image_infos.back().ui_pass_ds
            );
        }

        // UI pass: render pass
        if (will_display_images_in_ui)
        {
            bv::Attachment color_attachment{
                .flags = 0,
                .format = UI_FORMAT,
                .samples = VK_SAMPLE_COUNT_1_BIT,
                .load_op = VK_ATTACHMENT_LOAD_OP_CLEAR,
                .store_op = VK_ATTACHMENT_STORE_OP_STORE,
                .stencil_load_op = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                .stencil_store_op = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                .initial_layout = VK_IMAGE_LAYOUT_UNDEFINED,
                .final_layout = VK_IMAGE_LAYOUT_GENERAL
            };

            bv::AttachmentReference color_attachment_ref{
                .attachment = 0,
                .layout = VK_IMAGE_LAYOUT_GENERAL
            };

            bv::Subpass subpass{
                .flags = 0,
                .pipeline_bind_point = VK_PIPELINE_BIND_POINT_GRAPHICS,
                .input_attachments = {},
                .color_attachments = { color_attachment_ref },
                .resolve_attachments = {},
                .depth_stencil_attachment = std::nullopt,
                .preserve_attachment_indices = {}
            };

            bv::SubpassDependency dependency{
                .src_subpass = VK_SUBPASS_EXTERNAL,
                .dst_subpass = 0,
                .src_stage_mask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                .dst_stage_mask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                .src_access_mask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                .dst_access_mask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                .dependency_flags = 0
            };

            uip_render_pass = bv::RenderPass::create(
                state.device,
                bv::RenderPassConfig{
                    .flags = 0,
                    .attachments = { color_attachment },
                    .subpasses = { subpass },
                    .dependencies = { dependency }
                }
            );
        }

        // UI pass: framebuffer
        if (will_display_images_in_ui)
        {
            uip_framebuf = bv::Framebuffer::create(
                state.device,
                bv::FramebufferConfig{
                    .flags = 0,
                    .render_pass = uip_render_pass,
                    .attachments = { ui_imgview },
                    .width = ui_img->config().extent.width,
                    .height = ui_img->config().extent.height,
                    .layers = 1
                }
            );
        }

        // UI pass: pipeline layout
        if (will_display_images_in_ui)
        {
            bv::PushConstantRange frag_push_constant_range{
                .stage_flags = VK_SHADER_STAGE_FRAGMENT_BIT,
                .offset = 0,
                .size = sizeof(UiPassFragPushConstants)
            };

            uip_pipeline_layout = bv::PipelineLayout::create(
                state.device,
                bv::PipelineLayoutConfig{
                    .flags = 0,
                    .set_layouts = { uip_descriptor_set_layout },
                    .push_constant_ranges = { frag_push_constant_range }
                }
            );
        }

        // UI pass: graphics pipeline
        if (will_display_images_in_ui)
        {
            bv::Viewport viewport{
                .x = 0.f,
                .y = 0.f,
                .width = (float)uip_framebuf->config().width,
                .height = (float)uip_framebuf->config().height,
                .min_depth = 0.f,
                .max_depth = 1.f
            };

            bv::Rect2d scissor{
                .offset = { 0, 0 },
                .extent = {
                    uip_framebuf->config().width,
                    uip_framebuf->config().height
                }
            };

            bv::ViewportState viewport_state{
                .viewports = { viewport },
                .scissors = { scissor }
            };

            bv::RasterizationState rasterization_state{
                .depth_clamp_enable = false,
                .rasterizer_discard_enable = false,
                .polygon_mode = VK_POLYGON_MODE_FILL,
                .cull_mode = VK_CULL_MODE_BACK_BIT,
                .front_face = VK_FRONT_FACE_CLOCKWISE,
                .depth_bias_enable = false,
                .depth_bias_constant_factor = 0.f,
                .depth_bias_clamp = 0.f,
                .depth_bias_slope_factor = 0.f,
                .line_width = 1.f
            };

            bv::MultisampleState multisample_state{
                .rasterization_samples = VK_SAMPLE_COUNT_1_BIT,
                .sample_shading_enable = false,
                .min_sample_shading = 1.f,
                .sample_mask = {},
                .alpha_to_coverage_enable = false,
                .alpha_to_one_enable = false
            };

            bv::DepthStencilState depth_stencil_state{
                .flags = 0,
                .depth_test_enable = false,
                .depth_write_enable = false,
                .depth_compare_op = VK_COMPARE_OP_LESS,
                .depth_bounds_test_enable = false,
                .stencil_test_enable = false,
                .front = {},
                .back = {},
                .min_depth_bounds = 0.f,
                .max_depth_bounds = 1.f
            };

            bv::ColorBlendAttachment color_blend_attachment{
                .blend_enable = false,
                .src_color_blend_factor = VK_BLEND_FACTOR_ONE,
                .dst_color_blend_factor = VK_BLEND_FACTOR_ZERO,
                .color_blend_op = VK_BLEND_OP_ADD,
                .src_alpha_blend_factor = VK_BLEND_FACTOR_ONE,
                .dst_alpha_blend_factor = VK_BLEND_FACTOR_ZERO,
                .alpha_blend_op = VK_BLEND_OP_ADD,
                .color_write_mask =
                VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT
            };

            bv::ColorBlendState color_blend_state{
                .flags = 0,
                .logic_op_enable = false,
                .logic_op = VK_LOGIC_OP_COPY,
                .attachments = { color_blend_attachment },
                .blend_constants = { 0.f, 0.f, 0.f, 0.f }
            };

            uip_graphics_pipeline = bv::GraphicsPipeline::create(
                state.device,
                bv::GraphicsPipelineConfig{
                    .flags = 0,
                    .stages = {
                        fullscreen_quad_vert_shader_stage,
                        uip_frag_shader_stage
                    },
                    .vertex_input_state = bv::VertexInputState{
                        .binding_descriptions = {},
                        .attribute_descriptions = {}
                    },
                    .input_assembly_state = bv::InputAssemblyState{
                        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                        .primitive_restart_enable = false
                    },
                    .tessellation_state = std::nullopt,
                    .viewport_state = viewport_state,
                    .rasterization_state = rasterization_state,
                    .multisample_state = multisample_state,
                    .depth_stencil_state = depth_stencil_state,
                    .color_blend_state = color_blend_state,
                    .dynamic_states = {},
                    .layout = uip_pipeline_layout,
                    .render_pass = uip_render_pass,
                    .subpass_index = 0,
                    .base_pipeline = std::nullopt
                }
            );
        }

        // UI pass: fence
        uip_fence = bv::Fence::create(state.device, 0);
    }

    void GridWarper::create_ui_pass_descriptor_set(
        const bv::ImageViewPtr& image_view,
        VkImageLayout image_layout,
        bv::DescriptorSetPtr& out_descriptor_set
    )
    {
        out_descriptor_set = bv::DescriptorPool::allocate_set(
            uip_descriptor_pool,
            uip_descriptor_set_layout
        );

        bv::DescriptorImageInfo img_info{
            .sampler = sampler,
            .image_view = image_view,
            .image_layout = image_layout
        };

        std::vector<bv::WriteDescriptorSet> descriptor_writes;

        descriptor_writes.push_back({
            .dst_set = out_descriptor_set,
            .dst_binding = 0,
            .dst_array_element = 0,
            .descriptor_count = 1,
            .descriptor_type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .image_infos = { img_info },
            .buffer_infos = {},
            .texel_buffer_views = {}
            });

        bv::DescriptorSet::update_sets(state.device, descriptor_writes, {});
    }

    bv::CommandBufferPtr GridWarper::create_grid_warp_pass_cmd_buf(
        bool hires,
        size_t thread_idx
    )
    {
        bv::CommandBufferPtr cmd_buf = bv::CommandPool::allocate_buffer(
            state.cmd_pools[thread_idx],
            VK_COMMAND_BUFFER_LEVEL_PRIMARY
        );
        cmd_buf->begin(0);

        VkClearValue clear_val{};
        clear_val.color = { { 0.f, 0.f, 0.f, 0.f } };

        auto& framebuf = (hires ? gwp_framebuf_hires : gwp_framebuf);

        VkRenderPassBeginInfo render_pass_info{
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .pNext = nullptr,
            .renderPass = gwp_render_pass->handle(),
            .framebuffer = framebuf->handle(),
            .renderArea = VkRect2D{
                .offset = { 0, 0 },
                .extent = {
                    framebuf->config().width,
                    framebuf->config().height
                }
            },
            .clearValueCount = 1,
            .pClearValues = &clear_val
        };
        vkCmdBeginRenderPass(
            cmd_buf->handle(),
            &render_pass_info,
            VK_SUBPASS_CONTENTS_INLINE
        );

        vkCmdBindPipeline(
            cmd_buf->handle(),
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            gwp_graphics_pipeline->handle()
        );

        VkBuffer vk_vertex_buf = vertex_buf->handle();
        VkDeviceSize vertex_buf_offset = 0;
        vkCmdBindVertexBuffers(
            cmd_buf->handle(),
            0,
            1,
            &vk_vertex_buf,
            &vertex_buf_offset
        );

        vkCmdBindIndexBuffer(
            cmd_buf->handle(),
            index_buf->handle(),
            0,
            VK_INDEX_TYPE_UINT32
        );

        VkViewport viewport{
            .x = 0.f,
            .y = 0.f,
            .width = (float)framebuf->config().width,
            .height = (float)framebuf->config().height,
            .minDepth = 0.f,
            .maxDepth = 1.f
        };
        vkCmdSetViewport(cmd_buf->handle(), 0, 1, &viewport);

        VkRect2D scissor{
            .offset = { 0, 0 },
            .extent = { framebuf->config().width,framebuf->config().height }
        };
        vkCmdSetScissor(cmd_buf->handle(), 0, 1, &scissor);

        auto vk_descriptor_set = gwp_descriptor_set->handle();
        vkCmdBindDescriptorSets(
            cmd_buf->handle(),
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            gwp_pipeline_layout->handle(),
            0,
            1,
            &vk_descriptor_set,
            0,
            nullptr
        );

        vkCmdPushConstants(
            cmd_buf->handle(),
            gwp_pipeline_layout->handle(),
            VK_SHADER_STAGE_FRAGMENT_BIT,
            0,
            sizeof(GridWarpPassFragPushConstants),
            &gwp_frag_push_constants
        );

        vkCmdDrawIndexed(
            cmd_buf->handle(),
            n_triangle_vertices,
            1,
            0,
            0,
            0
        );

        vkCmdEndRenderPass(cmd_buf->handle());

        cmd_buf->end();

        return cmd_buf;
    }

    bv::CommandBufferPtr GridWarper::create_difference_pass_cmd_buf(
        size_t thread_idx
    )
    {
        bv::CommandBufferPtr cmd_buf = bv::CommandPool::allocate_buffer(
            state.cmd_pools[thread_idx],
            VK_COMMAND_BUFFER_LEVEL_PRIMARY
        );
        cmd_buf->begin(0);

        VkClearValue clear_val{};
        clear_val.color = { { 0.f, 0.f, 0.f, 0.f } };

        VkRenderPassBeginInfo render_pass_info{
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .pNext = nullptr,
            .renderPass = dfp_render_pass->handle(),
            .framebuffer = dfp_framebuf->handle(),
            .renderArea = VkRect2D{
                .offset = { 0, 0 },
                .extent = {
                    dfp_framebuf->config().width,
                    dfp_framebuf->config().height
                }
            },
            .clearValueCount = 1,
            .pClearValues = &clear_val
        };
        vkCmdBeginRenderPass(
            cmd_buf->handle(),
            &render_pass_info,
            VK_SUBPASS_CONTENTS_INLINE
        );

        vkCmdBindPipeline(
            cmd_buf->handle(),
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            dfp_graphics_pipeline->handle()
        );

        auto vk_descriptor_set = dfp_descriptor_set->handle();
        vkCmdBindDescriptorSets(
            cmd_buf->handle(),
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            dfp_pipeline_layout->handle(),
            0,
            1,
            &vk_descriptor_set,
            0,
            nullptr
        );

        vkCmdPushConstants(
            cmd_buf->handle(),
            dfp_pipeline_layout->handle(),
            VK_SHADER_STAGE_FRAGMENT_BIT,
            0,
            sizeof(DifferencePassFragPushConstants),
            &dfp_frag_push_constants
        );

        vkCmdDraw(
            cmd_buf->handle(),
            6,
            1,
            0,
            0
        );

        vkCmdEndRenderPass(cmd_buf->handle());

        // memory barrier to wait for the render pass to finish rendering to the
        // difference image before we do mipmapping.
        VkImageMemoryBarrier difference_img_memory_barrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = difference_img->handle(),
            .subresourceRange = VkImageSubresourceRange{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        };
        vkCmdPipelineBarrier(
            cmd_buf->handle(),
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &difference_img_memory_barrier
        );

        // generate mipmaps. this will also add a barrier so that the
        // copy-to-buffer operation below waits for the mipmaps to be generated.
        generate_mipmaps(
            state,
            cmd_buf,
            difference_img,
            true,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_TRANSFER_READ_BIT
        );

        // copy the last mip level (which is 1x1) to the average difference
        // buffer.
        VkBufferImageCopy copy_region{
            .bufferOffset = 0,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource = VkImageSubresourceLayers{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .mipLevel = difference_img->config().mip_levels - 1,
                .baseArrayLayer = 0,
                .layerCount = 1
            },
            .imageOffset = { 0, 0, 0 },
            .imageExtent = { 1, 1, 1 }
        };
        vkCmdCopyImageToBuffer(
            cmd_buf->handle(),
            difference_img->handle(),
            VK_IMAGE_LAYOUT_GENERAL,
            avg_difference_buf->handle(),
            1,
            &copy_region
        );

        cmd_buf->end();

        return cmd_buf;
    }

    bv::CommandBufferPtr GridWarper::create_ui_pass_cmd_buf(
        const bv::DescriptorSetPtr& descriptor_set,
        size_t thread_idx
    )
    {
        bv::CommandBufferPtr cmd_buf = bv::CommandPool::allocate_buffer(
            state.cmd_pools[thread_idx],
            VK_COMMAND_BUFFER_LEVEL_PRIMARY
        );
        cmd_buf->begin(0);

        VkClearValue clear_val{};
        clear_val.color = { { 0.f, 0.f, 0.f, 0.f } };

        VkRenderPassBeginInfo render_pass_info{
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .pNext = nullptr,
            .renderPass = uip_render_pass->handle(),
            .framebuffer = uip_framebuf->handle(),
            .renderArea = VkRect2D{
                .offset = { 0, 0 },
                .extent = {
                    uip_framebuf->config().width,
                    uip_framebuf->config().height
                }
            },
            .clearValueCount = 1,
            .pClearValues = &clear_val
        };
        vkCmdBeginRenderPass(
            cmd_buf->handle(),
            &render_pass_info,
            VK_SUBPASS_CONTENTS_INLINE
        );

        vkCmdBindPipeline(
            cmd_buf->handle(),
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            uip_graphics_pipeline->handle()
        );

        auto vk_descriptor_set = descriptor_set->handle();
        vkCmdBindDescriptorSets(
            cmd_buf->handle(),
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            uip_pipeline_layout->handle(),
            0,
            1,
            &vk_descriptor_set,
            0,
            nullptr
        );

        vkCmdPushConstants(
            cmd_buf->handle(),
            uip_pipeline_layout->handle(),
            VK_SHADER_STAGE_FRAGMENT_BIT,
            0,
            sizeof(UiPassFragPushConstants),
            &uip_frag_push_constants
        );

        vkCmdDraw(
            cmd_buf->handle(),
            6,
            1,
            0,
            0
        );

        vkCmdEndRenderPass(cmd_buf->handle());

        cmd_buf->end();

        return cmd_buf;
    }

}
