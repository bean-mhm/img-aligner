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
        const bv::QueuePtr& queue
    )
        : state(state),
        rng(params.rng_seed),
        base_imgview(params.base_imgview),
        target_imgview(params.target_imgview)
    {
        if (base_imgview.expired())
        {
            throw std::invalid_argument("provided base image view has expired");
        }
        if (target_imgview.expired())
        {
            throw std::invalid_argument(
                "provided target image view has expired"
            );
        }

        auto base_imgview_locked = base_imgview.lock();
        auto target_imgview_locked = target_imgview.lock();

        if (base_imgview_locked->image().expired())
        {
            throw std::invalid_argument(
                "provided base image view's parent image has expired"
            );
        }
        if (target_imgview_locked->image().expired())
        {
            throw std::invalid_argument(
                "provided target image view's parent image has expired"
            );
        }

        auto base_extent = base_imgview_locked->image().lock()->config().extent;
        auto target_extent =
            target_imgview_locked->image().lock()->config().extent;

        if (base_extent.width != target_extent.width
            || base_extent.height != target_extent.height)
        {
            throw std::invalid_argument(std::format(
                "provided base and target images must have the same resolution "
                "instead of {}x{} and {}x{} respectively.",
                base_extent.width, base_extent.height,
                target_extent.width, target_extent.height
            ).c_str());
        }

        img_width = base_extent.width;
        img_height = base_extent.height;
        if (img_width < 1 || img_height < 1)
        {
            throw std::invalid_argument(
                "image resolution must be at least 1 pixel in every axis"
            );
        }

        // figure out the intermediate resolution

        double area_fac =
            (double)params.intermediate_res_area
            / (double)(img_width * img_height);
        double size_fac = std::clamp(std::sqrt(area_fac), 0., 1.);

        intermediate_res_x = (uint32_t)std::floor(size_fac * (double)img_width);
        intermediate_res_y =
            (uint32_t)std::floor(size_fac * (double)img_height);

        intermediate_res_x = std::clamp(
            intermediate_res_x,
            (uint32_t)1,
            img_width
        );
        intermediate_res_y = std::clamp(
            intermediate_res_y,
            (uint32_t)1,
            img_height
        );

        // figure out the cost resolution

        area_fac =
            (double)params.cost_res_area
            / (double)(img_width * img_height);
        size_fac = std::clamp(std::sqrt(area_fac), 0., 1.);

        cost_res_x = (uint32_t)std::floor(size_fac * (double)img_width);
        cost_res_y = (uint32_t)std::floor(size_fac * (double)img_height);

        cost_res_x = std::clamp(
            cost_res_x,
            (uint32_t)1,
            intermediate_res_x
        );
        cost_res_y = std::clamp(
            cost_res_y,
            (uint32_t)1,
            intermediate_res_y
        );

        // figure out the grid resolution

        area_fac =
            (double)params.grid_res_area
            / (double)(intermediate_res_x * intermediate_res_y);
        size_fac = std::clamp(std::sqrt(area_fac), 0., .5);

        grid_res_x =
            (uint32_t)std::floor(size_fac * (double)intermediate_res_x);
        grid_res_y =
            (uint32_t)std::floor(size_fac * (double)intermediate_res_y);

        grid_res_x = std::clamp(
            grid_res_x,
            (uint32_t)1,
            intermediate_res_x
        );
        grid_res_y = std::clamp(
            grid_res_y,
            (uint32_t)1,
            intermediate_res_y
        );

        // figure out the padded grid resolution. basically we add a border on
        // the outside for padding. without the bordering cells, we might get
        // black empty spaces in the warped image.

        double actual_grid_padding = std::max((double)params.grid_padding, 0.);
        double grid_res_diagonal = std::sqrt(
            (double)((grid_res_x * grid_res_x) + (grid_res_y * grid_res_y))
        );

        double absolute_padding_in_cells =
            2. * actual_grid_padding * grid_res_diagonal;

        padded_grid_res_x = (uint32_t)std::ceil(
            (double)grid_res_x + absolute_padding_in_cells
        );
        padded_grid_res_y = (uint32_t)std::ceil(
            (double)grid_res_y + absolute_padding_in_cells
        );

        // padded resolution must be original resolution plus an even number
        if ((padded_grid_res_x - grid_res_x) % 2 != 0)
            padded_grid_res_x++;
        if ((padded_grid_res_y - grid_res_y) % 2 != 0)
            padded_grid_res_y++;

        // set up push constants
        gwp_frag_push_constants.base_img_mul = params.base_img_mul;
        dfp_frag_push_constants.target_img_mul = params.target_img_mul;

        create_vertex_and_index_buffer_and_generate_vertices(queue);
        make_copy_of_vertices();
        create_sampler_and_images(queue);
        create_passes();
    }

    GridWarper::~GridWarper()
    {
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

        warped_img = nullptr;
        warped_img_mem = nullptr;
        warped_imgview = nullptr;

        warped_hires_img = nullptr;
        warped_hires_img_mem = nullptr;
        warped_hires_imgview = nullptr;

        difference_img = nullptr;
        difference_img_mem = nullptr;
        difference_imgview = nullptr;

        cost_img = nullptr;
        cost_img_mem = nullptr;
        cost_imgview = nullptr;

        cost_buf = nullptr;
        cost_buf_mem = nullptr;

        sampler = nullptr;
    }

    void GridWarper::run_grid_warp_pass(bool hires, const bv::QueuePtr& queue)
    {
        auto cmd_buf = create_grid_warp_pass_cmd_buf(hires);
        queue->submit({}, {}, { cmd_buf }, {}, gwp_fence);
        gwp_fence->wait();
        gwp_fence->reset();
    }

    CostInfo GridWarper::run_difference_and_cost_pass(const bv::QueuePtr& queue)
    {
        auto cmd_buf = create_difference_pass_cmd_buf();
        queue->submit({}, {}, { cmd_buf }, {}, dfp_fence);
        dfp_fence->wait();
        dfp_fence->reset();

        cmd_buf = create_cost_pass_cmd_buf();
        queue->submit({}, {}, { cmd_buf }, {}, csp_fence);
        csp_fence->wait();
        csp_fence->reset();

        // find average and maximum value in the cost image
        float avg_diff = 0.f;
        float max_local_diff = 0.f;
        for (size_t y = 0; y < cost_res_y; y++)
        {
            for (size_t x = 0; x < cost_res_x; x++)
            {
                float v = cost_buf_mapped[x + (y * cost_res_x)];

                avg_diff += v;

                max_local_diff = std::max(
                    max_local_diff,
                    v
                );
            }
        }
        avg_diff /= (float)(cost_res_x * cost_res_y);

        return CostInfo{
            .avg_diff = avg_diff,
            .max_local_diff = max_local_diff
        };
    }

    void GridWarper::add_images_to_ui_pass(UiPass& ui_pass)
    {
        ui_pass.add_image(
            warped_imgview,
            VK_IMAGE_LAYOUT_GENERAL,
            WARPED_IMAGE_NAME,
            warped_img->config().extent.width,
            warped_img->config().extent.height,
            1.f,
            false
        );

        ui_pass.add_image(
            warped_hires_imgview,
            VK_IMAGE_LAYOUT_GENERAL,
            WARPED_HIRES_IMAGE_NAME,
            warped_hires_img->config().extent.width,
            warped_hires_img->config().extent.height,
            1.f,
            false
        );

        ui_pass.add_image(
            difference_imgview,
            VK_IMAGE_LAYOUT_GENERAL,
            DIFFERENCE_IMAGE_NAME,
            difference_img->config().extent.width,
            difference_img->config().extent.height,
            1.f,
            true
        );

        ui_pass.add_image(
            cost_imgview,
            VK_IMAGE_LAYOUT_GENERAL,
            COST_IMAGE_NAME,
            cost_img->config().extent.width,
            cost_img->config().extent.height,
            1.f,
            true
        );
    }

    bool GridWarper::optimize(
        float warp_strength,
        const bv::QueuePtr& queue
    )
    {
        // keep track of the cost
        if (!last_avg_diff || !initial_max_local_diff)
        {
            auto cost_info = run_difference_and_cost_pass(queue);
            last_avg_diff = cost_info.avg_diff;
            initial_max_local_diff = cost_info.max_local_diff;
        }
        float old_avg_diff = *last_avg_diff;

        // make a copy of the vertices in case we decide to undo the
        // displacement
        make_copy_of_vertices();

        std::uniform_real_distribution<float> dist(0.f, 1.f);

        // warp vertices based on an unnormalized gaussian distribution

        // gaussian center
        glm::vec2 center{
            dist(rng) * (float)intermediate_res_x,
            dist(rng) * (float)intermediate_res_y
        };

        // radius = standard deviation

        float min_radius = std::max(
            (float)intermediate_res_x / (float)grid_res_x,
            (float)intermediate_res_y / (float)grid_res_y
        );
        float max_radius = .5f * (float)std::max(
            intermediate_res_x,
            intermediate_res_y
        );

        float log_min_radius = std::log(min_radius);
        float log_max_radius = std::log(max_radius);

        float log_radius = lerp(
            log_min_radius,
            log_max_radius,
            dist(rng)
        );
        float radius = std::exp(log_radius);

        // strength
        float strength = (warp_strength * dist(rng)) * radius;

        // direction
        float angle = glm::tau<float>() * dist(rng);
        glm::vec2 direction{ std::cos(angle), std::sin(angle) };

        // move vertices
        uint32_t stride_y = padded_grid_res_x + 1;
        for (uint32_t y = 0; y < padded_grid_res_y; y++)
        {
            for (uint32_t x = 0; x < padded_grid_res_x; x++)
            {
                auto& vert = vertex_buf_mapped[x + y * stride_y];

                // position in pixel space
                auto pos = vert.warped_pos * glm::vec2{
                    (float)intermediate_res_x,
                    (float)intermediate_res_y
                };

                // displace (warp)
                float displacement = strength * unnormalized_gaussian(
                    radius,
                    glm::distance(pos, center)
                );
                pos += displacement * direction;

                // convert from pixel space back to normalized space
                vert.warped_pos = pos / glm::vec2{
                    (float)intermediate_res_x,
                    (float)intermediate_res_y
                };
            }
        }

        // see if the displacement did any good (decreased the cost)
        run_grid_warp_pass(false, queue);
        auto new_cost_info = run_difference_and_cost_pass(queue);

        // undo the displacement (warping) if it wasn't good
        if (new_cost_info.avg_diff > old_avg_diff
            || new_cost_info.max_local_diff > *initial_max_local_diff)
        {
            restore_copy_of_vertices();
            return false;
        }
        else
        {
            last_avg_diff = new_cost_info.avg_diff;
        }
        return true;
    }

    void GridWarper::create_vertex_and_index_buffer_and_generate_vertices(
        const bv::QueuePtr& queue
    )
    {
        // please keep in mind that the 2D resolution of the vertex array is
        // (padded_grid_res_x + 1) by (padded_grid_res_y + 1) to account for
        // vertices at the edges. for example, for a 2x2 grid we would need 3x3
        // vertices.
        n_vertices = (padded_grid_res_x + 1) * (padded_grid_res_y + 1);
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
            for (uint32_t y = 0; y < padded_grid_res_y; y++)
            {
                for (uint32_t x = 0; x < padded_grid_res_x; x++)
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
            end_single_time_commands(cmd_buf, queue);

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

    void GridWarper::make_copy_of_vertices()
    {
        vertices_copy.resize(n_vertices);
        std::copy(
            vertex_buf_mapped,
            vertex_buf_mapped + n_vertices,
            vertices_copy.data()
        );
    }

    void GridWarper::restore_copy_of_vertices()
    {
        if (vertices_copy.size() != n_vertices)
        {
            throw std::runtime_error(
                "the vertices copy vector doesn't have the expected size"
            );
        }
        std::copy(
            vertices_copy.data(),
            vertices_copy.data() + vertices_copy.size(),
            vertex_buf_mapped
        );
    }

    void GridWarper::create_sampler_and_images(const bv::QueuePtr& queue)
    {
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

        // command buffer for image layout transitions
        auto cmd_buf = begin_single_time_commands(state, true);

        // create images and image views, and transition layouts

        // warped image uses the intermediate resolution.
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

        // high-resolution warped image uses the original resolution
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

        // difference image uses the intermediate resolution
        create_image(
            state,
            intermediate_res_x,
            intermediate_res_y,
            1,
            VK_SAMPLE_COUNT_1_BIT,
            R_FORMAT,
            VK_IMAGE_TILING_OPTIMAL,

            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,

            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            difference_img,
            difference_img_mem
        );
        difference_imgview = create_image_view(
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
            1
        );

        // cost image uses the cost resolution and a host-visible memory chunk
        create_image(
            state,
            cost_res_x,
            cost_res_y,
            1,
            VK_SAMPLE_COUNT_1_BIT,
            R_FORMAT,
            VK_IMAGE_TILING_OPTIMAL,

            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
            | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,

            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            cost_img,
            cost_img_mem
        );
        cost_imgview = create_image_view(
            state,
            cost_img,
            R_FORMAT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            1
        );
        transition_image_layout(
            cmd_buf,
            cost_img,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL,
            1
        );

        // end, submit, and wait for the command buffer
        end_single_time_commands(cmd_buf, queue);

        // cost buffer
        create_buffer(
            state,
            cost_res_x * cost_res_y * sizeof(float),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,

            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,

            cost_buf,
            cost_buf_mem
        );
        cost_buf_mapped = (float*)cost_buf_mem->mapped();
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

        bv::ShaderModulePtr csp_frag_shader_module = nullptr;
        bv::ShaderStage csp_frag_shader_stage{};

        {
            std::vector<uint8_t> shader_code = read_file(
                exec_dir() / "shaders/fullscreen_quad_vert.spv"
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
                exec_dir() / "shaders/grid_warp_pass_vert.spv"
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
                exec_dir() / "shaders/grid_warp_pass_frag.spv"
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
                exec_dir() / "shaders/difference_pass_frag.spv"
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
                exec_dir() / "shaders/cost_pass_frag.spv"
            );
            csp_frag_shader_module = bv::ShaderModule::create(
                state.device,
                std::move(shader_code)
            );
            csp_frag_shader_stage = bv::ShaderStage{
                .flags = {},
                .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                .module = csp_frag_shader_module,
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
                .load_op = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
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
                .attachments = { difference_imgview },
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

        // cost pass: descriptor set layout
        {
            bv::DescriptorSetLayoutBinding binding_difference_img{
                .binding = 0,
                .descriptor_type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptor_count = 1,
                .stage_flags = VK_SHADER_STAGE_FRAGMENT_BIT,
                .immutable_samplers = { sampler }
            };

            csp_descriptor_set_layout = bv::DescriptorSetLayout::create(
                state.device,
                {
                    .flags = 0,
                    .bindings = { binding_difference_img }
                }
            );
        }

        // cost pass: descriptor pool
        {
            // 1 image in every descriptor set * 1 set in total
            bv::DescriptorPoolSize image_pool_size{
                .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptor_count = 1
            };

            csp_descriptor_pool = bv::DescriptorPool::create(
                state.device,
                {
                    .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
                    .max_sets = 1,
                    .pool_sizes = { image_pool_size }
                }
            );
        }

        // cost pass: descriptor set
        {
            csp_descriptor_set = bv::DescriptorPool::allocate_set(
                csp_descriptor_pool,
                csp_descriptor_set_layout
            );

            bv::DescriptorImageInfo difference_img_info{
                .sampler = sampler,
                .image_view = difference_imgview,
                .image_layout = VK_IMAGE_LAYOUT_GENERAL
            };

            std::vector<bv::WriteDescriptorSet> descriptor_writes;
            descriptor_writes.push_back({
                .dst_set = csp_descriptor_set,
                .dst_binding = 0,
                .dst_array_element = 0,
                .descriptor_count = 1,
                .descriptor_type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .image_infos = { difference_img_info },
                .buffer_infos = {},
                .texel_buffer_views = {}
                });
            bv::DescriptorSet::update_sets(state.device, descriptor_writes, {});
        }

        // cost pass: render pass
        {
            bv::Attachment color_attachment{
                .flags = 0,
                .format = R_FORMAT,
                .samples = VK_SAMPLE_COUNT_1_BIT,
                .load_op = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
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

            csp_render_pass = bv::RenderPass::create(
                state.device,
                bv::RenderPassConfig{
                    .flags = 0,
                    .attachments = { color_attachment },
                    .subpasses = { subpass },
                    .dependencies = { dependency }
                }
            );
        }

        // cost pass: framebuffer
        csp_framebuf = bv::Framebuffer::create(
            state.device,
            bv::FramebufferConfig{
                .flags = 0,
                .render_pass = csp_render_pass,
                .attachments = { cost_imgview },
                .width = cost_img->config().extent.width,
                .height = cost_img->config().extent.height,
                .layers = 1
            }
        );

        // cost pass: pipeline layout
        {
            bv::PushConstantRange frag_push_constant_range{
                .stage_flags = VK_SHADER_STAGE_FRAGMENT_BIT,
                .offset = 0,
                .size = sizeof(CostPassFragPushConstants)
            };

            csp_pipeline_layout = bv::PipelineLayout::create(
                state.device,
                bv::PipelineLayoutConfig{
                    .flags = 0,
                    .set_layouts = { csp_descriptor_set_layout },
                    .push_constant_ranges = { frag_push_constant_range }
                }
            );
        }

        // cost pass: graphics pipeline
        {
            bv::Viewport viewport{
                .x = 0.f,
                .y = 0.f,
                .width = (float)csp_framebuf->config().width,
                .height = (float)csp_framebuf->config().height,
                .min_depth = 0.f,
                .max_depth = 1.f
            };

            bv::Rect2d scissor{
                .offset = { 0, 0 },
                .extent = {
                    csp_framebuf->config().width,
                    csp_framebuf->config().height
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

            csp_graphics_pipeline = bv::GraphicsPipeline::create(
                state.device,
                bv::GraphicsPipelineConfig{
                    .flags = 0,
                    .stages = {
                        fullscreen_quad_vert_shader_stage,
                        csp_frag_shader_stage
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
                    .layout = csp_pipeline_layout,
                    .render_pass = csp_render_pass,
                    .subpass_index = 0,
                    .base_pipeline = std::nullopt
                }
            );
        }

        // cost pass: fence
        csp_fence = bv::Fence::create(state.device, 0);
    }

    bv::CommandBufferPtr GridWarper::create_grid_warp_pass_cmd_buf(bool hires)
    {
        bv::CommandBufferPtr cmd_buf = begin_single_time_commands(state, true);

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
            sizeof(gwp_frag_push_constants),
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

    bv::CommandBufferPtr GridWarper::create_difference_pass_cmd_buf()
    {
        bv::CommandBufferPtr cmd_buf = begin_single_time_commands(state, true);

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
            .clearValueCount = 0,
            .pClearValues = nullptr
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
            sizeof(dfp_frag_push_constants),
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

        cmd_buf->end();

        return cmd_buf;
    }

    bv::CommandBufferPtr GridWarper::create_cost_pass_cmd_buf()
    {
        bv::CommandBufferPtr cmd_buf = begin_single_time_commands(state, true);

        VkRenderPassBeginInfo render_pass_info{
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .pNext = nullptr,
            .renderPass = csp_render_pass->handle(),
            .framebuffer = csp_framebuf->handle(),
            .renderArea = VkRect2D{
                .offset = { 0, 0 },
                .extent = {
                    csp_framebuf->config().width,
                    csp_framebuf->config().height
                }
            },
            .clearValueCount = 0,
            .pClearValues = nullptr
        };
        vkCmdBeginRenderPass(
            cmd_buf->handle(),
            &render_pass_info,
            VK_SUBPASS_CONTENTS_INLINE
        );

        vkCmdBindPipeline(
            cmd_buf->handle(),
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            csp_graphics_pipeline->handle()
        );

        auto vk_descriptor_set = csp_descriptor_set->handle();
        vkCmdBindDescriptorSets(
            cmd_buf->handle(),
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            csp_pipeline_layout->handle(),
            0,
            1,
            &vk_descriptor_set,
            0,
            nullptr
        );

        csp_frag_push_constants.cost_res = {
            (int32_t)csp_framebuf->config().width,
            (int32_t)csp_framebuf->config().height
        };
        vkCmdPushConstants(
            cmd_buf->handle(),
            csp_pipeline_layout->handle(),
            VK_SHADER_STAGE_FRAGMENT_BIT,
            0,
            sizeof(csp_frag_push_constants),
            &csp_frag_push_constants
        );

        vkCmdDraw(
            cmd_buf->handle(),
            6,
            1,
            0,
            0
        );

        vkCmdEndRenderPass(cmd_buf->handle());

        // memory barrier to wait for the rendering
        VkImageMemoryBarrier barrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = 0,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = cost_img->handle(),
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
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &barrier
        );

        // copy to cost buffer
        copy_image_to_buffer(cmd_buf, cost_img, cost_buf);

        cmd_buf->end();

        return cmd_buf;
    }

}
