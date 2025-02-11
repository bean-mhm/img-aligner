#include "ui_pass.hpp"

namespace img_aligner
{

    UiImageInfo::UiImageInfo(
        std::string name,
        uint32_t width,
        uint32_t height,
        bool single_channel,
        UiPass* parent_ui_pass,
        bv::DescriptorSetPtr ui_pass_ds
    )
        : name(std::move(name)),
        width(width),
        height(height),
        parent_ui_pass(parent_ui_pass),
        single_channel(single_channel),
        ui_pass_ds(ui_pass_ds)
    {}

    UiPass::UiPass(
        AppState& state,
        uint32_t max_width,
        uint32_t max_height,
        uint32_t max_frames_in_flight
    )
        : state(state),
        _max_width(max_width),
        _max_height(max_height),
        _max_frames_in_flight(max_frames_in_flight)
    {
        // create sampler
        sampler = bv::Sampler::create(
            state.device,
            {
                .flags = 0,
                .mag_filter = VK_FILTER_LINEAR,
                .min_filter = VK_FILTER_LINEAR,
                .mipmap_mode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
                .address_mode_u = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                .address_mode_v = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                .address_mode_w = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
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

        // create display images
        auto cmd_buf = begin_single_time_commands(state, true, 0);
        for (size_t i = 0; i < _max_frames_in_flight; i++)
        {
            display_images.push_back(nullptr);
            display_image_mems.push_back(nullptr);
            display_image_views.push_back(nullptr);

            create_image(
                state,
                _max_width,
                _max_height,
                1,
                VK_SAMPLE_COUNT_1_BIT,
                UI_FORMAT,
                VK_IMAGE_TILING_OPTIMAL,

                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                | VK_IMAGE_USAGE_SAMPLED_BIT,

                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                display_images.back(),
                display_image_mems.back()
            );
            display_image_views.back() = create_image_view(
                state,
                display_images.back(),
                UI_FORMAT,
                VK_IMAGE_ASPECT_COLOR_BIT,
                1
            );
            transition_image_layout(
                cmd_buf,
                display_images.back(),
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_GENERAL,
                1
            );
        }
        end_single_time_commands(state, cmd_buf, true);

        // create descriptor sets for ImGui::Image()
        for (size_t i = 0; i < _max_frames_in_flight; i++)
        {
            imgui_descriptor_sets.push_back(ImGui_ImplVulkan_AddTexture(
                sampler->handle(),
                display_image_views[i]->handle(),
                VK_IMAGE_LAYOUT_GENERAL
            ));
        }

        // descriptor set layout
        {
            bv::DescriptorSetLayoutBinding binding_img{
                .binding = 0,
                .descriptor_type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptor_count = 1,
                .stage_flags = VK_SHADER_STAGE_FRAGMENT_BIT,
                .immutable_samplers = { sampler }
            };

            descriptor_set_layout = bv::DescriptorSetLayout::create(
                state.device,
                {
                    .flags = 0,
                    .bindings = { binding_img }
                }
            );
        }

        // descriptor pool
        {
            // 1 image in each descriptor set * MAX_UI_IMAGES sets at maximum
            bv::DescriptorPoolSize image_pool_size{
                .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptor_count = MAX_UI_IMAGES
            };

            descriptor_pool = bv::DescriptorPool::create(
                state.device,
                {
                    .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
                    .max_sets = MAX_UI_IMAGES,
                    .pool_sizes = { image_pool_size }
                }
            );
        }

        // render pass
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

            render_pass = bv::RenderPass::create(
                state.device,
                bv::RenderPassConfig{
                    .flags = 0,
                    .attachments = { color_attachment },
                    .subpasses = { subpass },
                    .dependencies = { dependency }
                }
            );
        }

        // framebuffers
        for (size_t i = 0; i < _max_frames_in_flight; i++)
        {
            framebufs.push_back(bv::Framebuffer::create(
                state.device,
                bv::FramebufferConfig{
                    .flags = 0,
                    .render_pass = render_pass,
                    .attachments = { display_image_views[i] },
                    .width = _max_width,
                    .height = _max_height,
                    .layers = 1
                }
            ));
        }

        // pipeline layout
        {
            bv::PushConstantRange frag_push_constant_range{
                .stage_flags = VK_SHADER_STAGE_FRAGMENT_BIT,
                .offset = 0,
                .size = sizeof(UiPassFragPushConstants)
            };

            pipeline_layout = bv::PipelineLayout::create(
                state.device,
                bv::PipelineLayoutConfig{
                    .flags = 0,
                    .set_layouts = { descriptor_set_layout },
                    .push_constant_ranges = { frag_push_constant_range }
                }
            );
        }

        // shaders

        bv::ShaderModulePtr fullscreen_quad_vert_shader_module = nullptr;
        bv::ShaderStage fullscreen_quad_vert_shader_stage{};

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

        // graphics pipeline
        {
            bv::Viewport viewport{
                .x = 0.f,
                .y = 0.f,
                .width = (float)_max_width,
                .height = (float)_max_height,
                .min_depth = 0.f,
                .max_depth = 1.f
            };

            bv::Rect2d scissor{
                .offset = { 0, 0 },
                .extent = { _max_width, _max_height }
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

            graphics_pipeline = bv::GraphicsPipeline::create(
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
                    .layout = pipeline_layout,
                    .render_pass = render_pass,
                    .subpass_index = 0,
                    .base_pipeline = std::nullopt
                }
            );
        }
    }

    UiPass::~UiPass()
    {
        clear_images();

        graphics_pipeline = nullptr;
        pipeline_layout = nullptr;
        framebufs.clear();
        render_pass = nullptr;

        descriptor_pool = nullptr;
        descriptor_set_layout = nullptr;

        for (VkDescriptorSet ds : imgui_descriptor_sets)
        {
            ImGui_ImplVulkan_RemoveTexture(ds);
        }

        display_images.clear();
        display_image_mems.clear();
        display_image_views.clear();
    }

    const UiImageInfo& UiPass::add_image(
        const bv::ImageViewPtr& view,
        VkImageLayout layout,
        std::string name,
        uint32_t width,
        uint32_t height,
        bool single_channel
    )
    {
        if (_images.size() >= MAX_UI_IMAGES)
        {
            throw std::runtime_error("maximum number of UI images reached");
        }

        if (width > _max_width || height > _max_height)
        {
            throw std::invalid_argument(std::format(
                "UI pass' max width and/or height ({}x{}) isn't large enough "
                "to fit image ({}x{})",
                _max_width,
                _max_height,
                width,
                height
            ).c_str());
        }

        auto ds = bv::DescriptorPool::allocate_set(
            descriptor_pool,
            descriptor_set_layout
        );

        bv::DescriptorImageInfo img_info{
            .sampler = sampler,
            .image_view = view,
            .image_layout = layout
        };

        bv::WriteDescriptorSet descriptor_write{
            .dst_set = ds,
            .dst_binding = 0,
            .dst_array_element = 0,
            .descriptor_count = 1,
            .descriptor_type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .image_infos = { img_info },
            .buffer_infos = {},
            .texel_buffer_views = {}
        };

        bv::DescriptorSet::update_sets(
            state.device,
            { descriptor_write },
            {}
        );

        _images.push_back(UiImageInfo{
            name,
            width,
            height,
            single_channel,
            this,
            ds
            });
        return _images.back();
    }

    void UiPass::clear_images()
    {
        _images.clear();
    }

    void UiPass::record_commands(
        VkCommandBuffer cmd_buf,
        size_t frame_idx,
        const UiImageInfo& image,
        float exposure,
        bool use_flim
    )
    {
        if (image.parent_ui_pass != this)
        {
            throw std::invalid_argument(
                "provided UI image info is not created with \"this\" UI pass"
            );
        }

        VkClearValue clear_val{};
        clear_val.color = { { 0.f, 0.f, 0.f, 0.f } };

        const auto& framebuf = framebufs.at(frame_idx);
        VkRenderPassBeginInfo render_pass_info{
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .pNext = nullptr,
            .renderPass = render_pass->handle(),
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
            cmd_buf,
            &render_pass_info,
            VK_SUBPASS_CONTENTS_INLINE
        );

        vkCmdBindPipeline(
            cmd_buf,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            graphics_pipeline->handle()
        );

        auto vk_descriptor_set = image.ui_pass_ds->handle();
        vkCmdBindDescriptorSets(
            cmd_buf,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipeline_layout->handle(),
            0,
            1,
            &vk_descriptor_set,
            0,
            nullptr
        );

        UiPassFragPushConstants frag_push_constants{
            .img_mul = std::exp2(exposure),
            .use_flim = use_flim ? 1 : 0,
            .single_channel = image.single_channel ? 1 : 0
        };
        vkCmdPushConstants(
            cmd_buf,
            pipeline_layout->handle(),
            VK_SHADER_STAGE_FRAGMENT_BIT,
            0,
            sizeof(UiPassFragPushConstants),
            &frag_push_constants
        );

        vkCmdDraw(
            cmd_buf,
            6,
            1,
            0,
            0
        );

        vkCmdEndRenderPass(cmd_buf);

        VkImageMemoryBarrier display_img_memory_barrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = display_images.at(frame_idx)->handle(),
            .subresourceRange = VkImageSubresourceRange{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        };
        vkCmdPipelineBarrier(
            cmd_buf,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &display_img_memory_barrier
        );
    }

    void UiPass::draw_imgui_image(
        uint32_t frame_idx,
        const UiImageInfo& image,
        float scale
    ) const
    {
        ImGui::Image(
            (ImTextureID)imgui_descriptor_sets.at(frame_idx),
            ImVec2(
                (float)image.width * scale,
                (float)image.height * scale
            ),
            {
                0,
                (float)image.height / (float)_max_height
            },
            {
                (float)image.width / (float)_max_width,
                0
            },
            { 1, 1, 1, 1 },
            COLOR_IMAGE_BORDER
        );
    }

}
