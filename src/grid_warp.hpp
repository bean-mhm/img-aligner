#pragma once

#include "common.hpp"

namespace img_aligner::grid_warp
{

    struct GridVertex
    {
        // the positions below are all normalized in the 0 to 1 range but they
        // can also have negative values or values more than 1 for padded cells.

        // position of the grid point after being warped. this defines where the
        // vertex will physically be on the warped image.
        glm::vec2 warped_pos;

        // original position of the grid point when it was created. this defines
        // where we'll sample from the base image.
        glm::vec2 orig_pos;

        static const bv::VertexInputBindingDescription binding;
        bool operator==(const GridVertex& other) const;
    };

    struct GridWarpPassFragPushConstants
    {
        float base_img_mul = 1.f;
    };

    struct DifferencePassFragPushConstants
    {
        float target_img_mul = 1.f;
    };

    struct UiPassFragPushConstants
    {
        float img_mul = 1.f;
        int32_t use_flim = 0;
    };

    struct Params
    {
        uint32_t img_width = 1;
        uint32_t img_height = 1;
        std::span<float> base_img_pixels_rgba;
        std::span<float> target_img_pixels_rgba;
        uint32_t grid_res_smallest_axis = 12;
        float grid_padding = .25f;
        uint32_t intermediate_res_smallest_axis = 800;
        bool will_display_images_in_ui = false;
    };

    class GridWarper
    {
    public:
        GridWarper(
            AppState& state,
            const Params& params,
            size_t thread_idx
        );
        ~GridWarper();

        // run the grid warp pass. if hires is set to true, the warped image
        // will be rendered to warped_hires_img, otherwise warped_img will be
        // used.
        void run_grid_warp_pass(bool hires, size_t thread_idx);

        // run the difference pass and return the average difference (error)
        // between warped_img and target_img. the average is calculated by
        // generating mipmaps for the difference image which has a square
        // resolution of a power of 2.
        float run_difference_pass(size_t thread_idx);

        // run the UI pass. this will render one of the images to ui_img. said
        // image can be selected by changing ui_pass_selected_ds.
        void run_ui_pass(size_t thread_idx);

        // descriptor set for ImGUI's Vulkan implementation to display the UI
        // image using ImGui::Image(). this will be nullptr if
        // will_display_images_in_ui is false.
        VkDescriptorSet ui_img_ds_imgui = nullptr;

        // this defines which descriptor set we'll use for the next UI pass, and
        // therefore which image will ultimately be displayed in the UI. the
        // application is in charge of changing this to point to one of the
        // UI pass desciptor sets below and then calling run_ui_pass() to update
        // the UI image. after that, it can use ImGui::Image() with
        // ui_img_ds_imgui to display the UI image.
        bv::DescriptorSetPtr ui_pass_selected_ds = nullptr;

        // the application should set ui_pass_selected_ds to point to one of
        // these descriptor sets to choose which image to display in the UI.
        bv::DescriptorSetPtr ui_pass_ds_base_img = nullptr;
        bv::DescriptorSetPtr ui_pass_ds_target_img = nullptr;
        bv::DescriptorSetPtr ui_pass_ds_warped_img = nullptr;
        bv::DescriptorSetPtr ui_pass_ds_warped_hires_img = nullptr;
        bv::DescriptorSetPtr ui_pass_ds_difference_img = nullptr;

    private:
        void create_vertex_and_index_buffer_and_generate_vertices(
            size_t thread_idx
        );
        void create_sampler_and_images(
            std::span<float> base_img_pixels_rgba,
            std::span<float> target_img_pixels_rgba,
            size_t thread_idx
        );
        void create_avg_difference_buffer();
        void create_passes();

        void create_ui_pass_descriptor_set(
            const bv::ImageViewPtr& image_view,
            VkImageLayout image_layout,
            bv::DescriptorSetPtr& out_descriptor_set
        );

        bv::CommandBufferPtr create_grid_warp_pass_cmd_buf(
            bool hires,
            size_t thread_idx
        );
        bv::CommandBufferPtr create_difference_pass_cmd_buf(size_t thread_idx);
        bv::CommandBufferPtr create_ui_pass_cmd_buf(
            const bv::DescriptorSetPtr& descriptor_set,
            size_t thread_idx
        );

    private:
        static constexpr VkFormat RGBA_FORMAT = VK_FORMAT_R32G32B32A32_SFLOAT;
        static constexpr VkFormat R_FORMAT = VK_FORMAT_R32_SFLOAT;
        static constexpr VkFormat UI_FORMAT = VK_FORMAT_R16G16B16A16_SFLOAT;

        AppState& state;

        uint32_t img_width = 1;
        uint32_t img_height = 1;

        uint32_t intermediate_res_x = 1;
        uint32_t intermediate_res_y = 1;

        uint32_t grid_res_x = 1;
        uint32_t grid_res_y = 1;

        uint32_t padded_grid_res_x = 1;
        uint32_t padded_grid_res_y = 1;

        bool will_display_images_in_ui = false;

        // vertex buffer for the grid vertices, host-visible and host-coherent
        // because we'll keep moving the vertices in every iteration.
        bv::BufferPtr vertex_buf = nullptr;
        bv::MemoryChunkPtr vertex_buf_mem = nullptr;
        GridVertex* vertex_buf_mapped = nullptr;

        // index buffer for the grid vertices
        uint32_t n_triangle_vertices = 0;
        bv::BufferPtr index_buf = nullptr;
        bv::MemoryChunkPtr index_buf_mem = nullptr;

        // the one sampler we use for all images
        bv::SamplerPtr sampler = nullptr;

        // base image at original resolution, mipmapped
        bv::ImagePtr base_img = nullptr;
        bv::MemoryChunkPtr base_img_mem = nullptr;
        bv::ImageViewPtr base_imgview = nullptr;

        // target image at original resolution, mipmapped. the target image is
        // only used to calculate the difference (error) between the warped
        // image and the target image.
        bv::ImagePtr target_img = nullptr;
        bv::MemoryChunkPtr target_img_mem = nullptr;
        bv::ImageViewPtr target_imgview = nullptr;

        // grid warped image at intermediate resolution, no mipmapping. the base
        // image will be rendered to this image but it'll be downsampled and
        // warped.
        bv::ImagePtr warped_img = nullptr;
        bv::MemoryChunkPtr warped_img_mem = nullptr;
        bv::ImageViewPtr warped_imgview = nullptr;

        // grid warped image at original resolution, no mipmapping. this will
        // only be updated after optimization is stopped.
        bv::ImagePtr warped_hires_img = nullptr;
        bv::MemoryChunkPtr warped_hires_img_mem = nullptr;
        bv::ImageViewPtr warped_hires_imgview = nullptr;

        // logarithmic difference image (difference between the warped image and
        // the target image), square resolution which is equal to the upper
        // power of 2 of the intermediate resolution, mipmapped so we can get
        // the averaged error.
        uint32_t difference_res_square = 1;
        bv::ImagePtr difference_img = nullptr;
        bv::MemoryChunkPtr difference_img_mem = nullptr;
        bv::ImageViewPtr difference_imgview = nullptr;
        bv::ImageViewPtr difference_imgview_first_mip = nullptr;

        // the UI image. whenever we wanna display any of the images, we'll
        // render it to this image while also applying the sRGB OETF.
        bv::ImagePtr ui_img = nullptr;
        bv::MemoryChunkPtr ui_img_mem = nullptr;
        bv::ImageViewPtr ui_imgview = nullptr;

        // the average difference buffer is a host-visible buffer into which we
        // copy the last mip level of the difference image which contains a
        // single float value representing the average difference (error)
        // between warped_img and target_img.
        bv::BufferPtr avg_difference_buf = nullptr;
        bv::MemoryChunkPtr avg_difference_buf_mem = nullptr;
        float* avg_difference_buf_mapped = nullptr;

        // NOTE: there are 3 types of passes:
        // 1. grid warp pass
        //    - renders to warped_img or final_img (we make 2 framebuffers and
        //      use the appropriate one).
        //    - samples base_img
        //    - uses the grid vertex buffer
        // 2. difference pass
        //    - renders to difference_img
        //    - samples warped_img and target_img
        //    - does not use a vertex buffer. instead, generates vertices for a
        //      "full-screen" quad in the vertex shader.
        // 3. UI pass (for displaying the images)
        //    - renders to ui_img
        //    - samples any of the other images
        //    - does not use a vertex buffer

        // grid warp pass: descriptor stuff
        bv::DescriptorSetLayoutPtr gwp_descriptor_set_layout = nullptr;
        bv::DescriptorPoolPtr gwp_descriptor_pool = nullptr;
        bv::DescriptorSetPtr gwp_descriptor_set;

        // grid warp pass
        bv::RenderPassPtr gwp_render_pass = nullptr;
        bv::FramebufferPtr gwp_framebuf = nullptr;
        bv::FramebufferPtr gwp_framebuf_hires = nullptr;
        bv::PipelineLayoutPtr gwp_pipeline_layout = nullptr;
        bv::GraphicsPipelinePtr gwp_graphics_pipeline = nullptr;
        GridWarpPassFragPushConstants gwp_frag_push_constants;
        bv::FencePtr gwp_fence = nullptr;

        // difference pass: descriptor stuff
        bv::DescriptorSetLayoutPtr dfp_descriptor_set_layout = nullptr;
        bv::DescriptorPoolPtr dfp_descriptor_pool = nullptr;
        bv::DescriptorSetPtr dfp_descriptor_set;

        // difference pass
        bv::RenderPassPtr dfp_render_pass = nullptr;
        bv::FramebufferPtr dfp_framebuf = nullptr;
        bv::PipelineLayoutPtr dfp_pipeline_layout = nullptr;
        bv::GraphicsPipelinePtr dfp_graphics_pipeline = nullptr;
        DifferencePassFragPushConstants dfp_frag_push_constants;
        bv::FencePtr dfp_fence = nullptr;

        // UI pass: descriptor stuff
        bv::DescriptorSetLayoutPtr uip_descriptor_set_layout = nullptr;
        bv::DescriptorPoolPtr uip_descriptor_pool = nullptr;

        // UI pass
        bv::RenderPassPtr uip_render_pass = nullptr;
        bv::FramebufferPtr uip_framebuf = nullptr;
        bv::PipelineLayoutPtr uip_pipeline_layout = nullptr;
        bv::GraphicsPipelinePtr uip_graphics_pipeline = nullptr;
        UiPassFragPushConstants uip_frag_push_constants;
        bv::FencePtr uip_fence = nullptr;

    };

}
