#pragma once

#include "common.hpp"
#include "ui_pass.hpp"

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

    struct Params
    {
        bv::ImageViewWPtr base_imgview;
        bv::ImageViewWPtr target_imgview;

        uint32_t grid_res_area = 256;
        float grid_padding = .25f;

        uint32_t intermediate_res_area = 920000;
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

        void add_images_to_ui_pass(UiPass& ui_pass);

        constexpr const uint32_t get_padded_grid_res_x() const
        {
            return padded_grid_res_x;
        }

        constexpr const uint32_t get_padded_grid_res_y() const
        {
            return padded_grid_res_y;
        }

        const GridVertex* get_vertices() const
        {
            return vertex_buf_mapped;
        }

    private:
        void create_vertex_and_index_buffer_and_generate_vertices(
            size_t thread_idx
        );
        void create_sampler_and_images(size_t thread_idx);
        void create_avg_difference_buffer();
        void create_passes();

        bv::CommandBufferPtr create_grid_warp_pass_cmd_buf(
            bool hires,
            size_t thread_idx
        );
        bv::CommandBufferPtr create_difference_pass_cmd_buf(size_t thread_idx);

    private:
        AppState& state;

        // base and target images provided in the constructor
        bv::ImageViewWPtr base_imgview;
        bv::ImageViewWPtr target_imgview;

        // size of the base and target images
        uint32_t img_width = 1;
        uint32_t img_height = 1;

        uint32_t intermediate_res_x = 1;
        uint32_t intermediate_res_y = 1;

        uint32_t grid_res_x = 1;
        uint32_t grid_res_y = 1;

        uint32_t padded_grid_res_x = 1;
        uint32_t padded_grid_res_y = 1;

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

        // the average difference buffer is a host-visible buffer into which we
        // copy the last mip level of the difference image which contains a
        // single float value representing the average difference (error)
        // between warped_img and target_img.
        bv::BufferPtr avg_difference_buf = nullptr;
        bv::MemoryChunkPtr avg_difference_buf_mem = nullptr;
        float* avg_difference_buf_mapped = nullptr;

        // NOTE: there are 2 types of passes:
        // 1. grid warp pass
        //    - renders to warped_img or warped_hires_img (we make 2
        //      framebuffers and use the appropriate one).
        //    - samples base_img
        //    - uses the grid vertex buffer
        // 2. difference pass
        //    - renders to difference_img
        //    - samples warped_img and target_img
        //    - does not use a vertex buffer. instead, generates vertices for a
        //      "full-screen" quad in the vertex shader.

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

    };

}
