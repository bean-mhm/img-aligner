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

    struct DifferencePassFragPushConstants
    {
        float warped_img_mul = 1.f;
    };

    struct WarpPass
    {
        AppState& state;

        uint32_t texture_mip_levels = 1;
        bv::ImagePtr texture_img = nullptr;
        bv::MemoryChunkPtr texture_img_mem = nullptr;
        bv::ImageViewPtr texture_imgview = nullptr;
        bv::SamplerPtr texture_sampler = nullptr;

        std::vector<GridVertex> vertices;

        bv::BufferPtr vertex_buf = nullptr;
        bv::MemoryChunkPtr vertex_buf_mem = nullptr;

        std::vector<bv::BufferPtr> uniform_bufs;
        std::vector<bv::MemoryChunkPtr> uniform_bufs_mem;
        std::vector<void*> uniform_bufs_mapped;

        bv::DescriptorPoolPtr descriptor_pool = nullptr;
        std::vector<bv::DescriptorSetPtr> descriptor_sets;

        void create_texture_image();
        void create_texture_sampler();
        void load_model();
        void create_vertex_buffer();
        void create_index_buffer();
        void create_uniform_buffers();
        void create_descriptor_pool();
        void create_descriptor_sets();
        void create_command_buffers();

        void record_command_buffer(
            const bv::CommandBufferPtr& cmd_buf,
            uint32_t img_idx,
            float elapsed
        );

        void update_uniform_buffer(uint32_t frame_idx, float elapsed);
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
    };

    class GridWarper
    {
    public:
        GridWarper(
            AppState& state,
            const Params& params
        );

    private:
        void create_vertex_and_index_buffer_and_generate_vertices();
        void create_sampler_and_images(
            std::span<float> base_img_pixels_rgba,
            std::span<float> target_img_pixels_rgba
        );

    private:
        static constexpr VkFormat RGBA_FORMAT = VK_FORMAT_R32G32B32A32_SFLOAT;
        static constexpr VkFormat R_FORMAT = VK_FORMAT_R32_SFLOAT;

        AppState& state;

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

        // logarithmic difference image (difference between the warped image and
        // the target image), square resolution which is equal to the upper
        // power of 2 of the intermediate resolution, mipmapped so we can get
        // the averaged error.
        uint32_t difference_res_square = 1;
        bv::ImagePtr difference_img = nullptr;
        bv::MemoryChunkPtr difference_img_mem = nullptr;
        bv::ImageViewPtr difference_imgview = nullptr;

        // grid warped image at original resolution, no mipmapping. this will
        // only be updated after optimization is stopped.
        bv::ImagePtr final_img = nullptr;
        bv::MemoryChunkPtr final_img_mem = nullptr;
        bv::ImageViewPtr final_imgview = nullptr;

        // NOTE: there are two types of passes:
        // 1. grid warp pass
        //    - renders to warped_img or final_img (we make 2 framebuffers and
        //      use the appropriate one).
        //    - samples base_img
        //    - uses the grid vertex buffer
        // 2. difference pass
        //    - renders to difference_img.
        //    - samples warped_img and target_img
        //    - does NOT use the vertex buffer, instead, generates vertices for
        //      a "full-screen" quad in the vertex shader.

        // grid warp pass: descriptor stuff
        bv::DescriptorSetLayoutPtr gwp_descriptor_set_layout = nullptr;
        bv::DescriptorPoolPtr gwp_descriptor_pool = nullptr;
        std::vector<bv::DescriptorSetPtr> gwp_descriptor_sets;

        // grid warp pass
        bv::PipelineLayoutPtr gwp_pipeline_layout = nullptr;
        bv::GraphicsPipelinePtr gwp_graphics_pipeline = nullptr;
        bv::RenderPassPtr gwp_render_pass = nullptr;
        bv::FramebufferPtr gwp_framebuf_warped_img = nullptr;
        bv::FramebufferPtr gwp_framebuf_final_img = nullptr;

        // difference pass: descriptor stuff
        bv::DescriptorSetLayoutPtr dfp_descriptor_set_layout = nullptr;
        bv::DescriptorPoolPtr dfp_descriptor_pool = nullptr;
        std::vector<bv::DescriptorSetPtr> dfp_descriptor_sets;

        // difference pass
        bv::PipelineLayoutPtr dfp_pipeline_layout = nullptr;
        bv::GraphicsPipelinePtr dfp_graphics_pipeline = nullptr;
        bv::RenderPassPtr dfp_render_pass = nullptr;
        bv::FramebufferPtr dfp_framebuf = nullptr;
        DifferencePassFragPushConstants dfp_frag_push_constants;

    };

}
