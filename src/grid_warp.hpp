#pragma once

#include "common.hpp"
#include "ui_pass.hpp"

namespace img_aligner::grid_warp
{

    static constexpr size_t N_ITERS_TO_CHECK_CHANGE_IN_COST = 100;

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

    struct CostPassFragPushConstants
    {
        glm::ivec2 cost_res;
    };

    struct CostInfo
    {
        float avg_diff; // average per-pixel logarithmic difference
        float max_local_diff; // maximum value in the cost image
    };

    struct Params
    {
        bv::ImageViewWPtr base_imgview;
        bv::ImageViewWPtr target_imgview;

        float base_img_mul = 1.f;
        float target_img_mul = 1.f;

        uint32_t grid_res_area = 200;
        float grid_padding = .05f;

        uint32_t intermediate_res_area = 1200000;
        uint32_t cost_res_area = 60;

        uint32_t rng_seed = 8191;
    };

    // there are 3 types of passes in GridWarper:
    // 1. grid warp pass
    //    - samples base_img
    //    - renders to warped_img or warped_hires_img (we make 2
    //      framebuffers and use the appropriate one).
    //    - uses the grid vertex buffer
    // 2. difference pass
    //    - samples warped_img and target_img
    //    - calculates the per-pixel logarithmic difference
    //    - renders to difference_img
    //    - does not use a vertex buffer. instead, generates vertices for a
    //      "full-screen" quad in the vertex shader.
    // 3. cost pass
    //    - samples difference_img using as many texel fetches needed to
    //      avoid aliasing.
    //    - renders to the cost image at the cost resolution
    //    - does not use a vertex buffer. instead, generates vertices for a
    //      "full-screen" quad in the vertex shader.

    class GridWarper
    {
    public:
        GridWarper(
            AppState& state,
            const Params& params,
            const bv::QueuePtr& queue
        );
        ~GridWarper();

        void run_grid_warp_pass(bool hires, const bv::QueuePtr& queue);

        // returns the cost values
        CostInfo run_difference_and_cost_pass(const bv::QueuePtr& queue);

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

        constexpr uint32_t get_n_vertices() const
        {
            return n_vertices;
        }

        constexpr const std::optional<float>& get_last_avg_diff() const
        {
            return last_avg_diff;
        }

        constexpr const std::optional<float>& get_initial_max_local_diff() const
        {
            return initial_max_local_diff;
        }

        // displace the grid vertices using an unnormalized gaussian
        // distribution with randomly generated center point, radius (standard
        // deviation), displacement direction and strength. the grid warp and
        // difference passes will then be run. if the displacement caused the
        // cost to increase, we will undo the displacement and return false,
        // otherwise we'll keep the changes and return true. ideally, you would
        // call this many times in a row to minimize the difference between the
        // warped image and the target image.
        bool optimize(float max_warp_strength, const bv::QueuePtr& queue);

    private:
        void create_vertex_and_index_buffer_and_generate_vertices(
            const bv::QueuePtr& queue
        );
        void make_copy_of_vertices();
        void restore_copy_of_vertices();
        void create_sampler_and_images(const bv::QueuePtr& queue);
        void create_passes();

        bv::CommandBufferPtr create_grid_warp_pass_cmd_buf(bool hires);
        bv::CommandBufferPtr create_difference_pass_cmd_buf();
        bv::CommandBufferPtr create_cost_pass_cmd_buf();

    private:
        AppState& state;
        std::mt19937 rng;

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

        uint32_t cost_res_x = 1;
        uint32_t cost_res_y = 1;

        // used for optimization
        std::optional<float> last_avg_diff;
        std::optional<float> initial_max_local_diff;

        // vertex buffer for the grid vertices, host-visible and host-coherent
        // because we'll keep moving the vertices in every iteration.
        uint32_t n_vertices = 0;
        bv::BufferPtr vertex_buf = nullptr;
        bv::MemoryChunkPtr vertex_buf_mem = nullptr;
        GridVertex* vertex_buf_mapped = nullptr;

        // vector to contain a copy of the vertices, ONLY used when undoing
        // grid displacement in case it increased the cost.
        std::vector<GridVertex> vertices_copy;

        // index buffer for the grid vertices
        uint32_t n_triangle_vertices = 0;
        bv::BufferPtr index_buf = nullptr;
        bv::MemoryChunkPtr index_buf_mem = nullptr;

        // the one sampler we use for all images
        bv::SamplerPtr sampler = nullptr;

        // grid warped image at intermediate resolution. the base image will be
        // rendered to this image but it'll be downsampled and warped.
        bv::ImagePtr warped_img = nullptr;
        bv::MemoryChunkPtr warped_img_mem = nullptr;
        bv::ImageViewPtr warped_imgview = nullptr;

        // grid warped image at original resolution. this will only be updated
        // after optimization is stopped.
        bv::ImagePtr warped_hires_img = nullptr;
        bv::MemoryChunkPtr warped_hires_img_mem = nullptr;
        bv::ImageViewPtr warped_hires_imgview = nullptr;

        // difference image (per-pixel logarithmic difference between the warped
        // image and the target image).
        bv::ImagePtr difference_img = nullptr;
        bv::MemoryChunkPtr difference_img_mem = nullptr;
        bv::ImageViewPtr difference_imgview = nullptr;

        // cost image. this is just a downscaled version of the difference
        // image.
        bv::ImagePtr cost_img = nullptr;
        bv::MemoryChunkPtr cost_img_mem = nullptr;
        bv::ImageViewPtr cost_imgview = nullptr;

        // host visible buffer to copy the cost image's pixels to the CPU
        bv::BufferPtr cost_buf = nullptr;
        bv::MemoryChunkPtr cost_buf_mem = nullptr;
        float* cost_buf_mapped = nullptr;

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

        // cost pass: descriptor stuff
        bv::DescriptorSetLayoutPtr csp_descriptor_set_layout = nullptr;
        bv::DescriptorPoolPtr csp_descriptor_pool = nullptr;
        bv::DescriptorSetPtr csp_descriptor_set;

        // cost pass
        bv::RenderPassPtr csp_render_pass = nullptr;
        bv::FramebufferPtr csp_framebuf = nullptr;
        bv::PipelineLayoutPtr csp_pipeline_layout = nullptr;
        bv::GraphicsPipelinePtr csp_graphics_pipeline = nullptr;
        CostPassFragPushConstants csp_frag_push_constants;
        bv::FencePtr csp_fence = nullptr;

    };

}
