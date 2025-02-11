#pragma once

#include "common.hpp"

namespace img_aligner
{

    class UiPass;

    // this contains the descriptor set used in the UI pass, along with more
    // information about the image.
    class UiImageInfo
    {
    public:
        std::string name;
        uint32_t width;
        uint32_t height;

    private:
        bool single_channel;

        UiPass* parent_ui_pass;

        // descriptor set to use with the UI pass
        bv::DescriptorSetPtr ui_pass_ds = nullptr;

        UiImageInfo(
            std::string name,
            uint32_t width,
            uint32_t height,
            bool single_channel,
            UiPass* parent_ui_pass,
            bv::DescriptorSetPtr ui_pass_ds
        );

        friend class UiPass;

    };

    struct UiPassFragPushConstants
    {
        float img_mul = 1.f;
        int32_t use_flim = 0;
        int32_t single_channel = 0;
    };

    // the UI pass renders a given image (provided through a descriptor set
    // inside a UiImageInfo) to an internal image while also applying view
    // transforms. it has just as many internal images as the maximum number of
    // frames that can be in flight at a time.
    class UiPass
    {
    public:
        // this constructor must be called on the main thread (thread_idx=0)
        UiPass(
            AppState& state,
            uint32_t max_width,
            uint32_t max_height,
            uint32_t max_frames_in_flight
        );
        ~UiPass();

        constexpr uint32_t max_width() const
        {
            return _max_width;
        }

        constexpr uint32_t max_height() const
        {
            return _max_height;
        }

        constexpr uint32_t max_frames_in_flight() const
        {
            return _max_frames_in_flight;
        }

        constexpr const std::vector<UiImageInfo>& images() const
        {
            return _images;
        }

        const UiImageInfo& add_image(
            const bv::ImageViewPtr& view,
            VkImageLayout layout,
            std::string name,
            uint32_t width,
            uint32_t height,
            bool single_channel
        );

        void clear_images();

        // add commands to a command buffer to render an image into the
        // appropriate display image. the command buffer must be already created
        // and in recording state.
        // the provided image must have been created
        // within this UI pass by calling its add_image() function.
        void record_commands(
            VkCommandBuffer cmd_buf,
            size_t frame_idx,
            const UiImageInfo& image,
            float exposure,
            bool use_flim
        );

        // call ImGui::Image() with the appropraite arguments
        // NOTE: the image must be already rendered to the corresponding display
        // image by calling record_commands() and providing the command buffer
        // used for rendering the UI. this function assumes that's already
        // handled.
        void draw_imgui_image(
            uint32_t frame_idx,
            const UiImageInfo& image,
            float scale
        ) const;

    private:
        static constexpr VkFormat UI_FORMAT = VK_FORMAT_R16G16B16A16_SFLOAT;
        static constexpr uint32_t MAX_UI_IMAGES = 32;

        AppState& state;

        uint32_t _max_width;
        uint32_t _max_height;
        uint32_t _max_frames_in_flight;

        bv::SamplerPtr sampler = nullptr;

        // display images. whenever we wanna display an image, we'll render it
        // to one of these images, and the index depends on the current
        // swapchain frame index which has a range of 0 to
        // (max frames in flight - 1). keep in mind that when displaying one of
        // these, we'll crop it at the bottom left corner to only show the
        // appropriate region since these images are using the maximum width and
        // height to fit every image.
        std::vector<bv::ImagePtr> display_images;
        std::vector<bv::MemoryChunkPtr> display_image_mems;
        std::vector<bv::ImageViewPtr> display_image_views;

        // descriptor sets for ImGui::Image() to display the display images
        std::vector<VkDescriptorSet> imgui_descriptor_sets;

        // descriptor set layout and pool
        bv::DescriptorSetLayoutPtr descriptor_set_layout = nullptr;
        bv::DescriptorPoolPtr descriptor_pool = nullptr;

        // render pass stuff
        bv::RenderPassPtr render_pass = nullptr;
        std::vector<bv::FramebufferPtr> framebufs;
        bv::PipelineLayoutPtr pipeline_layout = nullptr;
        bv::GraphicsPipelinePtr graphics_pipeline = nullptr;

        // a list of images we wanna display by rendering to one of the display
        // images
        std::vector<UiImageInfo> _images;

    };

}
