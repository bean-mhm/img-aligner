#pragma once

#include "misc/common.hpp"
#include "misc/app_state.hpp"
#include "misc/constants.hpp"
#include "misc/io.hpp"
#include "misc/numbers.hpp"
#include "misc/vk_utils.hpp"

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

        // scale the RGB values of the image (this is separate from exposure but
        // they can be applied together. note that exposure is exponential but
        // this is a linear multiplier).
        float mul;

    private:
        bool single_channel;

        UiPass* parent_ui_pass;

        // descriptor set to use with the UI pass
        bv::DescriptorSetPtr ui_pass_ds = nullptr;

        UiImageInfo(
            std::string name,
            uint32_t width,
            uint32_t height,
            float mul,
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
    // inside a UiImageInfo) to an internal display image while also applying
    // view transforms.
    class UiPass
    {
    public:
        UiPass(
            AppState& state,
            uint32_t max_width,
            uint32_t max_height,
            const bv::QueuePtr& queue
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

        constexpr std::vector<UiImageInfo>& images()
        {
            return _images;
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
            float mul,
            bool single_channel
        );

        void clear_images();

        // render a given image to the display image. provided image must have
        // been created within this UI pass by calling its add_image() function.
        void run(
            const UiImageInfo& image,
            float exposure,
            bool use_flim,
            const bv::QueuePtr& queue
        );

        // call ImGui::Image() with the appropraite arguments.
        // NOTE: the image must be already rendered to the display image by
        // calling run().
        void draw_imgui_image(
            const UiImageInfo& image,
            float scale
        ) const;

    private:
        static constexpr uint32_t MAX_UI_IMAGES = 32;

        AppState& state;

        uint32_t _max_width;
        uint32_t _max_height;

        bv::SamplerPtr sampler = nullptr;

        // whenever we wanna display an image, we'll render it to the display
        // image. when displaying it with ImGui::Image(), we'll crop it at the
        // bottom left corner to only show the appropriate region since this
        // image is using the maximum width and height to fit every image.
        bv::ImagePtr display_img;
        bv::MemoryChunkPtr display_img_mem;
        bv::ImageViewPtr display_imgview;

        // descriptor set for ImGui::Image() to display the display image
        VkDescriptorSet imgui_descriptor_set;

        // descriptor set layout and pool
        bv::DescriptorSetLayoutPtr descriptor_set_layout = nullptr;
        bv::DescriptorPoolPtr descriptor_pool = nullptr;

        // render pass stuff
        bv::RenderPassPtr render_pass = nullptr;
        bv::FramebufferPtr framebuf;
        bv::PipelineLayoutPtr pipeline_layout = nullptr;
        bv::GraphicsPipelinePtr graphics_pipeline = nullptr;
        bv::FencePtr fence = nullptr;

        // a list of images we wanna display by rendering to the display image
        std::vector<UiImageInfo> _images;

    };

}
