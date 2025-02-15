#pragma once

#include "common.hpp"
#include "ui_pass.hpp"
#include "grid_warp.hpp"

namespace img_aligner
{

    class App
    {
    public:
        App() = default;
        void run();

    private:
        static constexpr auto ERROR_DIALOG_TITLE = "Error";
        static constexpr auto BASE_IMAGE_NAME = "Base Image";
        static constexpr auto TARGET_IMAGE_NAME = "Target Image";

        AppState state;

        ImFont* font = nullptr;
        ImFont* font_bold = nullptr;

        float ui_scale = 1.f;
        bool ui_scale_updated = false;

        int selected_image_idx = 0;
        float image_viewer_zoom = 1.f;
        float image_viewer_exposure = 0.f;
        bool image_viewer_use_flim = false;

        bool preview_grid = false;

        std::vector<std::string> current_errors;

        // base image, mipmapped
        bv::ImagePtr base_img = nullptr;
        bv::MemoryChunkPtr base_img_mem = nullptr;
        bv::ImageViewPtr base_imgview = nullptr;

        // target image, mipmapped
        bv::ImagePtr target_img = nullptr;
        bv::MemoryChunkPtr target_img_mem = nullptr;
        bv::ImageViewPtr target_imgview = nullptr;

        grid_warp::Params grid_warp_params;
        std::unique_ptr<grid_warp::GridWarper> grid_warper = nullptr;

        std::unique_ptr<UiPass> ui_pass = nullptr;
        bool need_to_run_ui_pass = false;

        void init();
        void main_loop();
        void cleanup();

    private:
        void init_window();
        void init_context();
        void setup_debug_messenger();
        void create_surface();
        void pick_physical_device();
        void create_logical_device();
        void create_memory_bank();
        void create_command_pools();
        void create_imgui_descriptor_pool();
        void init_imgui_vk_window_data();
        void init_imgui();

        void recreate_ui_pass();
        void recreate_image(
            bv::ImagePtr& img,
            bv::MemoryChunkPtr& img_mem,
            bv::ImageViewPtr& imgview,
            uint32_t width,
            uint32_t height,
            std::span<float> pixels_rgba
        );

        void layout_controls();
        void layout_misc();
        void layout_image_viewer();

        void setup_imgui_style();
        void update_ui_scale_reload_fonts_and_style();

        void imgui_div();
        void imgui_horiz_div();
        void imgui_bold(std::string_view s);
        bool imgui_combo(
            const std::string& label,
            const std::vector<std::string>& items,
            int* selected_idx,
            bool full_width
        );
        void imgui_tooltip(std::string_view s);
        void imgui_dialogs();
        ImVec2 dialog_button_size();

        void render_frame(ImDrawData* draw_data);
        void present_frame();

        bool browse_and_load_image(
            bv::ImagePtr& img,
            bv::MemoryChunkPtr& img_mem,
            bv::ImageViewPtr& imgview
        );

        void ui_pass_select_image(std::string_view name);

    };

}
