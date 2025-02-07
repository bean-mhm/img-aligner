#pragma once

#include "common.hpp"
#include "grid_warp.hpp"

namespace img_aligner
{

    class App
    {
    public:
        App() = default;
        void run();

    private:
        AppState state;

        grid_warp::Params grid_warp_params;
        std::unique_ptr<grid_warp::GridWarper> grid_warper = nullptr;

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

        void layout_image_viewer();
        void layout_misc();
        void layout_controls();

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

        void render_frame(ImDrawData* draw_data);
        void present_frame();

    };

}
