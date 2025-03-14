#pragma once

#include "common.hpp"
#include "ui_pass.hpp"
#include "grid_warp.hpp"

namespace img_aligner
{

    struct GridWarpOptimizationParams
    {
        float max_warp_strength = .0001f;
        float min_change_in_cost_in_last_n_iters = .00001f;
        uint32_t max_iters = 10000;
        float max_runtime_sec = 600.f;
    };

    struct GridWarpOptimizationInfo
    {
        size_t n_iters = 0; // num. total iterations
        size_t n_good_iters = 0; // num. iterations where the cost decreased

        std::vector<float> cost_history;
        float change_in_cost_in_last_n_iters = FLT_MAX;

        std::optional<TimePoint> start_time;

        // elapsed time accumulated from previous optimization runs. while
        // optimization is running, this should be added to the elapsed time
        // since start_time.
        float accum_elapsed = 0.f;
    };

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

        // base image, mipmapped
        bv::ImagePtr base_img = nullptr;
        bv::MemoryChunkPtr base_img_mem = nullptr;
        bv::ImageViewPtr base_imgview = nullptr;

        // target image, mipmapped
        bv::ImagePtr target_img = nullptr;
        bv::MemoryChunkPtr target_img_mem = nullptr;
        bv::ImageViewPtr target_imgview = nullptr;

        // grid warper params and itself
        grid_warp::Params grid_warp_params;
        std::unique_ptr<grid_warp::GridWarper> grid_warper = nullptr;

        // grid warp optimization thread
        bool is_optimizing = false;
        std::unique_ptr<std::jthread> optimization_thread = nullptr;
        bool optimization_thread_stop = false;
        std::shared_mutex optimization_mutex;
        std::atomic_bool need_the_optimization_mutex = false;

        // grid warp optimization stuff
        GridWarpOptimizationParams optimization_params;
        GridWarpOptimizationInfo optimization_info;
        std::shared_mutex optimization_info_mutex;

        // copy of the grid vertices to use for grid preview in the UI
        std::vector<grid_warp::GridVertex> grid_vertices_copy_for_ui_preview;

        // last time we updated the UI stuff when optimization is running.
        TimePoint last_ui_update_when_optimizing_time;

        void init();
        void main_loop();
        void cleanup();

    private:
        // exclusively UI-related

        ImFont* font = nullptr;
        ImFont* font_bold = nullptr;

        // list of errors to display in the error dialog
        std::vector<std::string> current_errors;

        // used for displaying linear (HDR) images in the UI
        std::unique_ptr<UiPass> ui_pass = nullptr;
        bool need_to_run_ui_pass = false;

        float ui_scale = 1.f;
        bool ui_scale_updated = false;

        int selected_image_idx = 0;
        bool image_viewer_fit = true;
        float image_viewer_zoom = 1.f;
        float image_viewer_exposure = 0.f;
        bool image_viewer_use_flim = false;

        bool preview_grid = true;

    private:
        // initialization functions
        void init_window();
        void init_context();
        void setup_debug_messenger();
        void create_surface();
        void pick_physical_device();
        void create_logical_device();
        void create_memory_bank();
        void create_imgui_descriptor_pool();
        void init_imgui_vk_window_data();
        void init_imgui();

    private:
        void recreate_image(
            bv::ImagePtr& img,
            bv::MemoryChunkPtr& img_mem,
            bv::ImageViewPtr& imgview,
            uint32_t width,
            uint32_t height,
            std::span<float> pixels_rgba
        );

        void load_image(
            std::string_view filename,
            bv::ImagePtr& img,
            bv::MemoryChunkPtr& img_mem,
            bv::ImageViewPtr& imgview
        );

        bool try_recreate_grid_warper(std::string* out_error = nullptr);
        void destroy_grid_warper(
            bool recreate_ui_pass_if_destroyed_grid_warper
        );

        void start_optimization();
        void stop_optimization();

    private:
        // exclusively UI-related

        void recreate_ui_pass();
        void copy_grid_vertices_for_ui_preview();

        void layout_controls();
        void layout_misc();
        void layout_image_viewer();

        void setup_imgui_style();
        void update_ui_scale_reload_fonts_and_style();

        void imgui_div();
        void imgui_small_div();
        void imgui_horiz_div();
        void imgui_bold(std::string_view s);
        bool imgui_combo(
            const std::string& label,
            const std::vector<std::string>& items,
            int* selected_idx,
            bool full_width
        );
        bool imgui_button_full_width(const char* label);
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

        void select_ui_pass_image(std::string_view name);

    };

}
