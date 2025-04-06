#pragma once

#include "misc/common.hpp"
#include "misc/app_state.hpp"
#include "misc/constants.hpp"
#include "misc/io.hpp"
#include "misc/numbers.hpp"
#include "misc/time.hpp"
#include "misc/transform2d.hpp"
#include "misc/vk_utils.hpp"

#include "ui_pass.hpp"
#include "grid_warp.hpp"

namespace img_aligner
{

    enum class CliGridWarpOptimizationStatsMode : uint32_t
    {
        Disabled,
        AtEnd,
        Realtime
    };

    // values parsed from command line arguments that don't belong anywhere
    // else. for example, --max-iters can be bound to
    // optimization_params.max_runtime_sec but these values have no home.
    struct CliParams
    {
        bool flag_help = false;
        bool flag_version = false;
        bool flag_silent = false;

        std::string base_img_path;
        std::string target_img_path;
        std::string output_img_path;
        std::string difference_img_before_opt_path;
        std::string difference_img_after_opt_path;
        std::string metadata_path;

        CliGridWarpOptimizationStatsMode optimization_stats_mode =
            CliGridWarpOptimizationStatsMode::AtEnd;
    };

    struct GridWarpOptimizationParams
    {
        // transform optimization
        float scale_jitter = 1.01f;
        float rotation_jitter = .8f;
        float offset_jitter = .005f;
        uint32_t n_transform_optimization_iters = 200;

        // warp optimization

        float warp_strength = .0001f;
        float warp_strength_decay_rate = 0.f;
        float min_warp_strength = .00001f;

        float min_change_in_cost_in_last_n_iters = .00001f;
        uint32_t max_iters = 10000;
        float max_runtime_sec = 600.f;

        // calculate warp strength based on number of iterations (apply decaying
        // and clamping).
        float calc_warp_strength(size_t n_iters);
    };

    enum class GridWarpOptimizationStopReason : uint32_t
    {
        None,
        ManuallyStopped,
        LowChangeInCost,
        ReachedMaxIters,
        ReachedMaxRuntime
    };

    const char* GridWarpOptimizationStopReason_to_str(
        GridWarpOptimizationStopReason reason
    );
    const char* GridWarpOptimizationStopReason_to_str_friendly(
        GridWarpOptimizationStopReason reason
    );

    struct GridWarpOptimizationInfo
    {
        size_t n_iters = 0; // num. total iterations
        size_t n_good_iters = 0; // num. iterations where the cost decreased

        // last jittered grid transform that was potentially optimized in
        // transform optimization. if transform optimization was disabled, this
        // will be equal to the current grid transform.
        Transform2d last_jittered_transform;

        std::vector<float> cost_history;
        float change_in_cost_in_last_n_iters = FLT_MAX;

        std::optional<TimePoint> start_time;

        // elapsed time accumulated from previous optimization runs. while
        // optimization is running, this should be added to the elapsed time
        // since start_time.
        float accum_elapsed = 0.f;

        GridWarpOptimizationStopReason stop_reason =
            GridWarpOptimizationStopReason::None;
    };

    struct MetadataExportOptions
    {
        bool params_and_res = true;
        bool optimization_info = true;
        bool grid_vertices = false;
        bool pretty_print = true;
    };

    class App
    {
    public:
        App(int argc, char** argv);

        void run();

    private:
        static constexpr auto ERROR_DIALOG_TITLE = "Error";
        static constexpr auto BASE_IMAGE_NAME = "Base Image";
        static constexpr auto TARGET_IMAGE_NAME = "Target Image";

        // these can be used for physical_device_idx
        static constexpr int32_t PHYSICAL_DEVICE_IDX_AUTO = -2;
        static constexpr int32_t PHYSICAL_DEVICE_IDX_PROMPT = -1;

        int argc;
        char** argv;
        std::unique_ptr<CLI::App> cli_app = nullptr;
        CliParams cli_params;

        AppState state;

        bool init_was_called = false;
        int32_t physical_device_idx = PHYSICAL_DEVICE_IDX_AUTO;

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
        Transform2d grid_transform;
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

        // export options
        bool export_warped_img_undo_base_img_mul = true;
        MetadataExportOptions metadata_export_options;

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

        std::vector<float> warp_strength_plot;

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
        void parse_command_line();
        void handle_command_line();

        void export_metadata(
            const std::filesystem::path& path
        );

        void recreate_grid_warper();
        void destroy_grid_warper(
            bool recreate_ui_pass_if_destroyed_grid_warper
        );

        void start_optimization();
        void stop_optimization();

        void print_optimization_statistics(bool clear);

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
        void browse_and_save_image(const bv::ImagePtr& img, float mul = 1.f);
        void browse_and_export_metadata();

        void select_ui_pass_image(std::string_view name);

        template<typename T>
        bool imgui_slider_or_drag(
            std::string_view label, // "Example"
            std::string_view tag, // "##example"
            std::string_view tooltip, // use "" to disable
            T* v,
            T min, // set both to 0 to disable (only for drag)
            T max, // set both to 0 to disable (only for drag)
            float drag_speed = -1.f, // negative: draw slider instead of drag
            int32_t precision = -1, // negative: use default value
            bool logarithmic = false,
            uint32_t n_components = 1, // >1 components will use fixed precision
            bool force_clamp = false,
            bool wrap_around = false
        )
        {
            // default precision
            if (precision < 0)
            {
                precision = 3;
            }

            if (!label.empty())
            {
                ImGui::TextWrapped(label.data());
                imgui_tooltip(tooltip);
            }

            ImGuiDataType data_type;
            if constexpr (std::is_same_v<T, float>)
            {
                data_type = ImGuiDataType_Float;
            }
            else if constexpr (std::is_same_v<T, double>)
            {
                data_type = ImGuiDataType_Double;
            }
            else if constexpr (std::is_same_v<T, int32_t>)
            {
                data_type = ImGuiDataType_S32;
            }
            else if constexpr (std::is_same_v<T, uint32_t>)
            {
                data_type = ImGuiDataType_U32;
            }
            else if constexpr (std::is_same_v<T, int64_t>)
            {
                data_type = ImGuiDataType_S64;
            }
            else if constexpr (std::is_same_v<T, uint64_t>)
            {
                data_type = ImGuiDataType_U64;
            }
            else
            {
                throw std::invalid_argument(
                    "unsupported type for slider or drag"
                );
            }

            std::string format;
            if constexpr (std::is_floating_point_v<T>)
            {
                if (n_components > 1)
                {
                    format = std::format("%.{}f", precision);
                }
                else
                {
                    // unused because ImGui is buggy
                    /*format = determine_precision_for_imgui(
                        *v,
                        5,
                        min_precision
                    );*/

                    format = std::format("%.{}f", precision);
                }
            }

            ImGuiSliderFlags flags = ImGuiSliderFlags_NoRoundToFormat;
            if (logarithmic)
            {
                flags |= ImGuiSliderFlags_Logarithmic;
            }
            if (force_clamp)
            {
                flags |= ImGuiSliderFlags_AlwaysClamp;
            }
            if (wrap_around)
            {
                flags |= ImGuiSliderFlags_WrapAround;
            }

            ImGui::SetNextItemWidth(-FLT_MIN);
            if (drag_speed < 0.f)
            {
                return ImGui::SliderScalarN(
                    tag.data(),
                    data_type,
                    v,
                    (int)n_components,
                    &min,
                    &max,
                    format.empty() ? nullptr : format.c_str(),
                    flags
                );
            }
            else
            {
                bool no_clamp = (min == (T)0 && max == (T)0);
                return ImGui::DragScalarN(
                    tag.data(),
                    data_type,
                    v,
                    (int)n_components,
                    drag_speed,
                    no_clamp ? nullptr : &min,
                    no_clamp ? nullptr : &max,
                    format.empty() ? nullptr : format.c_str(),
                    flags
                );
            }
        }

    };

}
