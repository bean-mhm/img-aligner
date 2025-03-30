#pragma once

#include "common.hpp"

namespace img_aligner
{

    static constexpr auto APP_TITLE = "img-aligner";
    static constexpr auto APP_VERSION = "0.1.0-alpha";
    static constexpr auto APP_GITHUB_URL =
        "https://github.com/bean-mhm/img-aligner";

    static constexpr uint32_t INITIAL_WIDTH = 1024;
    static constexpr uint32_t INITIAL_HEIGHT = 720;

    static constexpr ImVec4 COLOR_BG{ .04f, .03f, .08f, 1.f };
    static constexpr ImVec4 COLOR_IMAGE_BORDER{ .05f, .05f, .05f, 1.f };
    static constexpr ImVec4 COLOR_INFO_TEXT{ .33f, .74f, .91f, 1.f };
    static constexpr ImVec4 COLOR_WARNING_TEXT{ .94f, .58f, .28f, 1.f };
    static constexpr ImVec4 COLOR_ERROR_TEXT{ .95f, .3f, .23f, 1.f };

    static constexpr float FONT_SIZE = 20.f;
    static constexpr auto FONT_PATH = "fonts/Outfit-Regular.ttf";
    static constexpr auto FONT_BOLD_PATH = "fonts/Outfit-Bold.ttf";

    static constexpr VkFormat UI_DISPLAY_IMG_FORMAT =
        VK_FORMAT_R16G16B16A16_SFLOAT;

    // enable Vulkan validation layer and debug messages. this is usually only
    // available if the Vulkan SDK is installed on the user's machine, which is
    // rarely the case for a regular user, so disable this for final releases.
    static constexpr bool ENABLE_VALIDATION_LAYER = false;

    // the interval at which to run the UI pass and make a new copy of the grid
    // vertices for previewing in the UI, when grid warp optimization is
    // running.
    static constexpr float GRID_WARP_OPTIMIZATION_UI_UPDATE_INTERVAL = .7f;

    static constexpr float GRID_WARP_OPTIMIZATION_CLI_REALTIME_STATS_INTERVAL =
        .3f;

    static constexpr size_t GRID_WARP_OPTIMIZATION_WARP_STRENGTH_PLOT_N_ITERS
        = 5000;

}
