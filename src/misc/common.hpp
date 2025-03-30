#pragma once

#include <iostream>
#include <fstream>
#include <filesystem>
#include <format>
#include <string>
#include <vector>
#include <array>
#include <span>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <atomic>
#include <unordered_map>
#include <set>
#include <limits>
#include <algorithm>
#include <optional>
#include <chrono>
#include <random>
#include <type_traits>
#include <stdexcept>
#include <cmath>
#include <cstdlib>
#include <cstdint>

#include "CLI11/CLI11.hpp"

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_vulkan.h"

#include "vulkan/vulkan.h"
#include "vulkan/vk_enum_string_helper.h"
#include "GLFW/glfw3.h"

#include "beva/beva.hpp"

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

#include "stb/stb_image.h"
#include "stb/stb_image_write.h"

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#define WINDOWS
#endif

// access element in 2D array with row-major ordering
#define ACCESS_2D(arr, ix, iy, res_x) ((arr)[(ix) + (iy) * (res_x)])
#define INDEX_2D(ix, iy, res_x) ((ix) + (iy) * (res_x))

namespace img_aligner
{

    // zero out a vector's capacity to actually free the memory
    template<typename T>
    void clear_vec(std::vector<T>& vec)
    {
        vec.clear();
        std::vector<T>().swap(vec);
    }

    template <typename T>
    std::basic_string<T> lowercase(std::basic_string<T> s)
    {
        std::transform(
            s.begin(),
            s.end(),
            s.begin(),
            [](const T v)
            {
                return static_cast<T>(std::tolower(v));
            }
        );
        return s;
    }

    template <typename T>
    std::basic_string<T> uppercase(std::basic_string<T> s)
    {
        std::transform(
            s.begin(),
            s.end(),
            s.begin(),
            [](const T v)
            {
                return static_cast<T>(std::toupper(v));
            }
        );
        return s;
    }

    static void cli_add_toggle(
        CLI::App& cli_app,
        const std::string& flag_name,
        bool& flag_result,
        const std::string& flag_description = "toggle flag"
    )
    {
        cli_app.add_flag_callback(
            flag_name,
            [&flag_result]()
            {
                flag_result = !flag_result;
            },
            std::format("{} (default: {})", flag_description, flag_result)
        );
    }

}
