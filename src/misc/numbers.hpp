#pragma once

#include "common.hpp"

namespace img_aligner
{

    // closest upper power of 2 to an integer.
    // examples: 11 -> 16, 3000 -> 4096, 256 -> 256
    uint32_t upper_power_of_2(uint32_t n);

    // how many times an integer can be divided by 2 (or bit-shifted to the
    // right) until it reaches 0.
    uint32_t round_log2(uint32_t n);

    // linear interpolation
    template<typename V, std::floating_point T>
    T lerp(const V& a, const V& b, T t)
    {
        return a + t * (b - a);
    }

    template<typename T, T(*exp_fn)(T) = std::exp>
    T unnormalized_gaussian(T standard_deviation, T x)
    {
        T a = x / standard_deviation;
        return exp_fn((T)(-.5) * a * a);
    }

    constexpr ImVec2 imvec_from_glm(const glm::vec2& v)
    {
        return { v.x, v.y };
    }

    constexpr glm::vec2 imvec_to_glm(const ImVec2& v)
    {
        return { v.x, v.y };
    }

    constexpr bool vec2_is_outside_01(const glm::vec2& v)
    {
        return v.x < 0.f || v.y < 0.f || v.x > 1.f || v.y > 1.f;
    }

    template<std::floating_point T>
    int32_t determine_precision(
        T v,
        int32_t max_significant_digits = 5, // can only reduce decimal digits
        int32_t min_precision = 1,
        int32_t max_precision = 11
    )
    {
        // this may be negative and it's intentional
        int32_t n_integral_digits = (int32_t)std::floor(std::log10(v)) + 1;

        min_precision = std::max(min_precision, 0);
        max_precision = std::max(max_precision, min_precision);

        return std::clamp(
            max_significant_digits - n_integral_digits,
            min_precision,
            max_precision
        );
    }

    template<std::integral T>
    std::string to_str(T v)
    {
        return std::to_string(v);
    }

    template<std::floating_point T>
    std::string to_str(
        T v,
        int32_t max_significant_digits = 5, // can only reduce decimal digits
        int32_t min_precision = 1,
        int32_t max_precision = 11,
        int32_t* out_precision = nullptr,
        int32_t* out_n_trailing_zeros = nullptr
    )
    {
        int32_t precision = determine_precision(
            v,
            max_significant_digits,
            min_precision,
            max_precision
        );

        std::string s = fmt::format("{0:.{1}f}", v, precision);

        // remove redundant trailing zeros after decimal point
        int32_t n_trailing_zeros = 0;
        if (precision > 0)
        {
            while (s.ends_with('0'))
            {
                s = s.substr(0, s.length() - 1);
                n_trailing_zeros++;
            }
            if (s.ends_with('.'))
            {
                s = s.substr(0, s.length() - 1);
            }
        }

        if (out_precision)
        {
            *out_precision = precision;
        }
        if (out_n_trailing_zeros)
        {
            *out_n_trailing_zeros = n_trailing_zeros;
        }

        if (s == "-0")
        {
            s = "0";
        }

        return s;
    }

    // use higher precision for floats
    template<std::floating_point T>
    std::string to_str_hp(T v)
    {
        return to_str(v, 15, 0, 20);
    }

    // use highest precision for floats
    template<std::floating_point T>
    std::string to_str_hhp(T v)
    {
        std::string s = fmt::format(
            std::is_same_v<T, float> ? "{0:.50f}" : "{0:.326f}",
            v
        );

        // remove redundant zeros after decimal point
        while (s.ends_with('0'))
        {
            s = s.substr(0, s.length() - 1);
        }
        if (s.ends_with('.'))
        {
            s = s.substr(0, s.length() - 1);
        }

        if (s == "-0")
        {
            s = "0";
        }

        return s;
    }

    // no difference for non-float types
    template<std::integral T>
    std::string to_str_hp(T v)
    {
        return std::to_string(v);
    }

    // no difference for non-float types
    template<std::integral T>
    std::string to_str_hhp(T v)
    {
        return std::to_string(v);
    }

    // unused because ImGui is buggy
    template<std::floating_point T>
    std::string determine_precision_for_imgui(
        T v,
        int32_t max_significant_digits = 5, // can only reduce decimal digits
        int32_t min_precision = 1,
        int32_t max_precision = 11
    )
    {
        int32_t precision = 0;
        int32_t n_trailing_zeros = 0;
        std::string s = to_str(
            v,
            max_significant_digits,
            min_precision,
            max_precision,
            &precision,
            &n_trailing_zeros
        );

        // don't remove literally all trailing zeros for ImGui because it bugs
        // out sometimes.
        return fmt::format(
            "%.{}f",
            std::max(precision - n_trailing_zeros, min_precision)
        );
    }

}
