#pragma once

#include "common.hpp"
#include "app_state.hpp"

namespace img_aligner
{

    inline void print(std::string_view s)
    {
        std::cout << s << std::flush;
    }

    inline void println(std::string_view s = "")
    {
        std::cout << s << std::endl;
    }

    template<typename... Args>
    void fprint(fmt::format_string<Args...> format, Args&&... args)
    {
        std::cout << fmt::vformat(format.get(), fmt::make_format_args(args...));
        std::cout.flush();
    }

    template<typename... Args>
    void fprintln(fmt::format_string<Args...> format, Args&&... args)
    {
        std::cout << fmt::vformat(format.get(), fmt::make_format_args(args...));
        std::cout << std::endl;
    }

    const std::filesystem::path& exec_dir(
        const std::optional<std::filesystem::path>& new_value = std::nullopt
    );

    std::vector<uint8_t> read_file(const std::filesystem::path& path);

    void open_url(std::string_view url);

    void clear_console();

    template<
        typename T,
        T* (*stbi_load_fn)(const char*, int*, int*, int*, int)
    >
    T* stbi_load_throw_generic(
        char const* filename,
        int* x,
        int* y,
        int* comp,
        int req_comp
    )
    {
        auto result = stbi_load_fn(filename, x, y, comp, req_comp);
        if (!result)
        {
            throw std::runtime_error(fmt::format(
                "failed to load image from file \"{}\": {}",
                filename,
                stbi_failure_reason()
            ).c_str());
        }
        return result;
    }

    static stbi_uc* stbi_load_throw(
        char const* filename,
        int* x,
        int* y,
        int* comp,
        int req_comp
    )
    {
        return stbi_load_throw_generic<stbi_uc, stbi_load>(
            filename, x, y, comp, req_comp
        );
    }

    static float* stbi_loadf_throw(
        char const* filename,
        int* x,
        int* y,
        int* comp,
        int req_comp
    )
    {
        return stbi_load_throw_generic<float, stbi_loadf>(
            filename, x, y, comp, req_comp
        );
    }

    void load_image(
        AppState& state,
        const std::filesystem::path& path,
        bv::ImagePtr& img,
        bv::MemoryChunkPtr& img_mem,
        bv::ImageViewPtr& imgview
    );

    void save_image(
        AppState& state,
        const bv::ImagePtr& img,
        const std::filesystem::path& path,
        float mul = 1.f
    );

}
