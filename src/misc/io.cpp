#include "io.hpp"

#ifdef WINDOWS
#define NOMINMAX
#include <Windows.h>
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

namespace img_aligner
{

    const std::filesystem::path& exec_dir(
        const std::filesystem::path& new_value
    )
    {
        static std::filesystem::path dir;
        if (!new_value.empty())
        {
            dir = new_value;
        }
        else if (dir.empty())
        {
            throw std::runtime_error("executable directory has not been set");
        }
        return dir;
    }

    std::vector<uint8_t> read_file(const std::filesystem::path& path)
    {
        std::ifstream f(path, std::ios::ate | std::ios::binary);
        if (!f.is_open())
        {
            throw std::runtime_error(std::format(
                "failed to read file \"{}\"",
                path.string()
            ).c_str());
        }

        size_t size_in_chars = (size_t)f.tellg();
        size_t size_in_bytes = size_in_chars * sizeof(char);

        std::vector<uint8_t> buf(size_in_bytes);
        f.seekg(0);
        f.read(reinterpret_cast<char*>(buf.data()), size_in_chars);
        f.close();

        return buf;
    }

    void open_url(std::string_view url)
    {
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
        ShellExecuteA(NULL, "open", url.data(), NULL, NULL, SW_SHOWNORMAL);
#elif __APPLE__
        std::system(std::format("open \"{}\"", url).c_str());
#elif __linux__
        std::system(std::format("xdg-open \"{}\"", url).c_str());
#else
        throw std::exception(std::format(
            "{} is not implemented for this platform",
            __FUNCTION__
        ).c_str());
#endif
    }

    void clear_console()
    {
#if defined(_WIN32) || defined(_WIN64) || defined(WINDOWS)
        std::system("cls");
#else
        std::system("clear");
#endif
    }

    stbi_uc* stbi_load_throw(
        char const* filename,
        int* x,
        int* y,
        int* comp,
        int req_comp
    )
    {
        auto result = stbi_load(
            filename,
            x,
            y,
            comp,
            req_comp
        );
        if (!result)
        {
            throw std::runtime_error(std::format(
                "failed to load image from file \"{}\": {}",
                filename,
                stbi_failure_reason()
            ).c_str());
        }
        return result;
    }

}
