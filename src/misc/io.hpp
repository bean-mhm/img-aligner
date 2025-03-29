#pragma once

#include "common.hpp"

namespace img_aligner
{

    const std::filesystem::path& exec_dir(
        const std::filesystem::path& new_value = {}
    );

    std::vector<uint8_t> read_file(const std::filesystem::path& path);

    void open_url(std::string_view url);

    void clear_console();

    stbi_uc* stbi_load_throw(
        char const* filename,
        int* x,
        int* y,
        int* comp,
        int req_comp
    );

}
