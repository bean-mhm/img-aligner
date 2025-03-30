#include "io.hpp"

#include "OpenEXR/ImfRgbaFile.h"
#include "OpenEXR/ImfOutputFile.h"
#include "OpenEXR/ImfArray.h"
#include "OpenEXR/ImfChannelList.h"

#ifdef WINDOWS
#define NOMINMAX
#include <Windows.h>
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include "vk_utils.hpp"

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

    void load_image(
        AppState& state,
        const std::filesystem::path& path,
        bv::ImagePtr& img,
        bv::MemoryChunkPtr& img_mem,
        bv::ImageViewPtr& imgview
    )
    {
        if (!std::filesystem::exists(path))
        {
            throw std::runtime_error("file doesn't exist");
        }
        if (std::filesystem::is_directory(path))
        {
            throw std::runtime_error(
                "provided path is a directory, not a file"
            );
        }

        int32_t width = 0, height = 0;
        std::vector<float> pixels_rgbaf32;

        // get the file extension
        std::string file_ext = lowercase(path.extension().string());

        if (file_ext == ".exr")
        {
            Imf::RgbaInputFile f(path.string().c_str());
            Imath::Box2i dw = f.dataWindow();
            width = dw.max.x - dw.min.x + 1;
            height = dw.max.y - dw.min.y + 1;

            std::vector<Imf::Rgba> pixels(width * height);

            f.setFrameBuffer(pixels.data(), 1, width);
            f.readPixels(dw.min.y, dw.max.y);

            // copy pixels by pixels while flipping vertically and converting
            // from half float to float.
            pixels_rgbaf32.resize(width * height * 4);
            for (int32_t y = 0; y < height; y++)
            {
                for (int32_t x = 0; x < width; x++)
                {
                    // flip vertically
                    int32_t src_pixel_idx = x + (height - y - 1) * width;
                    int32_t dst_red_idx = (x + (y * width)) * 4;

                    pixels_rgbaf32[dst_red_idx + 0] =
                        (float)pixels[src_pixel_idx].r;
                    pixels_rgbaf32[dst_red_idx + 1] =
                        (float)pixels[src_pixel_idx].g;
                    pixels_rgbaf32[dst_red_idx + 2] =
                        (float)pixels[src_pixel_idx].b;
                    pixels_rgbaf32[dst_red_idx + 3] =
                        (float)pixels[src_pixel_idx].a;
                }
            }
        }
        else if (file_ext == ".png"
            || file_ext == ".jpg"
            || file_ext == ".jpeg")
        {
            // this will perform the necessary conversions from sRGB 2.2 to
            // Linear BT.709 I-D65 if needed.
            float* pixels = stbi_loadf_throw(
                path.string().c_str(),
                &width,
                &height,
                nullptr,
                4 // RGBA
            );

            // copy row by row while flipping vertically
            pixels_rgbaf32.resize(width * height * 4);
            for (int32_t y = 0; y < height; y++)
            {
                int32_t src_red_idx = ((height - y - 1) * width) * 4;
                int32_t dst_red_idx = (y * width) * 4;

                std::copy(
                    pixels + src_red_idx,
                    pixels + src_red_idx + (width * 4),
                    pixels_rgbaf32.data() + dst_red_idx
                );
            }

            stbi_image_free(pixels);
        }
        else
        {
            throw std::invalid_argument(
                "unsupported file extension for loading images"
            );
        }

        create_texture(
            state,
            state.queue_main,
            width,
            height,
            VK_FORMAT_R32G32B32A32_SFLOAT,
            pixels_rgbaf32.data(),
            pixels_rgbaf32.size() * sizeof(pixels_rgbaf32[0]),
            true,
            img,
            img_mem,
            imgview
        );
    }

    void save_image(
        AppState& state,
        const bv::ImagePtr& img,
        const std::filesystem::path& path
    )
    {
        std::vector<float> pixels_rgbaf32 =
            read_back_image_rgbaf32(state, img, state.queue_main, true);

        uint32_t width = img->config().extent.width;
        uint32_t height = img->config().extent.height;

        // get the file extension
        std::string file_ext = lowercase(path.extension().string());

        std::vector<uint8_t> pixels_rgba8;
        if (file_ext == ".png"
            || file_ext == ".jpg"
            || file_ext == ".jpeg")
        {
            pixels_rgba8.resize(pixels_rgbaf32.size());
            for (size_t i = 0; i < pixels_rgba8.size(); i++)
            {
                float v = pixels_rgbaf32[i];

                // convert from Linear BT.709 I-D65 to sRGB 2.2
                v = std::pow(std::clamp(v, 0.f, 1.f), 1.f / 2.2f);

                pixels_rgba8[i] = (uint8_t)std::round(v * 255.f);
            }
        }

        if (file_ext == ".exr")
        {
            Imf::Header header(
                (int)width,
                (int)height,
                1.f,
                { 0.f, 0.f },
                1.f,
                Imf::LineOrder::INCREASING_Y,
                Imf::Compression::ZIP_COMPRESSION
            );
            header.channels().insert("R", Imf::Channel(Imf::PixelType::FLOAT));
            header.channels().insert("G", Imf::Channel(Imf::PixelType::FLOAT));
            header.channels().insert("B", Imf::Channel(Imf::PixelType::FLOAT));
            header.channels().insert("A", Imf::Channel(Imf::PixelType::FLOAT));

            Imf::FrameBuffer fb;
            fb.insert(
                "R",
                Imf::Slice(
                    Imf::FLOAT,
                    (char*)pixels_rgbaf32.data(),
                    4 * sizeof(float),
                    width * 4 * sizeof(float)
                )
            );
            fb.insert(
                "G",
                Imf::Slice(
                    Imf::FLOAT,
                    (char*)(pixels_rgbaf32.data() + 1),
                    4 * sizeof(float),
                    width * 4 * sizeof(float)
                )
            );
            fb.insert(
                "B",
                Imf::Slice(
                    Imf::FLOAT,
                    (char*)(pixels_rgbaf32.data() + 2),
                    4 * sizeof(float),
                    width * 4 * sizeof(float)
                )
            );
            fb.insert(
                "A",
                Imf::Slice(
                    Imf::FLOAT,
                    (char*)(pixels_rgbaf32.data() + 3),
                    4 * sizeof(float),
                    width * 4 * sizeof(float)
                )
            );

            Imf::OutputFile f(path.string().c_str(), header);
            f.setFrameBuffer(fb);
            f.writePixels(height);
        }
        else if (file_ext == ".png")
        {
            if (pixels_rgba8.empty())
            {
                throw std::runtime_error("pixels_rgba8 is empty");
            }

            if (!stbi_write_png(
                path.string().c_str(),
                (int)width,
                (int)height,
                4, // RGBA
                pixels_rgba8.data(),
                (int)(width * 4 * 1) // y stride, 4 channels, 1 byte per channel
            ))
            {
                throw std::runtime_error("failed to write PNG image");
            }
        }
        else if (file_ext == ".jpg" || file_ext == ".jpeg")
        {
            if (pixels_rgba8.empty())
            {
                throw std::runtime_error("pixels_rgba8 is empty");
            }

            if (!stbi_write_jpg(
                path.string().c_str(),
                (int)width,
                (int)height,
                4, // RGBA
                pixels_rgba8.data(),
                90 // quality
            ))
            {
                throw std::runtime_error("failed to write JPEG image");
            }
        }
        else
        {
            throw std::invalid_argument(
                "unsupported file extension for saving images"
            );
        }
    }

}
