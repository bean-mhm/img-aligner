#pragma once

#include "common.hpp"

namespace img_aligner
{

    class App
    {
    public:
        App() = default;
        void run();

    private:
        AppState state;

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
        void create_imgui_descriptor_pool();
        void init_imgui_vk_window_data();
        void init_imgui();

        void render_frame(ImDrawData* draw_data);
        void present_frame();

        friend void glfw_framebuf_resize_callback(
            GLFWwindow* window,
            int width,
            int height
        );

    };

}
