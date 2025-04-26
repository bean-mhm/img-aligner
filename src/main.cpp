#include "misc/common.hpp"

#include "app.hpp"

static void pause_on_error()
{
    std::cerr << "press [Enter] to exit" << std::endl;
    std::cin.get();
}

int main(int argc, char** argv)
{
    try
    {
        // set executable directory
        img_aligner::exec_dir(
            std::filesystem::absolute(argv[0]).parent_path()
        );

        img_aligner::App app(argc, argv);
        app.run();
    }
    IMG_ALIGNER_CATCH_ALL_IN_MAIN;
}
