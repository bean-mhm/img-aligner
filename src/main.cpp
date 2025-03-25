#include <iostream>
#include <filesystem>

#include "beva/beva.hpp"

#include "app.hpp"
#include "common.hpp"

static void pause_on_error()
{
    std::cerr << "press [Enter] to exit\n";
    std::cin.get();
}

int main(int argc, char** argv)
{
    img_aligner::exec_dir(
        std::filesystem::weakly_canonical(argv[0]).parent_path()
    );

    try
    {
        img_aligner::App app(argc, argv);
        app.run();
    }
    catch (const CLI::Error& e)
    {
        if (e.get_exit_code() != (int)CLI::ExitCodes::Success)
        {
            std::cerr << "CLI: " << e.what() << '\n';
            pause_on_error();
        }
        return e.get_exit_code();
    }
    catch (const bv::Error& e)
    {
        std::cerr << "beva: " << e.to_string() << '\n';
        pause_on_error();

        return EXIT_FAILURE;
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        pause_on_error();

        return EXIT_FAILURE;
    }
}
