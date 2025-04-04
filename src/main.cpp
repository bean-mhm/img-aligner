#include "misc/common.hpp"

#include "app.hpp"

static void pause_on_error()
{
    std::cerr << "press [Enter] to exit\n";
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
