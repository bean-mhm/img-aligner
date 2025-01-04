#include <iostream>
#include <filesystem>

#include "beva/beva.hpp"

#include "app.hpp"

int main(int argc, char** argv)
{
    // set working directory to the exectuable's parent directory
    std::filesystem::current_path(
        std::filesystem::weakly_canonical(argv[0]).parent_path()
    );

    try
    {
        img_aligner::App app;
        app.run();
    }
    catch (const bv::Error& e)
    {
        std::cerr << e.to_string() << '\n';
        std::cin.get();
        return 1;
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        std::cin.get();
        return 1;
    }
}
