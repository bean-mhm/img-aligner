#include "time.hpp"

namespace img_aligner
{

    ScopedTimer::ScopedTimer(
        bool should_print,
        std::string start_message,
        std::string end_message
    ) : start_time(std::chrono::high_resolution_clock::now()),
        should_print(should_print),
        end_message(std::move(end_message))
    {
        if (should_print && !start_message.empty())
        {
            std::cout << start_message << std::flush;
        }
    }

    ScopedTimer::~ScopedTimer()
    {
        if (!should_print || end_message.empty())
        {
            return;
        }

	std::string elapsed_str = to_str(elapsed_sec(start_time));

        std::cout << fmt::vformat(
            end_message,
            fmt::make_format_args(elapsed_str)
        ) << std::flush;
    }

    double elapsed_sec(const TimePoint& t)
    {
        const auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(now - t).count();
    }

    double elapsed_sec(
        const std::optional<TimePoint>& t
    )
    {
        if (!t.has_value())
            return 0.;

        const auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(now - *t).count();
    }

}
