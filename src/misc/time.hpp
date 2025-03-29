#pragma once

#include "common.hpp"
#include "numbers.hpp"

namespace img_aligner
{

    using TimePoint = std::chrono::high_resolution_clock::time_point;

    class ScopedTimer
    {
    public:
        ScopedTimer(
            bool should_print = true,
            std::string start_message = "processing",
            std::string end_message = " ({} s)\n"
        );
        ~ScopedTimer();

    private:
        TimePoint start_time;

        bool should_print;
        std::string end_message;

    };

    double elapsed_sec(const TimePoint& t);
    double elapsed_sec(const std::optional<TimePoint>& t);

}
