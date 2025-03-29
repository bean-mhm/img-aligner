#include "numbers.hpp"

namespace img_aligner
{

    uint32_t upper_power_of_2(uint32_t n)
    {
        if (n != 0)
        {
            n -= 1;
        }

        uint32_t i = 0;
        while (n > 0)
        {
            n >>= 1;
            i++;
        }
        return (uint32_t)1 << i;
    }

    uint32_t round_log2(uint32_t n)
    {
        uint32_t i = 0;
        while (n > 0)
        {
            n >>= 1;
            i++;
        }
        return i;
    }

}
