#include "transform2d.hpp"

namespace img_aligner
{

    bool Transform2d::is_identity() const
    {
        return scale == glm::vec2(1)
            && rotation == 0.f
            && offset == glm::vec2(0);
    }

    glm::vec2 Transform2d::apply(const glm::vec2& p) const
    {
        float angle_rad = glm::radians(rotation);
        float c = std::cos(angle_rad);
        float s = std::sin(angle_rad);

        // scale, rotate, offset
        return glm::mat2(c, -s, s, c) * (p * scale) + offset;
    }

}
