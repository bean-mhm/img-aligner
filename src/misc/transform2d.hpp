#pragma once

#include "common.hpp"

namespace img_aligner
{

    struct Transform2d
    {
        glm::vec2 scale{ 1.f };
        float rotation = 0.f; // degrees
        glm::vec2 offset{ 0.f };

        bool is_identity() const;
        glm::vec2 apply(const glm::vec2& p) const;
    };

}
