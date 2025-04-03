#pragma once

#include <cstdint>

namespace img_aligner
{

    // hash function collection
    // sources: https://nullprogram.com/blog/2018/07/31/
    //          https://www.shadertoy.com/view/WttXWX

    constexpr uint32_t triple32(uint32_t x)
    {
        x ^= x >> 17;
        x *= 0xed5ad4bbU;
        x ^= x >> 11;
        x *= 0xac4c1b51U;
        x ^= x >> 15;
        x *= 0x31848babU;
        x ^= x >> 14;
        return x;
    }

    constexpr uint32_t triple32(int32_t x)
    {
        return triple32((uint32_t)x);
    }

    constexpr uint32_t triple32(float x)
    {
        return triple32(*reinterpret_cast<uint32_t*>(&x));
    }

    // any -> u32

    template<typename X>
    constexpr uint32_t hash_u32(X x)
    {
        return triple32(x);
    }

    template<typename X, typename Y>
    constexpr uint32_t hash_u32(X x, Y y)
    {
        return triple32(x + triple32(y));
    }

    template<typename X, typename Y, typename Z>
    constexpr uint32_t hash_u32(X x, Y y, Z z)
    {
        return triple32(x + triple32(y + triple32(z)));
    }

    template<typename X, typename Y, typename Z, typename W>
    constexpr uint32_t hash_u32(X x, Y y, Z z, W w)
    {
        return triple32(x + triple32(y + triple32(z + triple32(w))));
    }

    // any -> i32

    template<typename X>
    constexpr int32_t hash_i32(X x)
    {
        return (int32_t)hash_u32<X>(x);
    }

    template<typename X, typename Y>
    constexpr int32_t hash_i32(X x, Y y)
    {
        return (int32_t)hash_u32<X, Y>(x, y);
    }

    template<typename X, typename Y, typename Z>
    constexpr int32_t hash_i32(X x, Y y, Z z)
    {
        return (int32_t)hash_u32<X, Y, Z>(x, y, z);
    }

    template<typename X, typename Y, typename Z, typename W>
    constexpr int32_t hash_i32(X x, Y y, Z z, W w)
    {
        return (int32_t)hash_u32<X, Y, Z, W>(x, y, z, w);
    }

    // any -> f32

    template<typename X>
    constexpr float hash_f32(X x)
    {
        return (float)hash_u32<X>(x) / 4294967295.f;
    }

    template<typename X, typename Y>
    constexpr float hash_f32(X x, Y y)
    {
        return (float)hash_u32<X, Y>(x, y) / 4294967295.f;
    }

    template<typename X, typename Y, typename Z>
    constexpr float hash_f32(X x, Y y, Z z)
    {
        return (float)hash_u32<X, Y, Z>(x, y, z) / 4294967295.f;
    }

    template<typename X, typename Y, typename Z, typename W>
    constexpr float hash_f32(X x, Y y, Z z, W w)
    {
        return (float)hash_u32<X, Y, Z, W>(x, y, z, w) / 4294967295.f;
    }

}
