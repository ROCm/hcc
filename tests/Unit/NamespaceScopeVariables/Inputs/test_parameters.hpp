#pragma once

#include <cstddef>

static constexpr std::size_t array_size = 13;

extern int global_scalar;
extern int global_array[array_size];

namespace ns
{
    extern int namespace_scalar;
    extern int namespace_array[array_size];
}