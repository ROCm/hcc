#pragma once

#ifndef _AMP_SHORT_VECTORS_H
#define _AMP_SHORT_VECTORS_H

#warning "Please note that C++AMP support is deprecated, and will be removed in ROCm 2.0. Please switch to using HC."

#include <cstddef>
#include <type_traits>
#include "kalmar_serialize.h"

namespace Concurrency
{
namespace graphics
{

#define __CPU_GPU__ restrict(cpu, amp)

#if 1
#include "hc_short_vector.inl"
#else
#include "kalmar_short_vectors.inl"
#endif

#undef __CPU_GPU__

} // namespace graphics
} // namespace Concurrency

#endif // _AMP_SHORT_VECTORS_H
