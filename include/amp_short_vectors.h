#pragma once

#ifndef _AMP_SHORT_VECTORS_H
#define _AMP_SHORT_VECTORS_H

#include <cstddef>
#include <type_traits>

namespace Concurrency
{
namespace graphics
{

#define __CPU_GPU__ restrict(cpu, amp)

#if 1
#include <cstddef>
#include <type_traits>
#include "hc_short_vector.inl"
#else
#include "kalmar_short_vectors.inl"
#endif

#undef __CPU_GPU__

} // namespace graphics
} // namespace Concurrency

#endif // _AMP_SHORT_VECTORS_H
