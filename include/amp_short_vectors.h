#pragma once

#ifndef _AMP_SHORT_VECTORS_H
#define _AMP_SHORT_VECTORS_H

namespace Concurrency
{
namespace graphics
{

#define __CPU_GPU__ restrict(cpu, amp)

#include "kalmar_short_vectors.inl"

#undef __CPU_GPU__

} // namespace graphics
} // namespace Concurrency

#endif // _AMP_SHORT_VECTORS_H
