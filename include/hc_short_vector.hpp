#pragma once

#ifndef _HC_SHORT_VECTORS_HPP
#define _HC_SHORT_VECTORS_HPP

#include <cstddef>
#include <type_traits>
#include "kalmar_serialize.h"
#include "hc_defines.h"

namespace hc
{

namespace short_vector
{

#ifdef __HCC__
#define __CPU_GPU__ [[cpu]] [[hc]]
#else
#define __CPU_GPU__
#endif

#if 1
#include "hc_short_vector.inl"
#else
#include "kalmar_short_vectors.inl"
#endif

#undef __CPU_GPU__

} // namespace short_vector

} // namespace hc

#endif // _HC_SHORT_VECTORS_H
