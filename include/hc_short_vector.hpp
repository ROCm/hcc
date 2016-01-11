#pragma once

#ifndef _HC_SHORT_VECTORS_HPP
#define _HC_SHORT_VECTORS_HPP

namespace hc
{

#ifdef __HCC__
#define __CPU_GPU__ [[cpu]] [[hc]]
#else
#define __CPU_GPU__
#endif

#include "kalmar_short_vectors.inl"

#undef __CPU_GPU__

} // namespace hc

#endif // _HC_SHORT_VECTORS_H
