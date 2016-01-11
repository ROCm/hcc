#pragma once

#ifndef _HC_SHORT_VECTORS_HPP
#define _HC_SHORT_VECTORS_HPP

namespace hc
{

#define __CPU_GPU__ [[cpu]] [[hc]]

#include "kalmar_short_vectors.inl"

#undef __CPU_GPU__

} // namespace hc

#endif // _HC_SHORT_VECTORS_H
