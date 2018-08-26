//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#warning "C++AMP support is deprecated in ROCm 1.9 and will be removed in ROCm 2.0!"

#include "kalmar_math.h"

namespace Concurrency {

// namespace alias

// namespace Concurrency::fast_math is an alias of namespace detail::fast_math
namespace fast_math = detail::fast_math;

// namespace Concurrency::precise_math is an alias of namespace detail::precise_math
namespace precise_math = detail::precise_math;

} // namespace Concurrency

