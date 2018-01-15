//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <memory>
#include <stdlib.h>

/** \cond HIDDEN_SYMBOLS */
namespace Kalmar {

constexpr inline bool kalmar_is_alignment(std::size_t value) noexcept {
    return (value > 0) && ((value & (value - 1)) == 0);
}

inline void* kalmar_aligned_alloc(std::size_t alignment, std::size_t size) noexcept {
    assert(kalmar_is_alignment(alignment));
    enum {
        N = std::alignment_of<void*>::value
    };
    if (alignment < N) {
        alignment = N;
    }
    void* memptr = NULL;
    // posix_memalign shall return 0 upon successfully allocate aligned memory
    posix_memalign(&memptr, alignment, size);
    assert(memptr);

    return memptr;
}

inline void kalmar_aligned_free(void* ptr) noexcept {
    if (ptr) {
        free(ptr);
    }
}

} // namespace Kalmar
/** \endcond */
