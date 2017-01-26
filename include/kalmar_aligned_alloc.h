//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <memory>

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
    std::size_t n = size + alignment - N;
    void* p1 = 0;
    void* p2 = std::malloc(n + sizeof p1);
    if (p2) {
        p1 = static_cast<char*>(p2) + sizeof(p1);
        posix_memalign(&p1,alignment,size);
        *(static_cast<void**>(p1) - 1) = p2;
    }
    return p1;
}

inline void kalmar_aligned_free(void* ptr) noexcept {
    if (ptr) {
        void* p = *(static_cast<void**>(ptr) - 1);
        std::free(p);
    }
}

} // namespace Kalmar
/** \endcond */
