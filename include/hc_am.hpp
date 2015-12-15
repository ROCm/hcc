#pragma once

#include <hc.hpp>

typedef int am_status_t;
#define AM_SUCCESS                           0
// TODO - provide better mapping of HSA error conditions to HC error codes.
#define AM_ERROR_MISC                       -1 /** Misellaneous error */


namespace hc {

/**
 * Allocates a block of @p size bytes of memory on the specified @p acc.  
 *
 * The contents of the newly allocated block of memory are not initialized.
 *
 * If @p size == 0, 0 is returned.
 *
 * Flags must be 0.
 *
 * @returns : On success, pointer to the newly allocated memory is returned.  
 * The pointer is typecast to the desired return type.
 *
 * If an error occurred trying to allocate the requested memory, 0 is returned.
 *
 * @see am_free, am_copy
 */
auto_voidp am_alloc(size_t size, hc::accelerator acc, unsigned flags);

/**
 * Frees a block of memory previously allocated with am_alloc.
 *
 * @see am_alloc, am_copy
 */
am_status_t am_free(void*  ptr);


/**
 * Copies @p size bytes of memory from @p src to @ dst.  The memory areas (src+size and dst+size) must not overlap.  
 *
 * @returns AM_SUCCESS on error or AM_ERROR_MISC if an error occurs.
 * @see am_alloc, am_free
 */
am_status_t am_copy(void*  dst, const void*  src, size_t size);

//am_status_t am_copy(void*  dst, const void*  src, size_t size, hc::accelerator dst_acc);


}; // namespace hc


