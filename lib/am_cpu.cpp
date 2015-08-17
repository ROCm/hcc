/*
 * Minimal AM implementation which provides host-CPU memory management functions - really just thin wrapper over malloc/free/copy.
 * Will ony work on HSA Devices with Kalmar tests.
 */
#include <stdlib.h>
#include <string.h>
#include "am.h"

namespace hc {

auto_voidp am_alloc(size_t size, unsigned flags, am_status_t *am_status) 
{
    void *p = malloc(size);

    if (*am_status) {
        if (p == NULL) {
            *am_status = AM_ERROR_MEMORY_ALLOCATION;
        } else {
            *am_status = AM_SUCCESS;
        }
    }

    return p;
};


am_status_t am_free(void* ptr) 
{
    if (ptr != NULL) {
        free(ptr);
    }
    return AM_SUCCESS;
}


am_status_t am_update(void* ptr, size_t size, am_accelerator_view_t av, unsigned flags)
{
    // On CPU, memory is coherent so this is a NOP.
    return AM_SUCCESS;
}



am_status_t am_copy(void* dst, const void* src, size_t size, am_accelerator_view_t dst_av)
{
    memcpy(dst, src, size);
    return AM_SUCCESS;
}

} // namespace hc
