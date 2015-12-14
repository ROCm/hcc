#include <map>

#include "hc_am.hpp"
#include "hsa.h"




//=========================================================================================================
// API Definitions.
//=========================================================================================================
//
//

namespace hc {

// Allocate accelerator memory, return NULL if memory could not be allocated:
auto_voidp am_alloc(size_t size, hc::accelerator acc, unsigned flags) 
{
    assert(flags == 0); // TODO - support other flags.

    void *ptr = NULL;

    if (size != 0 ) {
        if (acc.is_hsa_accelerator()) {
            hsa_agent_t *hsa_agent = static_cast<hsa_agent_t*> (acc.get_default_view().get_hsa_agent());
            hsa_region_t *am_region = static_cast<hsa_region_t*>(acc.get_hsa_am_region());

            //TODO - how does AMP return errors?


            hsa_status_t s1 = HSA_STATUS_SUCCESS;
            hsa_status_t s2 = HSA_STATUS_SUCCESS;

            s1 = hsa_memory_allocate(*am_region, size, &ptr);
            s2 = hsa_memory_assign_agent(ptr, *hsa_agent, HSA_ACCESS_PERMISSION_RW);



            if ((s1 != HSA_STATUS_SUCCESS) || (s2 != HSA_STATUS_SUCCESS)) {
                ptr = NULL;
            }

        } else if (acc.get_is_emulated()) {
            // TODO - handle host memory allocation here?
            assert(0);
        }
    }

    return ptr;
};


am_status_t am_free(void* ptr) 
{
    if (ptr != NULL) {
        hsa_memory_free(ptr);
    }
    return AM_SUCCESS;
}



am_status_t am_copy(void*  dst, const void*  src, size_t size)
{
    am_status_t am_status = AM_ERROR_MISC;
    hsa_status_t err = hsa_memory_copy(dst, src, size);

    if (err == HSA_STATUS_SUCCESS) {
        am_status = AM_SUCCESS;
    } else {
        am_status = AM_ERROR_MISC;
    }

    return am_status;
}

#if 0
// TODO:
// Should move physical location of destination, then copy to that new physical location.
// Need some additional runtime support before we can implement this.
am_status_t am_copy(void*  dst, const void*  src, size_t size, hc::accelerator dst_acc)
{
    am_status_t am_status = AM_ERROR_MISC;

    if (dst_acc.is_hsa_accelerator()) {
        hsa_agent_t *hsa_agent = static_cast <hsa_agent_t*> (dst_acc.get_default_view().get_hsa_agent());

        hsa_memory_assign_agent(dst,  *hsa_agent, HSA_ACCESS_PERMISSION_RW);
        hsa_status_t err = hsa_memory_copy(dst, src, size);

        if (err == HSA_STATUS_SUCCESS) {
            am_status = AM_SUCCESS;
        } else {
            am_status = AM_ERROR_MISC;
        }
    }
    return am_status;
}
#endif

} // end namespace hc.
