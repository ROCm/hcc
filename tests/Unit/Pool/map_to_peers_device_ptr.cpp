
// RUN: %hc %s -lhc_am -o %t.out && %t.out

#include <hc_am.hpp>
#include <hc.hpp>

int main()
{
    hc::accelerator acc;

    void* dev_ptr = am_alloc(1, acc, 0);

    // allocation fails if return NULL.
    if(dev_ptr == NULL)
        return -1;

    const auto& all = acc.get_peers();

    // map device pointer to all peers.
    if(all.size()!=0 && AM_SUCCESS != am_map_to_peers(dev_ptr, all.size(), all.data()))
        return -1;

    return 0;
}
