
// RUN: %hc %s -lhc_am -o %t.out && %t.out

#include <hc_am.hpp>
#include <hc.hpp>

int main()
{
    hc::accelerator acc;

    void* host_ptr = am_alloc(1, acc, amHostPinned);

    // allocation fails if return NULL.
    if(host_ptr == NULL)
        return -1;

    const auto& peers = acc.get_peers();

    // map device pointer to all peers.
    if(peers.size()!=0 && AM_SUCCESS != am_map_to_peers(host_ptr, peers.size(), peers.data()))
        return -1;

    return 0;
}
