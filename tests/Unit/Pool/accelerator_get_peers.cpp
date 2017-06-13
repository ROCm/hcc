
// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>

/**
 * Test if hc::accelerator::get_peers() works fine.
 * Create a default accelerator and query its peers.
 * accelerator acc is not peer of itself.
 * @FIXME: in current system, all dGPUs is peer of each 
 * other, we should expect the size of peers is equal to 
 * number of GPU accelerators.
 */

int main()
{
    hc::accelerator acc;

    const auto& all = hc::accelerator::get_all();

    auto size_all = all.size();

    const auto& peers = acc.get_peers();

    if(peers.size() >= size_all)
        return -1;

    return 0;
}
