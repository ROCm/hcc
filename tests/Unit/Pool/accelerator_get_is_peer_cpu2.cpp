// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>

/**
 * So far, CPU accelerator is not peer of any other
 * accelerator. This test will pass CPU accelerator
 * to default accelerator and check if 
 * get_is_peer() will return false.
 */

int main()
{
    // Get Default accelerator.
    hc::accelerator acc;

    const auto& all = hc::accelerator::get_all();

    hc::accelerator cpu;

    for(auto iter = all.begin(); iter != all.end(); iter++)
    {
        if(iter->get_is_emulated())
        {
            cpu = *iter;
            break;
        }
    }

    // Check get_is_peer() return value, if it is true,
    // then, test fails, return -1.
    if(acc.get_is_peer(cpu))
        return -1;

    return 0;
}
