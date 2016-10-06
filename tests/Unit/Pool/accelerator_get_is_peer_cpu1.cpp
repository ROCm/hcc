// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>

/**
 * So far, CPU accelerator is not peer of any other
 * accelerator. This test will pass each accelerator 
 * in the system to CPU accelerator and check if 
 * get_is_peer() will return false.
 */

int main()
{
    // Get CPU accelerator.
    hc::accelerator cpu;

    const auto& all = hc::accelerator::get_all();

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
    for(auto iter = all.begin(); iter != all.end(); iter++)
    {
        if(cpu.get_is_peer(*iter))
        {
            return -1;
        }
    }

    return 0;
}
