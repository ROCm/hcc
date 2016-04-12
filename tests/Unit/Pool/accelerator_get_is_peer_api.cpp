// XFAIL: Linux
// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>

/**
 * Test if hc::accelerator::get_is_peer() works fine.
 * Create the default accelerator and check if others 
 * is peer of it.
 * accelerator is peer of itself.
 * FIXME: on current system, dGPU is peer of any each  
 * other, we should expect is_get_peer() return true
 * always.
 */

int main()
{
    hc::accelerator acc;

    const auto& all = hc::accelerator::get_all();

    if(0 == all.size())
        return -1;

    for(auto iter = all.begin(); iter != all.end(); iter++)
    {
        if(!acc.get_is_peer(*iter))
            return -1;
    }

    return 0;
}
