
// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>
/**
 * Test if hc::accelerator::get_compute_unit_count() works fine.
 * Create the default accelerator and check if the tested api returns
 * a non-zero value.
 */
 

int main()
{
    hc::accelerator acc;

    unsigned int cu_count = acc.get_cu_count();
   
    if(0 == cu_count)
    {
        printf("Compute Unit count is 0!\n");
        return -1;
    }

    return 0;
}
