// XFAIL: Linux
// RUN: %hc %s -I%hsa_header_path -L%hsa_library_path -lhsa-runtime64 -o %t.out && %t.out

#include <hc.hpp>
#include <hsa.h>
#include <hsa_ext_amd.h>

/**
 * Test if hc::accelerator::get_compute_unit_count() works fine.
 * Create the default accelerator and check if the tested api returns
 * a non-zero value.
 */
 

int main()
{
    hc::accelerator acc;

    unsigned int cu_count = acc.get_compute_unit_count();
   
    if(0 == cu_count)
    {
        printf("Compute Unit count is 0!\n");
        return -1;
    }

    return 0;
}
