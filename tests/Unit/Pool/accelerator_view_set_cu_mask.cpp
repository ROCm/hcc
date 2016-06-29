// XFAIL: Linux
// RUN: %hc %s -I%hsa_header_path -L%hsa_library_path -lhsa-runtime64 -o %t.out && %t.out

#include <hc.hpp>
#include <hsa.h>
#include <hsa_ext_amd.h>
#include <vector>

/**
 * Test if hc::accelerator::set_cu_mask(const vector<bool> cu_mask) works fine.
 * This test will set the CU mask of the queue, and launch a kernel, we expect 
 * the kernel will be finished successfully.
 */
 

int main()
{
    hc::accelerator acc;
    hc::accelerator_view acc_view = acc.get_default_view();

    unsigned int cu_count = acc.get_compute_unit_count();
   
    if(0 == cu_count)
        return -1;

    std::vector<bool> cu_mask(cu_count, false);

    // Set the first half bit to true.
    for(int i = 0; i < (cu_count / 2 + 1); i++)
    {
        cu_mask[i] = true;;
    }

    if(!acc_view.set_cu_mask(cu_mask))
        return -1;

    // Launch a kernel.
    const int vec_size = 2048;
    int* ptr_a = (int*)malloc(vec_size);
    int* ptr_b = (int*)malloc(vec_size);
    int* ptr_c = (int*)malloc(vec_size);

    // Initialize input
    for(int i = 0; i < vec_size; i++)
    {
        ptr_a[i] = 1;
        ptr_b[i] = 2;
    }

    hcc::extent<1> e(vec_size);
    hc::completion_future fut = hc::parallel_for_each(acc_view, e, 
                                [=](hc::index<1> idx) restrict(amp) {
                                  ptr_c[idx[0]] = ptr_a[idx[0]] + ptr_b[idx[0]];
                                }
    fut.wait();

    // Verify output
    for(int i = 0; i < vec_size; i++)
    {
        if(ptr_c[i] != 3)
        {
            printf("Result verification fails\n");
            return -1;
        }
    }

    return 0;
}
