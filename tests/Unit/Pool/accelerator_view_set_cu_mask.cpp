
// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>
#include <vector>

/**
 * Test if hc::accelerator::set_cu_mask(const vector<bool>& cu_mask) works fine.
 * This test will set the CU mask of the queue, and launch a kernel, we expect
 * the kernel will be finished successfully.
 */


int main()
{
    hc::accelerator acc;
    hc::accelerator_view acc_view = acc.get_default_view();

    unsigned int cu_count = acc.get_cu_count();

    if(0 == cu_count)
        return -1;

    std::vector<bool> cu_mask(cu_count, false);

    // Set the first half bit to true.
    for(int i = 0; i < (cu_count / 2 + 1); i++)
    {
        cu_mask[i] = true;
    }

    if(!acc_view.set_cu_mask(cu_mask)) {
        printf ("set_cu_mask returned false (not successful)\n");
        return -1;
    }

    // Launch a kernel.
    const int vec_size = 2048;

    hc::array_view<int, 1> table_a(vec_size);
    hc::array_view<int, 1> table_b(vec_size);
    hc::array_view<int, 1> table_c(vec_size);

    // Initialize input
    for(int i = 0; i < vec_size; i++)
    {
        table_a[i] = 1;
        table_b[i] = 2;
    }

    hc::extent<1> e(vec_size);
    hc::completion_future fut = hc::parallel_for_each(acc_view, e,
                                [=](hc::index<1> idx) __HC__ {
                                  table_c[idx[0]] = table_a[idx[0]] + table_b[idx[0]];
                                });

    fut.wait();

    // Verify output
    for(int i = 0; i < vec_size; i++)
    {
        if(table_c[i] != 3)
        {
            printf("Result verification fails\n");
            return -1;
        }
    }

    return 0;
}
