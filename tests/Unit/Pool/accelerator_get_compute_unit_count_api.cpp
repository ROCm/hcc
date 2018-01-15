
// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>
/**
 * Test if hc::accelerator::get_compute_unit_count() works fine.
 * Create the default accelerator and check if the tested api returns
 * a non-zero value.
 */
 

int main()
{
    std::vector <hc::accelerator> accs = hc::accelerator::get_all();
    
    for (auto acc = accs.begin(); acc != accs.end(); acc++)
    { 
        unsigned int cu_count = acc->get_cu_count();
        int accSeqNum = acc->get_seqnum();

        printf ("acc=%d count=%d\n", accSeqNum, cu_count);

      
            if (!acc->get_is_emulated()) { 
                if(0 == cu_count)
                {
                    printf("Compute Unit count is 0!\n");
                    return -1;
                }

                if(-1 == accSeqNum) {
                    printf("accSeqnum Unit count is -1!\n");
                    return -1;
                }
            }
    }

    return 0;
}
