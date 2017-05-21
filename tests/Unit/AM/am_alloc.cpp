
// RUN: %hc %s -lhc_am -o %t.out && %t.out

#include <cstdlib>
#include <cstdio>
#include <hc.hpp>
#include <hc_am.hpp>
#include <iostream>

void accessFromAllAccs(int numElements, int *ptr) 
{
    auto accs = hc::accelerator::get_all();
    hc::extent<1> ext(numElements);;
    for (auto a=accs.begin(); a != accs.end(); a++) {

        if (!a->get_is_emulated()) {
            int devId = a->get_seqnum();

            std::cout << "test: running PFE on accelerator#" << devId << "\n";

            hc::parallel_for_each(a->get_default_view(), ext, [=] (hc::index<1> idx) [[hc]] 
            {
               ptr[idx[0]] = devId;
            }).wait();


            for (int i=0; i<numElements; i++) {
                if (ptr[i] != devId) {
                    printf ("ptr[%d](%d) != devId(%d)\n", i, ptr[i], devId);
                    assert (ptr[i] == devId);
                }
            }
        }
    }
}


// Simple tests to verify regions and pointers all work:
int main() 
{

    hc::accelerator defaultAcc;

    {
        const size_t cSize = 30000;


        char *a = am_alloc(10000, defaultAcc, 0);
        char *b = am_alloc(20000, defaultAcc, amHostPinned);
        char *c = am_alloc(cSize, defaultAcc, amHostCoherent);

        // Simple tests to verify that the memory allocations to all 3 regions succeeded.
        assert(a);
        assert(b);
        assert(c);

        assert (hc::am_free(a) == AM_SUCCESS);
        assert (hc::am_free(b) == AM_SUCCESS);
        assert (hc::am_free(c) == AM_SUCCESS);

        //  Try duplicate free, should return error:
        assert (hc::am_free(c) != AM_SUCCESS);
    }

    {
        int numElements = 1000;
        size_t sizeElements = 1000*sizeof(int);

        auto accs = hc::accelerator::get_all();
        for (auto a=accs.begin(); a != accs.end(); a++) {
            if (!a->get_is_emulated()) {

                int *hostPtr = nullptr;
                hostPtr = am_alloc(sizeElements, *a, amHostCoherent);
                std::cout << "test: alloc coherent host mem on accelerator#" << a->get_seqnum() << "\n";
                assert(hostPtr);

                accessFromAllAccs(numElements, hostPtr);

                assert (hc::am_free(hostPtr) == AM_SUCCESS);

                hostPtr = am_alloc(sizeElements, *a, amHostPinned);
                std::cout << "test: alloc non-coherent host mem on accelerator#" << a->get_seqnum() << "\n";
                assert(hostPtr);

                accessFromAllAccs(numElements, hostPtr);

                assert (hc::am_free(hostPtr) == AM_SUCCESS);
            }
        }


    }

    return 0; // passed!
}
