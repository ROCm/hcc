// RUN: %hc %s -o %t.out -lhc_am -L/opt/rocm/lib -lhsa-runtime64 && %t.out
//
// Test coherency and flushes.  Need to flush GPU caches before H2D copy

#include <hc.hpp>
#include <hc_am.hpp>


void memsetIntKernel(hc::accelerator_view &av, int * ptr, int val, size_t numElements)
{
    hc::parallel_for_each(av, hc::extent<1>(numElements), [=] (hc::index<1> idx) [[hc]] 
    {
        ptr[idx[0]] = val;
    } );
};


void memcpyIntKernel(hc::accelerator_view &av, const int * src, int *dst, int val, size_t numElements)
{
    hc::parallel_for_each(av, hc::extent<1>(numElements), [=] (hc::index<1> idx) [[hc]] 
    {
        dst[idx[0]] = src[idx[0]];
    } );
};


void check(const int *ptr, int numElements, int expected) {
    for (int i=numElements-1; i>=0; i--) {
        if (ptr[i] != expected) {
            printf ("i=%d, ptr[](%d) != expected (%d)\n", i, ptr[i], expected);
            assert (ptr[i] == expected);
        }
    }
}

int main()
{
    hc::accelerator acc;
    hc::accelerator_view av = acc.create_view();


    int numElements = 20000000;
    size_t sizeElements = numElements * sizeof(int);

    printf ("info: buffer size = %6.2f MB\n", sizeElements / 1024.0 / 1024.0);

    //int * A  = hc::am_alloc(sizeElements, acc, 0);
    int * B  = hc::am_alloc(sizeElements, acc, 0);
    int * Bh = hc::am_alloc(sizeElements, acc, amHostPinned);

    const int binit = 13;
    //memsetIntKernel(av, A, -1, numElements);
    memsetIntKernel(av, Bh, -3, numElements);
    av.wait();

    memsetIntKernel(av, B, -42, numElements);
    memsetIntKernel(av, B, binit, numElements);

    // check kernel followed by D2H copy:
    av.copy_async(B, Bh, sizeElements); 

    av.wait();
    check(Bh, numElements, binit);

    printf ("passed!\n");
    return 0;
};
