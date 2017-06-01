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


void memcpyIntKernel(hc::accelerator_view &av, const int * src, int *dst, size_t numElements)
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

void test(int numElements)
{
    hc::accelerator acc;
    hc::accelerator_view av = acc.create_view();

    size_t sizeElements = numElements * sizeof(int);

    printf ("info: buffer size = %6.2f MB\n", sizeElements / 1024.0 / 1024.0);

    int * B  = hc::am_alloc(sizeElements, acc, 0);
    int * C  = hc::am_alloc(sizeElements, acc, 0);
    int * Bh = hc::am_alloc(sizeElements, acc, amHostPinned);
    int * Ch = hc::am_alloc(sizeElements, acc, amHostPinned);

    const int expected = 42;
    memsetIntKernel(av, Bh, expected, numElements);
    memsetIntKernel(av, Ch, -4, numElements);
    memsetIntKernel(av, C, -3, numElements);
    av.wait();

    // Set some default values that should be overwritten:
    memsetIntKernel(av, B, -2, numElements);

    // Bh->B copy:
    av.copy_async(Bh, B, sizeElements); 

    // If no system acquire, may pick up stale B values here:
    memcpyIntKernel(av, B, C, numElements);

    memcpyIntKernel(av, C, Ch, numElements);

    av.wait();
    check(Ch, numElements, expected);

    hc::am_free(B);
    hc::am_free(C);
    hc::am_free(Bh);
    hc::am_free(Ch);
}

int main()
{

    test(64); // tiny
    test(1024*1024); // small, fits in reasonable-sized L2
    test(1024*1024*256); // big, 256MB

    printf ("passed!\n");
    return 0;
};
