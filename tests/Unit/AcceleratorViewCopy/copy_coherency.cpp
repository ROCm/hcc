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
    hc::accelerator_view av0 = acc.create_view();
    hc::accelerator_view av1 = acc.create_view();


    int numElements = 20000000;
    size_t sizeElements = numElements * sizeof(int);

    printf ("info: buffer size = %6.2f MB\n", sizeElements / 1024.0 / 1024.0);

    //int * A  = hc::am_alloc(sizeElements, acc, 0);
    int * B  = hc::am_alloc(sizeElements, acc, 0);
    int * Bh = hc::am_alloc(sizeElements, acc, amHostPinned);


    if (1) {
        printf ("test: running same-stream copy coherency test\n");
        // Reset values:
        const int bexpected = 13;
        memset(Bh, 0xAF, numElements); // dummy values to ensure we can tell if copy occurs
        av0.wait();
        printf ("test:   setup complete\n");

        // Set B to -42, followed immediately by setting to expected value:
        memsetIntKernel(av0, B, -42, numElements);
        memsetIntKernel(av0, B, bexpected, numElements);

        // Async copy back to host with no intervening code - 
        // This should fail unless we properly issue a release fence after the memset:
        av0.copy_async(B, Bh, sizeElements); 

        // Wait on host:
        av0.wait();
        check(Bh, numElements, bexpected);
    }


    {
        printf ("test: running cross-stream copy coherency test\n");
        // Reset values:
        const int bexpected = 13;
        memset(Bh, 0xAF, numElements); // dummy values to ensure we can tell if copy occurs
        av0.wait();
        printf ("test:   setup complete\n");

        // Set B to -42, followed immediately by setting to expected value:
        memsetIntKernel(av0, B, -42, numElements);
        memsetIntKernel(av0, B, bexpected, numElements);

        auto m0 = av0.create_marker(hc::accelerator_scope);
        auto m1 = av1.create_blocking_marker(m0, hc::accelerator_scope);


        // Async copy back to host with no intervening code - 
        // This should fail unless we properly issue a release fence after the memset:
        av1.copy_async(B, Bh, sizeElements); 

        // Wait on host:  This needs to flush the caches appropriately:
        av1.wait();
        check(Bh, numElements, bexpected);
    }

    printf ("passed!\n");
    return 0;
};
