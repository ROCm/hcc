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


void memsetIntKernelReverse(hc::accelerator_view &av, int * ptr, int val, size_t numElements)
{
    hc::parallel_for_each(av, hc::extent<1>(numElements), [=] (hc::index<1> idx) [[hc]] 
    {
        ptr[numElements-idx[0]] = val;
    } );
};



hc::completion_future 
memcpyIntKernel(hc::accelerator_view &av, const int * src, int *dst, size_t numElements)
{
    return hc::parallel_for_each(av, hc::extent<1>(numElements), [=] (hc::index<1> idx) [[hc]] 
    {
        dst[idx[0]] = src[idx[0]];
    } );
};


void checkReverse(const int *ptr, int numElements, int expected) {
    for (int i=numElements-1; i>=0; i--) {
        if (ptr[i] != expected) {
            printf ("i=%d, ptr[](%d) != expected (%d)\n", i, ptr[i], expected);
            assert (ptr[i] == expected);
        }
    }

    printf ("test:   passed\n");
}


void checkForward(const int *ptr, int numElements, int expected) {
    for (int i=0; i<numElements; i++) {
        if (ptr[i] != expected) {
            printf ("i=%d, ptr[](%d) != expected (%d)\n", i, ptr[i], expected);
            assert (ptr[i] == expected);
        }
    }

    printf ("test:   passed\n");
}

void singleAccelerator(int numElements)
{
    const size_t sizeElements = numElements * sizeof(int);

    hc::accelerator acc;
    hc::accelerator_view av0 = acc.create_view();
    hc::accelerator_view av1 = acc.create_view();

    //int * A  = hc::am_alloc(sizeElements, acc, 0);
    int * B  = hc::am_alloc(sizeElements, acc, 0);
    int * Bh = hc::am_alloc(sizeElements, acc, amHostPinned);

    if (1) {
        printf ("test: running same-stream copy coherency test\n");
        // Reset values:
        const int bexpected = 42;
        memset(Bh, 0xAF, sizeElements); // dummy values to ensure we can tell if copy occurs
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
        checkReverse(Bh, numElements, bexpected);
    }


    if (1) {
        printf ("test: running cross-stream copy coherency test\n");
        // Reset values:
        const int bexpected = 42;
        memset(Bh, 0xAF, sizeElements); // dummy values to ensure we can tell if copy occurs
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
        checkReverse(Bh, numElements, bexpected);
    }


    if (1) {
        printf ("test: running same-stream pinned host zero-copy read test\n");
        // Reset values:
        const int bexpected = 42;
        memset(Bh, 0xAF, sizeElements); // dummy values to ensure we can tell if copy occurs
        av0.wait();
        printf ("test:   setup complete\n");

        // Set Bh to 42
        memsetIntKernel(av0, Bh, bexpected, numElements);

        av0.wait();
        checkReverse(Bh, numElements, bexpected);
    }


    // clean up:
    assert (hc::am_free(B) == AM_SUCCESS);
    assert (hc::am_free(Bh) == AM_SUCCESS);
}


void resetMultiAccelerator(hc::accelerator_view av0, hc::accelerator_view av1, 
                           int *dataGpu0, int *dataGpu1, int *dataHost,
                           int numElements, int expected)
{
    const size_t sizeElements = numElements * sizeof(int);
    // Reset values on GPU0:
    memsetIntKernel(av0, dataGpu0 , expected, numElements); 
    memsetIntKernel(av1, dataGpu1, -1, numElements); 
    memset(dataHost, -2, sizeElements); // dummy values to ensure we can tell if copy occurs
    av0.wait();
    av1.wait();
    printf ("  test: init complete\n");
}

void multiAccelerator(int numElements)
{
    printf ("\ntest: running cross-accelerator copy tests\n");

    auto accs = hc::accelerator::get_all();
    std::vector <hc::accelerator> gpus;
    for (auto a : accs) {
        if (!a.get_is_emulated()) {
            gpus.push_back(a);
        }
    }

    if (gpus.size() < 2) {
        printf ("warning: found only %zu accelerators,skipping multi-accelerator tests\n", gpus.size());
        return;
    }

    if (!gpus[0].get_is_peer(gpus[1])  || !gpus[1].get_is_peer(gpus[0])) {
        printf ("warning: gpu0/1 are not peers\n");
        return;
    }

    const size_t sizeElements = numElements * sizeof(int);
    int * dataGpu0 = hc::am_alloc(sizeElements, gpus[0], 0);
    int * dataGpu1 = hc::am_alloc(sizeElements, gpus[1], 0);
    int * dataHost = hc::am_alloc(sizeElements, gpus[1], amHostPinned);

    hc::accelerator_view av0 = gpus[0].create_view();
    hc::accelerator_view av1 = gpus[1].create_view();

    // Make dataGpu0  accessible to GPU0 and 1
    assert (am_map_to_peers(dataGpu0 , 2, gpus.data()) == AM_SUCCESS);
    assert (am_map_to_peers(dataGpu1 , 2, gpus.data()) == AM_SUCCESS);

    hc::am_memtracker_print(0);
    const int expected = 42;

    if (1) {
        printf ("test: running cross-accelerator test.  av0.copy->gpu1, av1.copy->host\n");

        resetMultiAccelerator(av0, av1, dataGpu0, dataGpu1, dataHost, numElements, expected);

        // Copy to GPU 1 with SDMA copy:
        auto copy0 = av0.copy_async(dataGpu0  /*src*/, dataGpu1 /*dst*/, sizeElements);
        //auto m0    = av0.create_marker(hc::system_scope); 

        // av1 wait for copy to finish.  
        // This is cross-stream dependency and should add system-scope acquire.
        av1.create_blocking_marker(copy0); 

        // This should only execute after the copy has finished.
        av1.copy(dataGpu1 /*src*/, dataHost /*dst*/, sizeElements);

        checkForward(dataHost, numElements, expected); // TODO - change to reverse.
    }


    if (1) {
        printf ("test: running cross-accelerator test.  av0.kernel->gpu1, av1.copy->host\n");
        bool firstUseAv0 = 1;
        bool firstCopy   = 0;

        resetMultiAccelerator(av0, av1, dataGpu0, dataGpu1, dataHost, numElements, expected);

        // Copy to GPU 1 with kernel
        hc::completion_future m0;

        if (firstUseAv0) {
            if (firstCopy) {
                m0 = av0.copy_async(dataGpu0 /*src*/, dataGpu1 /*dst*/, sizeElements);
            } else {
                m0 = memcpyIntKernel(av0, dataGpu0, dataGpu1, numElements);
            }
        } else {
            if (firstCopy) {
                m0 = av1.copy_async(dataGpu0 /*src*/, dataGpu1 /*dst*/, sizeElements);
            } else {
                m0 = memcpyIntKernel(av1, dataGpu0, dataGpu1, numElements);
            }
        }

        // av1 wait for copy to finish.  
        // This is cross-stream dependency and should add system-scope acquire.
        av1.create_blocking_marker(m0); 

        // This should only execute after the copy has finished.
        av1.copy(dataGpu1 /*src*/, dataHost /*dst*/, sizeElements);

        checkReverse(dataHost, numElements, expected);
    }



    assert (hc::am_free(dataGpu0 ));
    assert (hc::am_free(dataGpu1));
    assert (hc::am_free(dataHost));
}

int main()
{
    int numElements = 20000000;

    const size_t sizeElements = numElements * sizeof(int);
    printf ("info: buffer size = %6.2f MB\n", sizeElements / 1024.0 / 1024.0);

    if (1) {
        singleAccelerator(numElements);
    }

    // TODO - need to re-enable multi-GPU tests:
    if (0) {
        multiAccelerator(numElements);
    }

    printf ("passed!\n");
    return 0;
};
