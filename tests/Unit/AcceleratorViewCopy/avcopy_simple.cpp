// RUN: %hc %s -o %t.out -lhc_am && %t.out 
#include <stdlib.h>
#include <iostream>

#include <hc.hpp>
#include <hc_am.hpp>


// A few helper routines for writing tests:
#include "common.h"



auto_voidp hostAlloc(hc::accelerator &acc, bool allocPinned, size_t size)
{
    void *ptr;
    if (allocPinned) {
        ptr = hc::am_alloc(size, acc, amHostPinned);
    } else {
        ptr = malloc(size);
    }

    assert(ptr);
    return ptr;
}
    

// ****************************************************************************
void memcopy(hc::accelerator_view &av, bool useAsyncCopy, const void * src, void *dst, size_t sizeBytes)
{
    if (useAsyncCopy) {
        av.copy_async(src, dst, sizeBytes);
    } else {
        av.copy(src, dst, sizeBytes);
    }
}



//---
// Test simple H2D copies, kernel, D2H copy.  Various knobs to control when synchronization occurs:
// acc : accelerator to run test on.  The default queue is used (get_default_view).
// N = size of arrays to use, in elements.
// useAsyncCopy = use accelerator_view::copy_async for all copies.  Else use accelerator_view::copy.
// usePinnedHost = allocate pinned memory.  Else use malloc for allocations.
// syncAfter* - wait for accelerator_view to drain after H2D,Kernel,D2H.  Makes test easier since device-side dependency resolution not used.
//
// Designed to stress a small number of simple smoke tests
template <typename T>
void simpleTest1(hc::accelerator &acc, size_t N, bool allocPinned, bool useAsyncCopy, bool syncAfterH2D, bool syncAfterKernel, bool syncAfterD2H, int db=0)
{
    size_t Nbytes = N*sizeof(T);

    printf ("test: %s", __func__);
    printf ("  N=%zu Nbytes=%6.2fMB", N, Nbytes/1024.0/1024.0);
    printf ("  allocPinned=%d useAsyncCopy=%d, syncAfterH2D=%d, syncAfterKernel=%d, syncAfterD2H=%d\n", 
            allocPinned, useAsyncCopy, syncAfterH2D, syncAfterKernel, syncAfterD2H);

    int *A_d, *B_d, *C_d;
    int *A_h, *B_h, *C_h;

    A_h = hostAlloc(acc, allocPinned, Nbytes);
    B_h = hostAlloc(acc, allocPinned, Nbytes);
    C_h = hostAlloc(acc, allocPinned, Nbytes);

    A_d = hc::am_alloc(Nbytes, acc, 0);
    B_d = hc::am_alloc(Nbytes, acc, 0);
    C_d = hc::am_alloc(Nbytes, acc, 0);

    assert(A_d);
    assert(B_d);
    assert(C_d);

    // Initialize the host data:
    for (size_t i=0; i<N; i++) {
        (A_h)[i] = 3.146f + i; // Pi
        (B_h)[i] = 1.618f + i; // Phi
        (C_h)[i] = 1000.0f + i; 
    }

    if (db) {
      printf ("A_d=%p B_d=%p C_d=%p  A_h=%p B_h=%p C_h=%p\n", A_d, B_d, C_d, A_h, B_d, C_h);
    }

    hc::accelerator_view av = acc.get_default_view();

    memcopy(av, useAsyncCopy, A_h, A_d, Nbytes);
    memcopy(av, useAsyncCopy, B_h, B_d, Nbytes);
    if (db) {
        printf ("db: H2D copies launched\n");
    }

    if (syncAfterH2D) {
        if (db) {
            printf ("db: syncAfterH2D av.wait()\n");
        }
        av.wait();
    }

    // Loop below can't handle big sizes:
    assert (N < std::numeric_limits<int>::max());

    hc::parallel_for_each(av, hc::extent<1>(N) , [=](hc::index<1> idx) [[hc]]  {
          int i = amp_get_global_id(0);
          C_d[i] = A_d[i] + B_d[i] ;
    });

    if (db) {
        printf ("db: Kernel launched\n");
    }

    if (syncAfterKernel) {
        if (db) {
            printf ("db: syncAfterKernel av.wait()\n");
        }
        av.wait();
    }

    memcopy(av, useAsyncCopy, C_d, C_h, Nbytes);
    if (db) {
        printf ("db: D2H launched\n");
    }

    if (useAsyncCopy) {
        assert (syncAfterD2H); // have to sync before using on host.
    }

    if (syncAfterD2H) {
        if (db) {
            printf ("db: syncAfterD2H av.wait()\n");
        }
        av.wait();
    }

    checkVectorADD(A_h, B_h, C_h, N);


    printf ("  %s success\n\n", __func__);
}


int main() 
{
  std::vector<hc::accelerator> accs = hc::accelerator::get_all();
  hc::accelerator gpu_acc;
  for (auto& it: accs)
    if (! it.get_is_emulated()) {
      gpu_acc = it;
      break;
    }


  size_t N = 1024*256; 

  // In-progress:
  // Test H2D -> kernel -> D2H dependency.  If this fails, likely indicates problem with D2H
  simpleTest1<float>(gpu_acc, N, true/*allocPinned*/,  true/*useAsyncCopy*/,  false/*syncAfterH2D*/, false/*syncAfterKernel*/, true/*syncAfterD2H*/, 0);

  // Use async calls, but av.wait() after all the important steps:
  simpleTest1<float>(gpu_acc, N, true/*allocPinned*/,  true/*useAsyncCopy*/,  true/*syncAfterH2D*/, true/*syncAfterKernel*/, true/*syncAfterD2H*/, 0);

  // Test H2D -> kernel dependency:
  simpleTest1<float>(gpu_acc, N, true/*allocPinned*/,  true/*useAsyncCopy*/,  false/*syncAfterH2D*/, true/*syncAfterKernel*/, true/*syncAfterD2H*/, 0);

  

  // Synchronous cases:
  simpleTest1<float>(gpu_acc, N, true/*allocPinned*/,  false/*useAsyncCopy*/,  true/*syncAfterH2D*/, true/*syncAfterKernel*/, true/*syncAfterD2H*/);
  simpleTest1<float>(gpu_acc, N, false/*allocPinned*/, false/*useAsyncCopy*/,  true/*syncAfterH2D*/, true/*syncAfterKernel*/, true/*syncAfterD2H*/);

  return 0;


}
