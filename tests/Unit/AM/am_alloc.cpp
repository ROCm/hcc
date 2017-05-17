
// RUN: %hc %s -lhc_am -o %t.out && %t.out

#include <cstdlib>
#include <cstdio>
#include <hc.hpp>
#include <hc_am.hpp>
#include <iostream>


// Simple tests to verify regions and pointers all work:
int main() 
{

    hc::accelerator acc;

    const size_t cSize = 30000;


    char *a = am_alloc(10000, acc, 0);
    char *b = am_alloc(20000, acc, amHostPinned);
    char *c = am_alloc(cSize, acc, amHostCoherent);

    // Simple tests to verify that the memory allocations to all 3 regions succeeded.
    assert(a);
    assert(b);
    assert(c);

    assert (hc::am_free(a) == AM_SUCCESS);
    assert (hc::am_free(b) == AM_SUCCESS);
    assert (hc::am_free(c) == AM_SUCCESS);

    //  Try duplicate free, should return error:
    assert (hc::am_free(c) != AM_SUCCESS);


    return 0; // passed!
}
