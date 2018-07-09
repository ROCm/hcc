// RUN: %hc %s -lhc_am -o %t.out && %t.out

#include <cstdlib>
#include <cstdio>
#include <hc.hpp>
#include <hc_am.hpp>
#include <iostream>

#define TRACKER_PRINT(_target)\
{\
    std::cerr << "\nhc::am_memtracker_print(" << #_target << "==" << (void*)(_target) << ");\n";\
    hc::am_memtracker_print(_target);\
}


int main()
{

    hc::accelerator acc;

    char *a = am_aligned_alloc(10000, acc, 0, 65536);
    char *b = am_alloc(20000, acc, 0);

    // print the whole table:

    TRACKER_PRINT(0x0);

    TRACKER_PRINT(a);
    TRACKER_PRINT(b);

    hc::am_free(b);
    hc::am_free(a);

    return 0; // passed!
}
