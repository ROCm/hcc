
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

    const size_t cSize = 30000;


    char *a = am_alloc(10000, acc, 0);
    char *b = am_alloc(20000, acc, 0);
    char *c = am_alloc(cSize, acc, 0);

    // print the whole table:

    TRACKER_PRINT(0x0);

    TRACKER_PRINT(a);
    TRACKER_PRINT(c);

    TRACKER_PRINT(c+5000);
    TRACKER_PRINT(a+5000);


    TRACKER_PRINT(c+cSize); // OOB
    TRACKER_PRINT(c-1);
    TRACKER_PRINT(c+cSize+1000);


    return 0; // passed!
}
