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
    bool ret = true;
    char *a = am_aligned_alloc(10000, acc, 0, 65536);
    char *b = am_alloc(20000, acc, 0);

    // print the whole table:

    TRACKER_PRINT(0x0);

    TRACKER_PRINT(a);
    TRACKER_PRINT(b);
    hc::AmPointerInfo amPointerInfo(NULL, NULL, NULL, 0, acc, 0, 0);
    am_status_t status = hc::am_memtracker_getinfo(&amPointerInfo, b);
    if (status == AM_SUCCESS) {
       if (amPointerInfo._hostPointer == NULL) {
           hc::am_free(b);
       }
       else { 
           printf("Failed device pointer check for b\n");
           ret = false;
       }
    } else {
           printf("Failed tracker info for b\n");
           ret = false;
    }

    status = hc::am_memtracker_getinfo(&amPointerInfo, a);
    if (status == AM_SUCCESS) {
       if (amPointerInfo._hostPointer == NULL)
           hc::am_free(a);
       else {
           printf("Failed device pointer check for a\n");
           ret = false;
       }
    } else {
        printf("Failed tracker info for a\n");
        ret = false;
    }
    return !(ret == true);
}
