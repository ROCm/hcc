// RUN: %hc %s -o %t.out -lhc_am && %t.out
#include<stdlib.h>
#include<iostream>

#include<hc.hpp>
#include<hc_am.hpp>

#include"common.h"

#define N 1024*1024

const size_t size = sizeof(float) * N;
float *A, *B, *C;
float *Ad, *Bd, *Cd;
float *Ah, *Bh, *Ch;

#include "common2.h"

template<int TEST_NUM, typename TEST_FUNC>
void RUN_TEST(TEST_NUM, TEST_FUNC)
{
    if (testsToRun & (1<< testNum)) {
        for(uint32_t i=0;testIters;i++){
            if ((i%1000 == 0)) {
                printf ("info: running Test%d %5d/%5d\n", i, TEST_NUM, testIters);
            }
            TEST_FUNC(av);
        }
    }
}


int main(){
    std::vector<hc::accelerator> accs = hc::accelerator::get_all();
    hc::accelerator gpu_acc;
    for(auto& it:accs){
        if(!it.get_is_emulated()){
            gpu_acc = it;
            break;
        }
    }

    Init(gpu_acc);
    hc::accelerator_view av = gpu_acc.get_default_view();



    unsigned testsToRun = 0xFF;
    int testIters = (i<SHRT_MAX);


    

    if (testsToRun & 0x1) {
        for(uint32_t i=0;i<SHRT_MAX;i++){
            if ((i%1000 == 0)) {
                printf ("info: running Test1 %5d/%5d\n", i, SHRT_MAX);
            }
            Test1(av);
        }
    }

    if (testsToRun & 0x2) {
        for(uint32_t i=0;i<SHRT_MAX;i++){
            if ((i%1000 == 0)) {
                printf ("info: running Test2 %5d/%5d\n", i, SHRT_MAX);
            }
            Test2(av);
        }
    }

    if (testsToRun & 0x4) {
        for(uint32_t i=0;i<SHRT_MAX;i++){
            if ((i%1000 == 0)) {
                printf ("info: running Test3 %5d/%5d\n", i, SHRT_MAX);
            }
            Test3(av);
        }
    }

    if (testsToRun & 0x8) {
        for(uint32_t i=0;i<SHRT_MAX;i++){
            if ((i%1000 == 0)) {
                printf ("info: running Test4 %5d/%5d\n", i, SHRT_MAX);
            }
            Test4(av);
        }
    }

    if (testsToRun & 0x10) {
        for(uint32_t i=0;i<SHRT_MAX;i++){
            if ((i%1000 == 0)) {
                printf ("info: running Test5 %5d/%5d\n", i, SHRT_MAX);
            }
            Test5(av);
        }
    }


    if (testsToRun & 0x20) {
        // Sync + Kernel + Sync
        Test1(av);
        Test2(av);
        Test3(av);
        // Async + Kernel + Async
        Test4(av);
        Test2(av);
        Test5(av);
    }
    
}

