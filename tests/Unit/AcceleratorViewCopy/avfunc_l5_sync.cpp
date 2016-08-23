// RUN: %hc %s -o %t.out -lhc_am && %t.out 
#include <stdlib.h>
#include <iostream>

#include <hc.hpp>
#include <hc_am.hpp>


// A few helper routines for writing tests:
#include "common.h"

#define LEN 1024

/*
  1 - Host to Device Memcpy
  2 - Kernel Launch
  3 - Device to Device Memcpy
  4 - Device to Host Memcpy
  5 - Host to Host Memcpy
*/

static hc::accelerator acc;

int* initDeviceArrays(size_t length){
    return (int*)hc::am_alloc(length*sizeof(int), acc, false);
}

int* initHostArrays(size_t length, int val){
    int* hPtr = hc::am_alloc(length*sizeof(int), acc, true);
    for(uint32_t i=0;i<length;i++){
        hPtr[i] = val;
    }
    return hPtr;
}

void memcpyHtoD(hc::accelerator_view &av, int *dst, int *src, size_t len){
    av.copy(src, dst, sizeof(int)*len);
}

void memcpyDtoH(hc::accelerator_view &av, int *dst, int *src, size_t len){
    av.copy(src, dst, sizeof(int)*len);
}

void memcpyDtoD(hc::accelerator_view &av, int *dst, int *src, size_t len){
    av.copy(src, dst, sizeof(int)*len);
}

void memcpyHtoH(hc::accelerator_view &av, int *dst, int *src, size_t len){
    memcpy(dst, src, sizeof(int)*len);
}

inline void assertTwoArrays(int *A, int *B, size_t length){
    for(uint32_t i=0;i<length;i++){
      if(A[i] != B[i]){std::cout<<A[i]<<" "<<B[i]<<" "<<i<<std::endl;assert(0);}
    }
}

void runKernel(hc::accelerator_view &av, int *Ad){
    hc::parallel_for_each(av, hc::extent<1>(LEN), [=](hc::index<1> idx)[[hc]]{
        int i = amp_get_global_id(0);
        Ad[i] = Ad[i] + 1;
    });
}

void do12345(hc::accelerator_view &av){
    int *A = initHostArrays(LEN, 1);
    int *B = initHostArrays(LEN, 2);
    int *C = initHostArrays(LEN, 3);
    int *Ad = initDeviceArrays(LEN);
    int *Bd = initDeviceArrays(LEN);

    memcpyHtoD(av, Ad, A, LEN);
    runKernel(av, Ad);
    memcpyDtoD(av, Bd, Ad, LEN);
    memcpyDtoH(av, C, Bd, LEN);
    memcpyHtoH(av, A, C, LEN);
    assertTwoArrays(A, B, LEN);
    
}

void do12354(hc::accelerator_view &av){
    int *A = initHostArrays(LEN, 1);
    int *Ad = initDeviceArrays(LEN);
    int *B = initHostArrays(LEN, 2);
    int *C = initHostArrays(LEN, 3);
    int *Bd = initDeviceArrays(LEN);

    memcpyHtoD(av, Ad, A, LEN);
    runKernel(av, Ad);
    memcpyDtoD(av, Bd, Ad, LEN);
    memcpyHtoH(av, C, B, LEN);
    memcpyDtoH(av, A, Bd, LEN);
    assertTwoArrays(A, C, LEN);
}

void do12435(hc::accelerator_view &av){
    int *A = initHostArrays(LEN, 1);
    int *B = initHostArrays(LEN, 2);
    int *C = initHostArrays(LEN, 3);
    int *Ad = initDeviceArrays(LEN);
    int *Bd = initDeviceArrays(LEN);

    memcpyHtoD(av, Ad, A, LEN);
    runKernel(av, Ad);
    memcpyDtoH(av, C, Ad, LEN);
    memcpyDtoD(av, Bd, Ad, LEN);
    memcpyHtoH(av, A, C, LEN);

    memcpyDtoH(av, C, Bd, LEN);
    assertTwoArrays(C, A, LEN);
    assertTwoArrays(B, C, LEN);
}

void do12453(hc::accelerator_view &av){
    int *A = initHostArrays(LEN, 1);
    int *B = initHostArrays(LEN, 2);
    int *C = initHostArrays(LEN, 3);
    int *Ad = initDeviceArrays(LEN);
    int *Bd = initDeviceArrays(LEN);

    memcpyHtoD(av, Ad, A, LEN);
    runKernel(av, Ad);
    memcpyDtoH(av, C, Ad, LEN);
    memcpyHtoH(av, A, C, LEN);
    memcpyDtoD(av, Bd, Ad, LEN);

    memcpyDtoH(av, C, Bd, LEN);

    assertTwoArrays(C, B, LEN);
    assertTwoArrays(A, B, LEN);
}

void do12534(hc::accelerator_view &av){
    int *A = initHostArrays(LEN, 1);
    int *B = initHostArrays(LEN, 2);
    int *C = initHostArrays(LEN, 3);
    int *Ad = initDeviceArrays(LEN);
    int *Bd = initDeviceArrays(LEN);

    memcpyHtoD(av, Ad, A, LEN);
    runKernel(av, Ad);
    memcpyHtoH(av, C, B, LEN);
    memcpyDtoD(av, Bd, Ad, LEN);
    memcpyDtoH(av, A, Bd, LEN);

    assertTwoArrays(A, B, LEN);
    assertTwoArrays(C, B, LEN);
}

void do12543(hc::accelerator_view &av){
    int *A = initHostArrays(LEN, 1);
    int *B = initHostArrays(LEN, 2);
    int *C = initHostArrays(LEN, 3);
    int *D = initHostArrays(LEN, 4);
    int *Ad = initDeviceArrays(LEN);
    int *Bd = initDeviceArrays(LEN);

    memcpyHtoD(av, Ad, A, LEN);
    runKernel(av, Ad);
    memcpyHtoH(av, C, B, LEN);
    memcpyDtoH(av, A, Ad, LEN);
    memcpyDtoD(av, Bd, Ad, LEN);

    memcpyDtoH(av, D, Bd, LEN);

    assertTwoArrays(C, B, LEN);
    assertTwoArrays(A, B, LEN);
    assertTwoArrays(D, B, LEN);
}

void do13245(hc::accelerator_view &av){
    int *A = initHostArrays(LEN, 1);
    int *B = initHostArrays(LEN, 2);
    int *C = initHostArrays(LEN, 3);
    int *Ad = initDeviceArrays(LEN);
    int *Bd = initDeviceArrays(LEN);

    memcpyHtoD(av, Ad, A, LEN);
    memcpyDtoD(av, Bd, Ad, LEN);
    runKernel(av, Bd);
    memcpyDtoH(av, C, Bd, LEN);
    memcpyHtoH(av, A, C, LEN);
    assertTwoArrays(A, B, LEN);
}

void do13254(hc::accelerator_view &av){
    int *A = initHostArrays(LEN, 1);
    int *B = initHostArrays(LEN, 2);
    int *C = initHostArrays(LEN, 3);
    int *Ad = initDeviceArrays(LEN);
    int *Bd = initDeviceArrays(LEN);

    memcpyHtoD(av, Ad, A, LEN);
    memcpyDtoD(av, Bd, Ad, LEN);
    runKernel(av, Bd);
    memcpyHtoH(av, C, B, LEN);
    memcpyHtoD(av, A, Bd, LEN);

    assertTwoArrays(A, B, LEN);
    assertTwoArrays(C, B, LEN);
}

void do13425(hc::accelerator_view &av){
    int *A = initHostArrays(LEN, 1);
    int *B = initHostArrays(LEN, 2);
    int *C = initHostArrays(LEN, 3);
    int *D = initHostArrays(LEN, 4);
    int *Ad = initDeviceArrays(LEN);
    int *Bd = initDeviceArrays(LEN);

    memcpyHtoD(av, Ad, A, LEN);
    memcpyDtoD(av, Bd, Ad, LEN);
    memcpyDtoH(av, C, Bd, LEN);
    runKernel(av, Bd);
    memcpyHtoH(av, A, B, LEN);

    memcpyDtoH(av, D, Bd, LEN);

    assertTwoArrays(A, B, LEN);
    assertTwoArrays(D, B, LEN);
}

void do13452(hc::accelerator_view &av){
    int *A = initHostArrays(LEN, 1);
    int *B = initHostArrays(LEN, 2);
    int *C = initHostArrays(LEN, 3);
    int *D = initHostArrays(LEN, 4);
    int *Ad = initDeviceArrays(LEN);
    int *Bd = initDeviceArrays(LEN);

    memcpyHtoD(av, Ad, A, LEN);
    memcpyDtoD(av, Bd, Ad, LEN);
    memcpyDtoH(av, C, Bd, LEN);
    memcpyHtoH(av, D, C, LEN);
    runKernel(av, Bd);

    memcpyDtoH(av, C, Bd, LEN);
    assertTwoArrays(A, D, LEN);
    assertTwoArrays(B, C, LEN);

}

void initAMP(){
    std::vector<hc::accelerator> accs = hc::accelerator::get_all();
    for(auto& it:accs){
        if(!it.get_is_emulated()){
            acc = it;
            break;
        }
    }
}

int main(){
    initAMP();
    hc::accelerator_view av = acc.get_default_view();
    do12345(av);
    do12354(av);
    do12435(av);
    do12453(av);
    do12534(av);
    do12543(av);


    do13245(av);
    do13254(av);
    do13425(av);
    do13452(av);
}
