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

inline void Memcpy(hc::accelerator_view &av, const void *Src, void *Dst, size_t Size){
    av.copy(Src, Dst, Size);
}

inline void MemcpyAsync(hc::accelerator_view &av, const void *Src, void *Dst, size_t Size){
    av.copy_async(Src, Dst, Size);
}

inline void* HostAlloc(hc::accelerator &Acc, size_t Size){
    return hc::am_alloc(Size, Acc, true);
}

inline void* DeviceAlloc(hc::accelerator &Acc, size_t Size){
    return hc::am_alloc(Size, Acc, false);
}

void RunKernel(hc::accelerator_view &Av, float *Ad, float *Bd, float *Cd){
    hc::parallel_for_each(Av, hc::extent<1>(N), [=](hc::index<1> idx)[[hc]] {
    int i = amp_get_global_id(0);
    Cd[i] = Ad[i] + Bd[i];
    });
}

void Test3(hc::accelerator_view &Av){
    Memcpy(Av, Bd, B, size);
}

void Test2(hc::accelerator_view &Av){
    RunKernel(Av, Ad, Bd, Cd);
}

void Test1(hc::accelerator_view &Av){
    Memcpy(Av, A, Ad, size);
}

void Test4(hc::accelerator_view &Av){
    MemcpyAsync(Av, Ah, Ad, size);
}

void Test5(hc::accelerator_view &Av){
    MemcpyAsync(Av, Bd, Bh, size);
}


void Init(hc::accelerator &Ac){
    A = (float*)malloc(size);
    B = (float*)malloc(size);
    C = (float*)malloc(size);

    Ad = (float*)DeviceAlloc(Ac, size);
    Bd = (float*)DeviceAlloc(Ac, size);
    Cd = (float*)DeviceAlloc(Ac, size);

    Ah = (float*)HostAlloc(Ac, size);
    Bh = (float*)HostAlloc(Ac, size);
    Ch = (float*)HostAlloc(Ac, size);

    for(uint32_t i=0;i<N;i++){
        A[i] = 3.146f + i;
        B[i] = 1.618f + i;
        Ah[i] = A[i];
        Bh[i] = B[i];
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
    for(uint32_t i=0;i<SHRT_MAX;i++){
        Test1(av);
    }

    for(uint32_t i=0;i<SHRT_MAX;i++){
        Test2(av);
    }

    for(uint32_t i=0;i<SHRT_MAX;i++){
        Test3(av);
    }

    for(uint32_t i=0;i<SHRT_MAX;i++){
        Test4(av);
    }

    for(uint32_t i=0;i<SHRT_MAX;i++){
        Test5(av);
    }

}

