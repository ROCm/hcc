float *A, *B, *C;
float *Ad, *Bd, *Cd;
float *Ah, *Bh, *Ch;

inline void Memcpy(hc::accelerator_view &av, const void *Src, void *Dst, size_t Size){
    av.copy(Src, Dst, Size);
}

inline hc::completion_future MemcpyAsync(hc::accelerator_view &av, const void *Src, void *Dst, size_t Size){
    return av.copy_async(Src, Dst, Size);
}

inline void* HostAlloc(hc::accelerator &Acc, size_t Size){
    return hc::am_alloc(Size, Acc, amHostPinned);
}

inline void* DeviceAlloc(hc::accelerator &Acc, size_t Size){
    return hc::am_alloc(Size, Acc, 0x0);
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

hc::completion_future Test4(hc::accelerator_view &Av){
    return MemcpyAsync(Av, Ah, Ad, size);
}

hc::completion_future Test5(hc::accelerator_view &Av){
    return MemcpyAsync(Av, Bd, Bh, size);
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


void Destroy(){
    free(A);
    free(B);
    free(C);

    hc::am_free(Ad);
    hc::am_free(Bd);
    hc::am_free(Cd);

    hc::am_free(Ah);
    hc::am_free(Bh);
    hc::am_free(Ch);
}
