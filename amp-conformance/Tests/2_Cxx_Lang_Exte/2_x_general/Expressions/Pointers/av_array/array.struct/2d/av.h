// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
// Test three scenarios. local memory, global memory and shared meory.

#include <amptest.h>
#include <amptest_main.h>

#ifndef AMP_ELEMENT_TYPE
#define AMP_ELEMENT_TYPE float
#endif

using std::vector;
using namespace concurrency;
using namespace concurrency::Test;

const static int DOMAIN_SIZE_1D = 64;
const static int BLOCK_SIZE_1D = 8;
const int LOCAL_SIZE = 0x4;

template<typename type, int rank>
struct s1
{
    s1(array<type, rank> &a, array<type, rank> &b, array<type, rank> &c) __GPU : av_a(a), av_b(b), av_c(c) {}
    ~s1() __GPU {}

    int placeholder1;
    array_view<type, rank> av_a;
    array_view<type, rank> av_b;
    array_view<type, rank> av_c;
    float placeholder2;
};

template<typename type, int rank>
struct s2
{
    s2(array<type, rank> &a, array<type, rank> &b, array<type, rank> &c) __GPU : av_a(a), av_b(b), av_c(c) {}
    s2(extent<rank> e, type *pa, type *pb, type *pc) __GPU : av_a(e, pa), av_b(e, pb), av_c(e, pc) {}
    ~s2() __GPU {}

    int placeholder1;
    array_view<type, rank> av_a;
    array_view<type, rank> av_b;
    array_view<type, rank> av_c;
    float placeholder2;
};

template<typename type>
void cf_test(type *pa, type *pb, type *pc, array_view<int, 1> &flag) __GPU_ONLY;

template<typename type>
struct kernel_local
{
    static void func(tiled_index<BLOCK_SIZE_1D, BLOCK_SIZE_1D> idx, s1<type, 2> *p, array_view<int, 1> &flag, int b1, int b2, int b3, int b4) __GPU_ONLY
        /*
        Test function tests pointers which point to local memory.
        idx: compute index
        p: input
        flag: control flags, which is used to test control flow.
        b1, b2, b3, b4: control flags. It's used to test pointer emulation. Make sure pointer can point to the correct values.
        */
    {
        type local_a[LOCAL_SIZE * LOCAL_SIZE];
        type local_fa[LOCAL_SIZE * LOCAL_SIZE];
        type local_b[LOCAL_SIZE * LOCAL_SIZE];
        type local_fb[LOCAL_SIZE * LOCAL_SIZE];
        type local_c[LOCAL_SIZE * LOCAL_SIZE];
        type local_fc[LOCAL_SIZE * LOCAL_SIZE];

        for (int i = 0; i < LOCAL_SIZE * LOCAL_SIZE; i++)
        {
            local_a[i] = p->av_a[idx.global];
            local_fa[i] = local_a[i] + 1;
            local_b[i] = p->av_b[idx.global];
            local_fb[i] = local_fb[i] + 1;
            local_c[i] = p->av_c[idx.global];
            local_fc[i] = local_c[i] + 1;
        }

        extent<2> e(LOCAL_SIZE, LOCAL_SIZE);

        s2<type, 2> inter_o(e, local_a, local_b, local_c);

        s2<type, 2> inter_fo(e, local_fa, local_fb, local_fc);

        type *pa = NULL, *pb = NULL, *pc =NULL;

        for (int y = 0; y < LOCAL_SIZE; y++)
        {
            for (int x = 0; x < LOCAL_SIZE; x++)
            {
                if (!b2 || b1)
                {
                    pa = &inter_fo.av_a[y][x];
                    pb = &inter_fo.av_b[y][x];
                    pc = &inter_fo.av_c[y][x];
                } else
                {
                    pa = &inter_o.av_a[y][x];
                    pb = &inter_o.av_b[y][x];
                    pc = &inter_o.av_c[y][x];
                }

                cf_test(pa, pb, pc, flag);
            }
        }

        type fc = 0;

        for (int y = 0; y < LOCAL_SIZE; y++)
        {
            for (int x = 0; x < LOCAL_SIZE; x++)
            {
                fc += inter_o.av_c[y][x];
            }
        }

        p->av_c[idx.global] = fc;
    }
};

template<typename type>
struct kernel_global
{
    static void func(tiled_index<BLOCK_SIZE_1D, BLOCK_SIZE_1D> idx, s1<type, 2> *p, array_view<int, 1> &flag, int b1, int b2, int b3, int b4) __GPU_ONLY
        /*
        Test function tests pointers which point to local memory.
        idx: compute index
        p: input
        flag: control flags, which is used to test control flow.
        b1, b2, b3, b4: control flags. It's used to test pointer emulation. Make sure pointer can point to the correct values.
        */
    {
        type *pa = NULL, *pb = NULL, *pc =NULL;

        if (!b2 || b1)
        {
            pa = &p->av_c[idx];
            pb = &p->av_b[idx];
            pc = &p->av_a[idx];
        } else
        {
            pa = &p->av_a[idx];
            pb = &p->av_b[idx];
            pc = &p->av_c[idx];
        }

        cf_test<type>(pa, pb, pc, flag);
        *pc *= LOCAL_SIZE * LOCAL_SIZE;
    }
};

template<typename type>
struct kernel_shared
{
    static void func(tiled_index<BLOCK_SIZE_1D, BLOCK_SIZE_1D> idx, s1<type, 2> *p, array_view<int, 1> &flag, int b1, int b2, int b3, int b4) __GPU_ONLY
        /*
        Test function tests pointers which point to local memory.
        idx: compute index
        p: input
        flag: control flags, which is used to test control flow.
        b1, b2, b3, b4: control flags. It's used to test pointer emulation. Make sure pointer can point to the correct values.
        */
    {
        int local_idx = idx.local[0] * BLOCK_SIZE_1D + idx.local[1];

        tile_static type share_a[BLOCK_SIZE_1D * BLOCK_SIZE_1D];
        share_a[local_idx] = p->av_a[idx.global];
        tile_static type share_fa[BLOCK_SIZE_1D * BLOCK_SIZE_1D];
        share_fa[local_idx] =  share_a[local_idx] + 1;

        tile_static type share_b[BLOCK_SIZE_1D * BLOCK_SIZE_1D];
        share_b[local_idx] = p->av_b[idx.global];
        tile_static type share_fb[BLOCK_SIZE_1D * BLOCK_SIZE_1D];
        share_fb[local_idx] =  share_b[local_idx] + 1;

        tile_static type share_c[BLOCK_SIZE_1D * BLOCK_SIZE_1D];
        share_c[local_idx] = p->av_c[idx.global];
        tile_static type share_fc[BLOCK_SIZE_1D * BLOCK_SIZE_1D];
        share_fc[local_idx] =  share_c[local_idx] + 1;

        idx.barrier.wait();

        s2<type, 2> inter_o(extent<2>(BLOCK_SIZE_1D, BLOCK_SIZE_1D), share_a, share_b, share_c);

        s2<type, 2> inter_fo(extent<2>(BLOCK_SIZE_1D, BLOCK_SIZE_1D), share_fa, share_fb, share_fc);

        type *pa = NULL, *pb = NULL, *pc =NULL;

        if (!b2 || b1)
        {
            pa = &inter_fo.av_a[idx.local];
            pb = &inter_fo.av_b[idx.local];
            pc = &inter_fo.av_c[idx.local];
        } else
        {
            pa = &inter_o.av_a[idx.local];
            pb = &inter_o.av_b[idx.local];
            pc = &inter_o.av_c[idx.local];
        }

        cf_test(pa, pb, pc, flag);

        idx.barrier.wait();

        p->av_c[idx.global] = inter_o.av_c[idx.local] * LOCAL_SIZE * LOCAL_SIZE;
    }
};

template<typename type, typename k>
void RunMyKernel(vector<type> &a, vector<type> &b, vector<type> &c, vector<type> &fa, vector<type> &fb, vector<type> &fc, vector<int> &flag, accelerator_view av)
{
    extent<2> g(DOMAIN_SIZE_1D, DOMAIN_SIZE_1D);
    array<type, 2> a_a(g, a.begin(), av);
    array<type, 2> a_b(g, b.begin(), av);
    array<type, 2> a_c(g, c.begin(), av);
    array<type, 2> a_fa(g, fa.begin(), av);
    array<type, 2> a_fb(g, fb.begin(), av);
    array<type, 2> a_fc(g, fc.begin(), av);
    extent<1> e_flag(DOMAIN_SIZE_1D);
    array<int, 1> a_flag(e_flag, flag.begin(), av);

    int b1 = 0;
    int b2 = 1;
    int b3 = 3;
    int b4 = 5;

    parallel_for_each(a_a.get_extent().template tile<BLOCK_SIZE_1D, BLOCK_SIZE_1D>(), [&, b1, b2, b3, b4] (tiled_index<BLOCK_SIZE_1D, BLOCK_SIZE_1D> idx) __GPU_ONLY {

        s1<type, 2> o(a_a, a_b, a_c);
        s1<type, 2> o_f(a_fa, a_fb, a_fc);
        array_view<int, 1> av_flag(a_flag);

        s1<type, 2> *p = !b1 ? &(o) : &(o_f);

        k::func(idx, p, av_flag, b1, b2, b3, b4);
    });

    c = a_c;
}

template<typename type>
void init(vector<type> &a, vector<type> &b, vector<type> &c, vector<type> &fa, vector<type> &fb, vector<type> &fc, vector<type> &ref_c, vector<int> &flag);

template<typename type, typename k>
bool test(accelerator_view av)
{
    vector<type> a(DOMAIN_SIZE_1D * DOMAIN_SIZE_1D);
    vector<type> b(DOMAIN_SIZE_1D * DOMAIN_SIZE_1D);
    vector<type> c(DOMAIN_SIZE_1D * DOMAIN_SIZE_1D);
    vector<type> fa(DOMAIN_SIZE_1D * DOMAIN_SIZE_1D);
    vector<type> fb(DOMAIN_SIZE_1D * DOMAIN_SIZE_1D);
    vector<type> fc(DOMAIN_SIZE_1D * DOMAIN_SIZE_1D);
    vector<type> ref_c(DOMAIN_SIZE_1D * DOMAIN_SIZE_1D);
    vector<int> flag(DOMAIN_SIZE_1D);

    init(a, b, c, fa, fb, fc, ref_c, flag);

    RunMyKernel<type, k>(a, b, c, fa, fb, fc, flag, av);

    bool ret = Verify(c, ref_c);

    return ret;
}

runall_result test_main()
{
    srand(2010);

    accelerator_view av = require_device_with_double(Device::ALL_DEVICES).get_default_view();

    Log(LogType::Info, true) << "test in local memory: \n";
    if (!test<AMP_ELEMENT_TYPE, kernel_local<AMP_ELEMENT_TYPE>>(av)) return runall_fail;
    Log(LogType::Info, true) << "pass\n";

    Log(LogType::Info, true) << "test in global memory: \n";
    if (!test<AMP_ELEMENT_TYPE, kernel_global<AMP_ELEMENT_TYPE>>(av)) return runall_fail;
    Log(LogType::Info, true) << "pass\n";

    Log(LogType::Info, true) << "test in shared memory: \n";
    if (!test<AMP_ELEMENT_TYPE, kernel_shared<AMP_ELEMENT_TYPE>>(av)) return runall_fail;
    Log(LogType::Info, true) << "pass\n";

    return runall_pass;
}

