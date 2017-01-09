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

const static int DOMAIN_SIZE = 64 * 64;
const static int BLOCK_SIZE = 16;
const int LOCAL_SIZE = 0xF;

template<typename type, int rank>
struct sa
{
    sa(array<type, rank> &a) __GPU : av_a(a) {}
    sa(extent<rank> e, type *pa) __GPU : av_a(e, pa) {}
    ~sa() __GPU {}

    int placeholder1;
    array_view<type, rank> av_a;
    float placeholder2;
};

template<typename type, int rank>
struct sbc
{
    sbc(array<type, rank> &b, array<type, rank> &c) __GPU : av_b(b), av_c(c) {}
    sbc(extent<rank> e, type *pb, type *pc) __GPU : av_b(e, pb), av_c(e, pc) {}
    ~sbc() __GPU {}

    int placeholder1;
    array_view<type, rank> av_b;
    array_view<type, rank> av_c;
    float placeholder2;
};

template<typename type>
void cf_test(type &pa, type &pb, type &pc, array_view<int, 1> &flag) __GPU_ONLY;

template<typename type>
struct kernel_local
{
    static void func(tiled_index<BLOCK_SIZE> idx, sa<type, 1> &oa, sbc<type, 1> &obc, sa<type, 1> &foa, sbc<type, 1> &fobc, array_view<int, 1> &flag, int b1, int b2, int b3, int b4) __GPU_ONLY
        /*
        Test function tests pointers which point to local memory.
        idx: compute index
        oa: input, the first operand
        obc: input, the second operand
        foa: fake input a, which is used to test pointer eumlation.
        fobc: fake input b, which is used to test pointer eumlation.
        flag: control flags, which is used to test control flow.
        b1, b2, b3, b4: control flags. It's used to test pointer emulation. Make sure pointer can point to the correct values.
        */
    {
        type local_a[LOCAL_SIZE];
        type local_fa[LOCAL_SIZE];
        type local_b[LOCAL_SIZE];
        type local_fb[LOCAL_SIZE];
        type local_c[LOCAL_SIZE];
        type local_fc[LOCAL_SIZE];

        for (int i = 0; i < LOCAL_SIZE; i++)
        {
            local_a[i] = oa.av_a[idx.global];
            local_fa[i] = foa.av_a[idx.global];
            local_b[i] = obc.av_b[idx.global];
            local_fb[i] = fobc.av_b[idx.global];
            local_c[i] = obc.av_c[idx.global];
            local_fc[i] = fobc.av_c[idx.global];
        }

        extent<1> e(LOCAL_SIZE);

        sa<type, 1> inter_oa(e, !b1 ? local_a : local_fa);
        sbc<type, 1> inter_obc(e, !b1 ? local_b : local_fb, !b1 ? local_c : local_fc);

        sa<type, 1> inter_foa(e, b1 ? local_a : local_fa);
        sbc<type, 1> inter_fobc(e, b1 ? local_b : local_fb, b1 ? local_c : local_fc);

        for (int i = 0; i < LOCAL_SIZE; i++)
        {
            cf_test(!b2 ? inter_foa.av_a[i] : inter_oa.av_a[i], !b2 ? inter_fobc.av_b[i] : inter_obc.av_b[i], !b2 ? inter_fobc.av_c[i] : inter_obc.av_c[i], flag);
        }

        type fc = 0;

        for (int i = 0; i < LOCAL_SIZE; i++)
        {
            fc += inter_obc.av_c[i];
        }

        obc.av_c[idx.global] = fc;
    }
};

template<typename type>
struct kernel_global
{
    static void func(tiled_index<BLOCK_SIZE> idx, sa<type, 1> &oa, sbc<type, 1> &obc, sa<type, 1> &foa, sbc<type, 1> &fobc, array_view<int, 1> &flag, int b1, int b2, int b3, int b4) __GPU_ONLY
        /*
        Test function tests pointers which point to local memory.
        idx: compute index
        oa: input, the first operand
        obc: input, the second operand
        foa: fake input a, which is used to test pointer eumlation.
        fobc: fake input b, which is used to test pointer eumlation.
        flag: control flags, which is used to test control flow.
        b1, b2, b3, b4: control flags. It's used to test pointer emulation. Make sure pointer can point to the correct values.
        */
    {
        cf_test<type>(b1 ? foa.av_a[idx] : (!b2 ? foa.av_a[idx] : oa.av_a[idx]), b1 ? fobc.av_b[idx] : (!b2 ? fobc.av_b[idx] : obc.av_b[idx]), b1 ? fobc.av_c[idx] : (!b2 ? fobc.av_c[idx] : obc.av_c[idx]), flag);
        obc.av_c[idx] *= LOCAL_SIZE;
    }
};

template<typename type>
struct kernel_shared
{
    static void func(tiled_index<BLOCK_SIZE> idx, sa<type, 1> &oa, sbc<type, 1> &obc, sa<type, 1> &foa, sbc<type, 1> &fobc, array_view<int, 1> &flag, int b1, int b2, int b3, int b4) __GPU_ONLY
        /*
        Test function tests pointers which point to local memory.
        idx: compute index
        oa: input, the first operand
        obc: input, the second operand
        foa: fake input a, which is used to test pointer eumlation.
        fobc: fake input b, which is used to test pointer eumlation.
        flag: control flags, which is used to test control flow.
        b1, b2, b3, b4: control flags. It's used to test pointer emulation. Make sure pointer can point to the correct values.
        */
    {
        tile_static type share_a[BLOCK_SIZE];
        share_a[idx.local[0]] = oa.av_a[idx.global];
        tile_static type share_fa[BLOCK_SIZE];
        share_fa[idx.local[0]] = foa.av_a[idx.local[0]];

        tile_static type share_b[BLOCK_SIZE];
        share_b[idx.local[0]] = obc.av_b[idx.global];
        tile_static type share_fb[BLOCK_SIZE];
        share_fb[idx.local[0]] = fobc.av_b[idx.local[0]];

        tile_static type share_c[BLOCK_SIZE];
        share_c[idx.local[0]] = obc.av_c[idx.global];
        tile_static type share_fc[BLOCK_SIZE];
        share_fc[idx.local[0]] = fobc.av_c[idx.local[0]] + 1;

        idx.barrier.wait();

        sa<type, 1> inter_oa(extent<1>(BLOCK_SIZE), !b1 ? share_a : share_fa);
        sbc<type, 1> inter_obc(extent<1>(BLOCK_SIZE), !b1 ? share_b : share_fb, !b1 ? share_c : share_fc);

        sa<type, 1> inter_foa(extent<1>(BLOCK_SIZE), b1 ? share_a : share_fa);
        sbc<type, 1> inter_fobc(extent<1>(BLOCK_SIZE), b1 ? share_b : share_fb, b1 ? share_c : share_fc);

        cf_test(!b2 ? inter_foa.av_a[idx.local] : inter_oa.av_a[idx.local], !b2 ? inter_fobc.av_b[idx.local] : inter_obc.av_b[idx.local], !b2 ? inter_fobc.av_c[idx.local] : inter_obc.av_c[idx.local], flag);

        idx.barrier.wait();

        obc.av_c[idx.global] = inter_obc.av_c[idx.local] * LOCAL_SIZE;
    }
};

template<typename type, typename k>
void RunMyKernel(vector<type> &a, vector<type> &b, vector<type> &c, vector<type> &fa, vector<type> &fb, vector<type> &fc, vector<int> &flag, accelerator_view av)
{
    extent<1> g(DOMAIN_SIZE);
    array<type, 1> a_a(g, a.begin(), av);
    array<type, 1> a_b(g, b.begin(), av);
    array<type, 1> a_c(g, c.begin(), av);
    array<type, 1> a_fa(g, fa.begin(), av);
    array<type, 1> a_fb(g, fb.begin(), av);
    array<type, 1> a_fc(g, fc.begin(), av);
    array<int, 1> a_flag(g, flag.begin(), av);

    int b1 = 0;
    int b2 = 1;
    int b3 = 3;
    int b4 = 5;

    parallel_for_each(a_a.get_extent().template tile<BLOCK_SIZE>(), [&, b1, b2, b3, b4] (tiled_index<BLOCK_SIZE> idx) __GPU_ONLY {
        sa<type, 1> oa(a_a);
        sbc<type, 1> obc(a_b, a_c);
        sa<type, 1> foa(a_fa);
        sbc<type, 1> fobc(a_fb, a_fc);
        array_view<int, 1> av_flag(a_flag);

        k::func(idx, oa, obc, foa, fobc, av_flag, b1, b2, b3, b4);
    });

    c = a_c;
}

template<typename type>
void init(vector<type> &a, vector<type> &b, vector<type> &c, vector<type> &fa, vector<type> &fb, vector<type> &fc, vector<type> &ref_c, vector<int> &flag);

template<typename type, typename k>
bool test(accelerator_view av)
{
    vector<type> a(DOMAIN_SIZE);
    vector<type> b(DOMAIN_SIZE);
    vector<type> c(DOMAIN_SIZE);
    vector<type> fa(DOMAIN_SIZE);
    vector<type> fb(DOMAIN_SIZE);
    vector<type> fc(DOMAIN_SIZE);
    vector<type> ref_c(DOMAIN_SIZE);
    vector<int> flag(DOMAIN_SIZE);

    init(a, b, c, fa, fb, fc, ref_c, flag);

    RunMyKernel<type, k>(a, b, c, fa, fb, fc, flag, av);

    bool ret = Verify(c, ref_c);

    return ret;
}

runall_result test_main()
{
    srand(2010);

    accelerator_view av = require_device_for<AMP_ELEMENT_TYPE>(Device::ALL_DEVICES).get_default_view();

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

