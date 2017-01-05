// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Initialize reference to pointers from two (or even more) array_views pointer which points to array_view which are members of a structure (controlled by a switch variable). Each pointer is initialized from two structures. Totally there are four structures.
/// One structure has real data. The other three have fake data. Test control flow. In control flow, all the correct data is from more than one struture. </summary>

#include <amptest.h>
#include <amptest_main.h>

#ifndef AMP_ELEMENT_TYPE
#define AMP_ELEMENT_TYPE int
#endif

using std::vector;
using namespace concurrency;
using namespace concurrency::Test;

const static int DOMAIN_SIZE_1D = 64;
const static int BLOCK_SIZE_1D = 4;
const static int LOCAL_SIZE_1D = 4;

template<typename type>
struct sab
{
    sab(array_view<type, 3> a, array_view<type, 3> b) __GPU : av_a(a), av_b(b) {}
    ~sab() __GPU {}

    int placeholder1;
    array_view<type, 3> av_a;
    array_view<type, 3> av_b;
    float placeholder2;
};

template<typename type>
struct sc
{
    sc(array_view<type, 3> c) __GPU : av_c(c){}
    ~sc() __GPU {}

    int placeholder1;
    array_view<type, 3> av_c;

    float placeholder2;
};

template<typename type>
void init(vector<type> &a, vector<type> &b,  vector<type> &c, vector<type> &fa1, vector<type> &fa2, vector<type> &fa3, vector<type> &fa4, vector<type> &refb, vector<type> &refc, vector<int> &flag)
{
    srand(2010);
    size_t size = a.size();

    Fill<type>(a, 0, size - 1);
    Fill<type>(b, 0, size - 1);
    Fill<type>(c, 0, size - 1);

    for (size_t i = 0; i < size; i++)
    {
        fa1[i] = fa2[i] = fa3[i] = fa4[i] = a[i] - 1;
        refc[i] = (std::max(a[i], b[i])) * LOCAL_SIZE_1D * LOCAL_SIZE_1D * LOCAL_SIZE_1D; // Because in kernel_local, the results have been added up. So here it needs multiplication.
        refb[i] = (b[i]) * LOCAL_SIZE_1D * LOCAL_SIZE_1D * LOCAL_SIZE_1D;
    }

    flag[0] = 0;
    flag[1] = 1;
    flag[2] = 0;
    flag[3] = 1;
    flag[4] = 0;
}

template<typename type>
void cf_test(type *&rpa, type *&rpb, type *&rpc, array_view<int, 1> &flag) __GPU_ONLY
{
    switch (flag[0])
    {
    case 0:
        {
            switch (flag[1])
            {
            case 0:
                {
                }
                break;
            default:
                {
                    switch (flag[2])
                    {
                    case 0:
                        {
                            switch (flag[3])
                            {
                            case 0:
                                {
                                }
                                break;
                            default:
                                {
                                    switch (flag[4])
                                    {
                                    case 0:
                                        {
                                            *rpc = *rpb;
                                            *rpb = Concurrency::atomic_fetch_max(rpc, *rpa);
                                            return;
                                        }
                                        break;
                                    default:
                                        {
                                        }
                                        break;
                                    }
                                }
                                break;
                            }
                        }
                        break;
                    default:
                        {
                        }
                        break;
                    }
                }
                break;
            }
        }
        break;
    default:
        {
        }
        break;
    }

    *rpb = 0; //never reach here.
}

template<typename type>
struct kernel_global
{
    static void func(tiled_index<BLOCK_SIZE_1D, BLOCK_SIZE_1D, BLOCK_SIZE_1D> idx, sab<type> *&rpab, sab<type> *&rpabf1, sab<type> *&rpabf2, sc<type> *&rpc, sc<type> *&rpcf1, sc<type> *&rpcf2, array_view<int, 1> flag, int b1, int b2, int b3, int b4) __GPU_ONLY
    {
        type *pa = (b2 >>= 1) ? &rpabf1->av_a[idx] : &rpab->av_a[idx]; // b2 is 0 now.
        type *paf1 = b3 ? &rpabf1->av_a[idx] : &rpabf2->av_a[idx];
        type *pb = (b1 |= 1) ? &rpab->av_b[idx] : &rpabf2->av_b[idx]; // b1 is true now.
        type *pbf1 = b4 ? &rpabf1->av_b[idx] : &rpabf2->av_b[idx];
        type *pc = b1 ? &rpc->av_c[idx] : &rpcf1->av_c[idx];
        type *pcf1 = b1 ? &rpcf1->av_c[idx] : &rpcf2->av_c[idx];

        cf_test<type>(b1 ? pa : paf1,
            !b2 ? pb : pbf1,
            b3 ? pc : pcf1, flag);

        rpab->av_b[idx] *= LOCAL_SIZE_1D * LOCAL_SIZE_1D * LOCAL_SIZE_1D;
        rpc->av_c[idx] *= LOCAL_SIZE_1D * LOCAL_SIZE_1D * LOCAL_SIZE_1D;
    }
};

template<typename type>
struct kernel_shared
{
    static void func(tiled_index<BLOCK_SIZE_1D, BLOCK_SIZE_1D, BLOCK_SIZE_1D> idx, sab<type> *&rpab, sab<type> *&rpabf1, sab<type> *&rpabf2, sc<type> *&rpc, sc<type> *&rpcf1, sc<type> *&rpcf2, array_view<int, 1> flag, int b1, int b2, int b3, int b4) __GPU_ONLY
    {
        tile_static type share_a[BLOCK_SIZE_1D][BLOCK_SIZE_1D][BLOCK_SIZE_1D];
        tile_static type share_b[BLOCK_SIZE_1D][BLOCK_SIZE_1D][BLOCK_SIZE_1D];
        tile_static type share_c[BLOCK_SIZE_1D][BLOCK_SIZE_1D][BLOCK_SIZE_1D];
        share_a[idx.local[0]][idx.local[1]][idx.local[2]] = rpab->av_a[idx.global];
        share_b[idx.local[0]][idx.local[1]][idx.local[2]] = rpab->av_b[idx.global];
        share_c[idx.local[0]][idx.local[1]][idx.local[2]] = rpc->av_c[idx.global];
        tile_static type share_af1[BLOCK_SIZE_1D][BLOCK_SIZE_1D][BLOCK_SIZE_1D];
        tile_static type share_bf1[BLOCK_SIZE_1D][BLOCK_SIZE_1D][BLOCK_SIZE_1D];
        tile_static type share_cf1[BLOCK_SIZE_1D][BLOCK_SIZE_1D][BLOCK_SIZE_1D];
        share_af1[idx.local[0]][idx.local[1]][idx.local[2]] = rpabf1->av_a[idx.global];
        share_bf1[idx.local[0]][idx.local[1]][idx.local[2]] = rpabf1->av_b[idx.global];
        share_cf1[idx.local[0]][idx.local[1]][idx.local[2]] = rpcf1->av_c[idx.global];
        tile_static type share_af2[BLOCK_SIZE_1D][BLOCK_SIZE_1D][BLOCK_SIZE_1D];
        tile_static type share_bf2[BLOCK_SIZE_1D][BLOCK_SIZE_1D][BLOCK_SIZE_1D];
        tile_static type share_cf2[BLOCK_SIZE_1D][BLOCK_SIZE_1D][BLOCK_SIZE_1D];
        share_af2[idx.local[0]][idx.local[1]][idx.local[2]] = rpabf2->av_a[idx.global];
        share_bf2[idx.local[0]][idx.local[1]][idx.local[2]] = rpabf2->av_b[idx.global];
        share_cf2[idx.local[0]][idx.local[1]][idx.local[2]] = rpcf2->av_c[idx.global];

        idx.barrier.wait();

        type *pa = (b2 >>= 1) ? &share_af1[idx.local[0]][idx.local[1]][idx.local[2]] : &share_a[idx.local[0]][idx.local[1]][idx.local[2]];
        type *paf1 = b3 ? &share_af1[idx.local[0]][idx.local[1]][idx.local[2]] : &share_af2[idx.local[0]][idx.local[1]][idx.local[2]];
        type *pb = (b1 |= 1) ? &share_b[idx.local[0]][idx.local[1]][idx.local[2]] : &share_bf1[idx.local[0]][idx.local[1]][idx.local[2]];
        type *pbf1 = b4 ? &share_bf1[idx.local[0]][idx.local[1]][idx.local[2]] : &share_bf2[idx.local[0]][idx.local[1]][idx.local[2]];
        type *pc = b1 ? &share_c[idx.local[0]][idx.local[1]][idx.local[2]] : &share_cf1[idx.local[0]][idx.local[1]][idx.local[2]];
        type *pcf1 = b1 ? &share_cf1[idx.local[0]][idx.local[1]][idx.local[2]] : &share_cf2[idx.local[0]][idx.local[1]][idx.local[2]];

        cf_test<type>(b1 ? pa : paf1,
            !b2 ? pb : pbf1,
            b3 ? pc : pcf1, flag);

        idx.barrier.wait();

        rpab->av_b[idx.global] = share_b[idx.local[0]][idx.local[1]][idx.local[2]] * LOCAL_SIZE_1D * LOCAL_SIZE_1D * LOCAL_SIZE_1D;
        rpc->av_c[idx.global] = share_c[idx.local[0]][idx.local[1]][idx.local[2]] * LOCAL_SIZE_1D * LOCAL_SIZE_1D * LOCAL_SIZE_1D;
    }
};

template<typename type>
struct kernel_local
{
    static void func(tiled_index<BLOCK_SIZE_1D, BLOCK_SIZE_1D, BLOCK_SIZE_1D> idx, sab<type> *&rpab, sab<type> *&rpabf1, sab<type> *&rpabf2, sc<type> *&rpc, sc<type> *&rpcf1, sc<type> *&rpcf2, array_view<int, 1> flag, int b1, int b2, int b3, int b4) __GPU_ONLY
    {
        type local_a[LOCAL_SIZE_1D][LOCAL_SIZE_1D][LOCAL_SIZE_1D];
        type local_b[LOCAL_SIZE_1D][LOCAL_SIZE_1D][LOCAL_SIZE_1D];
        type local_c[LOCAL_SIZE_1D][LOCAL_SIZE_1D][LOCAL_SIZE_1D];
        type local_af1[LOCAL_SIZE_1D][LOCAL_SIZE_1D][LOCAL_SIZE_1D];
        type local_bf1[LOCAL_SIZE_1D][LOCAL_SIZE_1D][LOCAL_SIZE_1D];
        type local_cf1[LOCAL_SIZE_1D][LOCAL_SIZE_1D][LOCAL_SIZE_1D];
        type local_af2[LOCAL_SIZE_1D][LOCAL_SIZE_1D][LOCAL_SIZE_1D];
        type local_bf2[LOCAL_SIZE_1D][LOCAL_SIZE_1D][LOCAL_SIZE_1D];
        type local_cf2[LOCAL_SIZE_1D][LOCAL_SIZE_1D][LOCAL_SIZE_1D];

        for (int i = 0; i < LOCAL_SIZE_1D; i++)
        {
            for (int j = 0; j < LOCAL_SIZE_1D; j++)
            {
                for (int k = 0; k < LOCAL_SIZE_1D; k++)
                {
                    local_a[i][j][k] = rpab->av_a[idx.global];
                    local_b[i][j][k] = rpab->av_b[idx.global];
                    local_c[i][j][k] = rpc->av_c[idx.global];
                }
            }
        }

        type *pa = (b2 >>= 1) ? &local_af1[0][0][0] : &local_a[0][0][0];
        type *paf1 = b3 ? &local_af1[0][0][0] : &local_af2[0][0][0];
        type *pb = (b1 |= 1) ? &local_b[0][0][0] : &local_bf1[0][0][0];
        type *pbf1 = b4 ? &local_bf1[0][0][0] : &local_bf2[0][0][0];
        type *pc = b1 ? &local_c[0][0][0] : &local_cf1[0][0][0];
        type *pcf1 = b1 ? &local_cf1[0][0][0] : &local_cf2[0][0][0];

        for (int i = 0; i < LOCAL_SIZE_1D * LOCAL_SIZE_1D * LOCAL_SIZE_1D; i++)
        {
            cf_test<type>(b1 ? pa : paf1,
                !b2 ? pb : pbf1,
                b3 ? pc : pcf1, flag);
            pa++;
            pb++;
            pc++;
        }

        type fb = 0;
        type fc = 0;

        for (int i = 0; i < LOCAL_SIZE_1D; i++)
        {
            for (int j = 0; j < LOCAL_SIZE_1D; j++)
            {
                for (int k = 0; k < LOCAL_SIZE_1D; k++)
                {
                    fb += local_b[i][j][k];
                    fc += local_c[i][j][k];
                }
            }
        }

        rpab->av_b[idx.global] = fb;
        rpc->av_c[idx.global] = fc;
    }
};

template<typename type, typename k>
void run_mykernel(vector<type> &a, vector<type> &b, vector<type> &c, vector<type> &fa1, vector<type> &fa2, vector<type> &fa3, vector<type> &fa4, vector<int> &flag, accelerator_view av)
{
    extent<3> g(DOMAIN_SIZE_1D, DOMAIN_SIZE_1D, DOMAIN_SIZE_1D);
    array_view<type, 3> a_a(g, a);
    array_view<type, 3> a_b(g, b);
    array_view<type, 3> a_c(g, c);
    array_view<type, 3> a_fa1(g, fa1);
    array_view<type, 3> a_fa2(g, fa2);
    array_view<type, 3> a_fa3(g, fa3);
    array_view<type, 3> a_fa4(g, fa4);
    extent<1> e_flag(DOMAIN_SIZE_1D);
    array_view<int, 1> a_flag(e_flag, flag);

    int b1 = 0;
    int b2 = 1;
    int b3 = 3;
    int b4 = 5;

    parallel_for_each(av, a_a.get_extent().template tile<BLOCK_SIZE_1D, BLOCK_SIZE_1D, BLOCK_SIZE_1D>(), [=] (tiled_index<BLOCK_SIZE_1D, BLOCK_SIZE_1D, BLOCK_SIZE_1D> idx) __GPU_ONLY {

        sab<type> oab(a_a, a_b);
        sc<type> oc(a_c);

        sab<type> oabf1(a_fa1, a_fa2);
        sc<type> ocf1(a_fa3);

        sab<type> oabf2(a_fa2, a_fa3);
        sc<type> ocf2(a_fa4);

		int local_b2 = b2;
        sab<type> *pab = b1 ? (--local_b2, &oabf1) : &oab; // local_b2 is still 0.
        sc<type> *pc = b1 ? (--local_b2, &ocf1) :  &oc; // local_b2 is still 0.

        sab<type> *pabf1 = b3 ? &oabf1 : &oabf2;
        sc<type> *pcf1 =  b3 ? &ocf1 : &ocf2;

        sab<type> *pabf2 = b4 ? &oabf1 : &oabf2;
        sc<type> *pcf2 = b4 ? &ocf1 : &ocf2;

        k::func(idx, (b3 % local_b2) ?  pabf1 : pab , pabf1, pabf2, local_b2 ? pc : pcf1, pcf1, pcf2, a_flag, b1, local_b2, b3, b4);
    });

    a_a.synchronize();
}

template<typename type, typename k>
bool test(accelerator_view av)
{
    vector<type> a(DOMAIN_SIZE_1D * DOMAIN_SIZE_1D * DOMAIN_SIZE_1D);
    vector<type> b(DOMAIN_SIZE_1D * DOMAIN_SIZE_1D * DOMAIN_SIZE_1D);
    vector<type> c(DOMAIN_SIZE_1D * DOMAIN_SIZE_1D * DOMAIN_SIZE_1D);
    vector<type> fa1(DOMAIN_SIZE_1D * DOMAIN_SIZE_1D * DOMAIN_SIZE_1D);
    vector<type> fa2(DOMAIN_SIZE_1D * DOMAIN_SIZE_1D * DOMAIN_SIZE_1D);
    vector<type> fa3(DOMAIN_SIZE_1D * DOMAIN_SIZE_1D * DOMAIN_SIZE_1D);
    vector<type> fa4(DOMAIN_SIZE_1D * DOMAIN_SIZE_1D * DOMAIN_SIZE_1D);
    vector<type> refb(DOMAIN_SIZE_1D * DOMAIN_SIZE_1D * DOMAIN_SIZE_1D);
    vector<type> refc(DOMAIN_SIZE_1D * DOMAIN_SIZE_1D * DOMAIN_SIZE_1D);
    vector<int> flag(DOMAIN_SIZE_1D * DOMAIN_SIZE_1D * DOMAIN_SIZE_1D);

    init(a, b, c, fa1, fa2, fa3, fa4, refb, refc, flag);

    run_mykernel<type, k>(a, b, c, fa1, fa2, fa3, fa4, flag, av);

    bool ret = Verify(b, refb) && Verify(c, refc);

    return ret;
}

runall_result test_main()
{
    srand(2010);

    accelerator_view av = require_device_for<AMP_ELEMENT_TYPE>(device_flags::NOT_SPECIFIED, false).get_default_view();

    runall_result ret;

    ret &= REPORT_RESULT((test<AMP_ELEMENT_TYPE, kernel_global<AMP_ELEMENT_TYPE>>(av)));
    ret &= REPORT_RESULT((test<AMP_ELEMENT_TYPE, kernel_shared<AMP_ELEMENT_TYPE>>(av)));
    ret &= REPORT_RESULT((test<AMP_ELEMENT_TYPE, kernel_local<AMP_ELEMENT_TYPE>>(av)));

    return ret;
}

