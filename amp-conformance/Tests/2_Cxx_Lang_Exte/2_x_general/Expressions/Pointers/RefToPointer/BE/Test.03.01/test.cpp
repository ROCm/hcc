// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Initialize reference to pointers from two (or even more) array_view pointers (controlled by a switch variable).
/// Each pointer is initialized from two c_array of array_view. Totally there are four c_array of array_view. One c_array of array_view has real data.
/// The other three have fake data. Test control flow. More than one real the array and use part of array. </summary>

#include <amptest.h>
#include <amptest_main.h>

using std::vector;
using namespace concurrency;
using namespace concurrency::Test;

const static int DOMAIN_SIZE = 64 * 64;
const static int BLOCK_SIZE = 16;
const static int LOCAL_SIZE = 15;

template<typename type>
void init(vector<type> &a, vector<int> &b, vector<type> &fa1, vector<type> &fa2, vector<type> &fa3, vector<int> &fb1,
    vector<int> &fb2, vector<type> &refa, vector<int> &refb, vector<int> &flag)
{
    srand(2010);
    size_t size = a.size();

    Fill<type>(a, 0, size - 1);

    for (size_t i = 0; i < size; i++)
    {
        fa1[i] = fa2[i] = fa3[i] = fb1[i] = fb2[i] = a[i] - 1;
        int tmp;
        refa[i] = frexp(a[i], &tmp) * LOCAL_SIZE; // Because in kernel_local, the results have been added up. So here it needs multiplication.
        refb[i] = tmp * LOCAL_SIZE; // Because in kernel_local, the results have been added up. So here it needs multiplication.
    }

    flag[0] = 10;
    flag[1] = 12;
    flag[2] = 20;
    flag[3] = 22;
    flag[4] = 30;
    flag[5] = 32;
    flag[6] = 40;
    flag[7] = 42;
    flag[8] = 50;
    flag[9] = 52;
    flag[10] = -1;
}

template<typename type>
void cf_test(type *&rpa, int *&rpb, array_view<int, 1> &flag) __GPU_ONLY
{
    for (int i1 = flag[0]; i1 < flag[1]; i1++)
    {
        for (int i2 = flag[2]; i2 < flag[3]; i2++)
        {
            switch (flag[10])
            {
            case -1:
                {
                    for (int i3 = flag[4]; i3 < flag[5]; i3++)
                    {
                        for (int i4 = flag[6]; i4 < flag[7]; i4++)
                        {
                            for (int i5 = flag[8]; i5 < flag[9]; i5++)
                            {
                                *rpa = precise_math::frexp(*rpa, rpb);
                                return;
                            }
                        }
                    }
                    break;
                }
            default:
                break;
            }
        }
    }

    *rpa = 0; // never reach here.
}

template<typename type>
struct kernel_global
{
    static void func(tiled_index<BLOCK_SIZE> idx, array_view<type, 1> *&rpa, array_view<int, 1> *&rpb, array_view<type, 1> *&rpaf1, array_view<int, 1> *&rpbf1, array_view<int, 1> &flag, int b1, int b2, int b3, int b4) __GPU_ONLY
    {
        type *pa = b3 ? (b3 ? &(*rpa)[idx] : (--b2, &(*rpaf1)[idx])): (--b2, &(*rpaf1)[idx]);
        int *pb = b2 ? &(*rpb)[idx] : &(*rpbf1)[idx];
        type *paf1 = &(*rpaf1)[idx];
        int *pbf1 = &(*rpbf1)[idx];
        cf_test<type>(
            b1? (--b2, paf1) : (b1 ? (--b2, paf1) : pa), b2 ? pb : pbf1,
            flag);
        (*rpa)[idx] *= LOCAL_SIZE;
        (*rpb)[idx] *= LOCAL_SIZE;
    }
};

template<typename type>
struct kernel_shared
{
    static void func(tiled_index<BLOCK_SIZE> idx, array_view<type, 1> *&rpa, array_view<int, 1> *&rpb, array_view<type, 1> *&rpaf1, array_view<int, 1> *&rpbf1, array_view<int, 1> &flag, int b1, int b2, int b3, int b4) __GPU_ONLY
    {
        tile_static type share_a[BLOCK_SIZE];
        share_a[idx.local[0]] = (*rpa)[idx.global];
        tile_static int share_b[BLOCK_SIZE];
        share_b[idx.local[0]] = (*rpb)[idx.global];
        tile_static type share_af1[BLOCK_SIZE];
        share_af1[idx.local[0]] = (*rpaf1)[idx.global];
        tile_static int share_bf1[BLOCK_SIZE];
        share_bf1[idx.local[0]] = (*rpbf1)[idx.global];

        idx.barrier.wait();

        type *pa = b3 ? (b3 ? &share_a[idx.local[0]] : (--b2, &share_af1[idx.local[0]])): (--b2, &share_af1[idx.local[0]]);
        int *pb = b2 ? &share_b[idx.local[0]] : &share_bf1[idx.local[0]];
        type *paf1 = &share_af1[idx.local[0]];
        int *pbf1 = &share_bf1[idx.local[0]];

        cf_test<type>(
            b1? (--b2, paf1) : (b1 ? (--b2, paf1) : pa), b2 ? pb : pbf1,
            flag);

        idx.barrier.wait();

        (*rpa)[idx.global] = share_a[idx.local[0]] * LOCAL_SIZE;
        (*rpb)[idx.global] = share_b[idx.local[0]] * LOCAL_SIZE;
    }
};

template<typename type>
struct kernel_local
{
    static void func(tiled_index<BLOCK_SIZE> idx, array_view<type, 1> *&rpa, array_view<int, 1> *&rpb, array_view<type, 1> *&rpaf1, array_view<int, 1> *&rpbf1, array_view<int, 1> &flag, int b1, int b2, int b3, int b4) __GPU_ONLY
    {
        type local_a[LOCAL_SIZE];
        int local_b[LOCAL_SIZE];
        type local_af1[LOCAL_SIZE];
        int local_bf1[LOCAL_SIZE];

        for (int i = 0; i < LOCAL_SIZE; i++)
        {
            local_a[i] = (*rpa)[idx.global];
            local_b[i] = (*rpb)[idx.global];
            local_af1[i] = (*rpaf1)[idx.global];
            local_bf1[i] = (*rpbf1)[idx.global];
        }

        extent<1> e(LOCAL_SIZE);

        type *pa = b3 ? (b3 ? &local_a[0] : (--b2, &local_af1[0])): (--b2, &local_af1[0]);
        int *pb = b2 ? &local_b[0] : &local_bf1[0];
        type *paf1 = &local_af1[0];
        int *pbf1 = &local_bf1[0];

        for (int i = 0; i < LOCAL_SIZE; i++)
        {
            cf_test<type>(
                b1? (--b2, paf1) : (b1 ? (--b2, paf1) : pa), b2 ? pb : pbf1,
                flag);
            pa++;
            pb++;
        }

        type fa = 0;
        int fb = 0;

        for (int i = 0; i < LOCAL_SIZE; i++)
        {
            fa += local_a[i];
            fb += local_b[i];
        }

        (*rpa)[idx.global] = fa;
        (*rpb)[idx.global] = fb;
    }
};

template<typename type, typename k>
void run_mykernel(vector<type> &a, vector<int> &b, vector<type> &fa1, vector<type> &fa2, vector<type> &fa3, vector<int> &fb1, vector<int> &fb2,
    vector<int> &flag, accelerator_view av)
{
    extent<1> g(DOMAIN_SIZE);
    array_view<type, 1> a_a(g, a);
    array_view<int, 1> a_b(g, b);
    array_view<type, 1> a_fa1(g, fa1);
    array_view<type, 1> a_fa2(g, fa2);
    array_view<type, 1> a_fa3(g, fa3);
    array_view<int, 1> a_fb1(g, fb1);
    array_view<int, 1> a_fb2(g, fb2);
    array_view<int, 1> a_flag(g, flag);

    int b1 = 0;
    int b2 = 1;
    int b3 = 3;
    int b4 = 5;

    parallel_for_each(av, a_a.get_extent().template tile<BLOCK_SIZE>(), [=] (tiled_index<BLOCK_SIZE> idx) __GPU_ONLY {

        array_view<type> ara[] = {a_fa1, a_a, a_fa2, a_fa3};
        array_view<int> arb[] = {a_fb1, a_fb2, a_b, a_fb1}; // use two arrays here

        array_view<type> araf1[] = {a_fa1, a_fa2};
        array_view<type> araf2[] = {a_fa1, a_fa2};
        array_view<int> arbf1[] = {a_fb1, a_fb2};
        array_view<int> arbf2[] = {a_fb1, a_fb2};

		int local_b2 = b2;
        array_view<type> *pa = b3 ? &ara[1] : (--local_b2, &araf1[0]);
        array_view<type> *paf1 = &araf1[0];
        array_view<type> *paf2 = &araf2[1];
        array_view<int> *pb = local_b2 ? &arb[2] : &arbf1[0];
        array_view<int> *pbf1 = &arbf1[0];
        array_view<int> *pbf2 = &arbf2[1];

        array_view<int, 1> av_flag(a_flag);

        k::func(idx,
            b1? (--local_b2, paf1) : pa, local_b2 ? pb : pbf1, paf1, pbf1,
            av_flag, b1, local_b2, b3, b4);
    });

    a_a.synchronize();
    a_b.synchronize();
}

template<typename type, typename k>
bool test(accelerator_view av)
{
	static_assert(std::is_floating_point<type>::value, "test<type>: template parameter 'type' must be a floating-point type.");

    vector<type> a(DOMAIN_SIZE);
    vector<int> b(DOMAIN_SIZE);
    vector<type> fa1(DOMAIN_SIZE);
    vector<type> fa2(DOMAIN_SIZE);
    vector<type> fa3(DOMAIN_SIZE);
    vector<int> fb1(DOMAIN_SIZE);
    vector<int> fb2(DOMAIN_SIZE);
    vector<type> refa(DOMAIN_SIZE);
    vector<int> refb(DOMAIN_SIZE);
    vector<int> flag(DOMAIN_SIZE);

    init(a, b, fa1, fa2, fa3, fb1, fb2, refa, refb, flag);

    run_mykernel<type, k>(a, b, fa1, fa2, fa3, fb1, fb2, flag, av);

	// Here, we're assuming that type is a floating-point type so we get the extra range parameters.
    bool ret = REPORT_RESULT(Verify(a, refa, 0.01f, 0.01f));
	ret &= REPORT_RESULT(Verify(b, refb));

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

