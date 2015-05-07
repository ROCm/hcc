// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Initialize reference to pointers from two struct pointers (controlled by a switch variable).
/// Each struct contain multiple fields. Test control flow. </summary>

#include <amptest.h>
#include <amptest/math.h>
#include <amptest_main.h>

using std::vector;
using namespace concurrency;
using namespace concurrency::Test;

const static int DOMAIN_SIZE = 16 * 16;
const static int BLOCK_SIZE = 4;
const static int LOCAL_SIZE = 9;

struct s1
{
    float x;
    double y;
	int z;
	float w;

	s1() __GPU
	{
		x = 0;
		y = 0;
		z = 0;
		w = 0;
	}

	s1 operator+(const s1 &o) const __GPU
	{
		s1 tmp;

		tmp.x = x + o.x;
		tmp.y = y + o.y;
		tmp.z = z + o.z;
		tmp.w = w + o.w;

		return tmp;
	}

	s1 operator+(const int &i) const // No __GPU here. Otherwise, it will have int -> double, which cannot run without extended double support.
	{
		s1 tmp;

		tmp.x = x + i;
		tmp.y = y + i;
		tmp.z = z + i;
		tmp.w = w + i;

		return tmp;
	}

	s1 operator*(const int i) const // No __GPU here. Otherwise, it will have int -> double, which cannot run without extended double support.
	{
		s1 tmp;

		tmp.x = x * i;
		tmp.y = y * i;
		tmp.z = z * i;
		tmp.w = w * i;

		return tmp;
	}

	s1& operator+=(const s1 &o) __GPU
	{
		x += o.x;
		y += o.y;
		z += o.z;
		w += o.w;

		return *this;
	}

	void clear() __GPU
	{
		x = 0;
		y = 0;
		z = 0;
		w = 0;
	}

	void inc() __GPU
	{
		x++;
		y++;
		z++;
		w++;
	}
};

void FillStruct(vector<s1> &arr, int min, int max)
{
	std::mt19937 mersenne_twister_engine;
	int size = arr.size();

    std::uniform_real_distribution<float> uni_xw((float)min, (float)max);
    std::uniform_real_distribution<double> uni_y((double)min, (double)max);
    std::uniform_int_distribution<int> uni_z(min, max);

    for(int i = 0; i < size; ++i)
    {
        arr[i].x = uni_xw(mersenne_twister_engine);
        arr[i].w = uni_xw(mersenne_twister_engine);
        arr[i].y = uni_y(mersenne_twister_engine);
        arr[i].z = uni_z(mersenne_twister_engine);
    }
}

bool VerifyStruct(const std::vector< s1 > &c, const std::vector< s1 > &refc)
{
    if (c.size() != refc.size()) { return false; }

    bool passed = true;

    const int size = c.size();

	type_comparer<decltype(s1::z)> zcomparer;
    for(size_t i = 0; i < size; ++i)
    {
        float f1 = c[i].x;
        float f2 = refc[i].x;
        if (!AreAlmostEqual(f1, f2))
        {
            std::stringstream ss;
            ss.setf(std::ios::showpoint | std::ios::fixed);
            ss << "x failed " << "c[" << i << "]=" << f1 << ", refc[" << i << "]=" << f2;
            Log_writeline(LogType::Error, ss.str().c_str());

            passed = false;
            break;
        }

        double d1 = c[i].y;
        double d2 = refc[i].y;
        if (!AreAlmostEqual(d1, d2))
        {
            std::stringstream ss;
            ss.setf(std::ios::showpoint | std::ios::fixed);
            ss << "y failed " << "c[" << i << "]=" << d1 << ", refc[" << i << "]=" << d2;
            Log_writeline(LogType::Error, ss.str().c_str());

            passed = false;
            break;
        }

        if (!zcomparer.are_equal(c[i].z, refc[i].z))
        {
            std::stringstream ss;
            ss.setf(std::ios::showpoint | std::ios::fixed);
            ss << "z failed " << "c[" << i << "]=" << c[i].z << ", refc[" << i << "]=" << refc[i].z;
            Log_writeline(LogType::Error, ss.str().c_str());

            passed = false;
            break;
        }

        f1 = c[i].w;
        f2 = refc[i].w;
        if (!AreAlmostEqual(f1, f2, DEFAULT_MAX_ABS_DIFF_FLT, DEFAULT_MAX_REL_DIFF_FLT))
        {
            std::stringstream ss;
            ss.setf(std::ios::showpoint | std::ios::fixed);
            ss << "w failed " << "c[" << i << "]=" << f1 << ", refc[" << i << "]=" << f2;
            Log_writeline(LogType::Error, ss.str().c_str());

            passed = false;
            break;
        }
	}
    return passed;
}

void init(vector<s1> &a1, vector<s1> &a1f1, vector<s1> &a2, vector<s1> &a2f1, vector<s1> &a3,
    vector<s1> &a3f1, vector<s1> &ref, vector<int> &flag)
{
    srand(2010);
    size_t size = a1.size();

    FillStruct(a2, 0, size - 1);
    FillStruct(a3, 0, size - 1);

    for (size_t i = 0; i < size; i++)
    {
		a1f1[i].clear();
		a2f1[i].clear();
		a3f1[i].clear();

		ref[i] = (a2[i] + a3[i] + 6) * LOCAL_SIZE; // Because in kernel_local, the results have been added up. So here it needs multiplication.
	}

    flag[0] = 11;
    flag[1] = 12;
    flag[2] = 13;
    flag[3] = -1;
	flag[4] = 6;
    flag[5] = 9;
}

void cf_test(s1 *&rpa1, s1 *&rpa2, s1 *&rpa3, array_view<int, 1> &flag) __GPU_ONLY
{
    if (flag[0] < 10)
    {
        rpa1->clear(); // never reach here
    } else
    {
        if (flag[1] < 11)
        {
            rpa1->clear(); // never reach here
        } else {
            if (flag[2] < 12)
            {
                rpa1->clear(); // never reach here
            } else
            {
                switch (flag[3])
                {
                case 2:
                    rpa1->clear(); // never reach here
                    break;
                default:
					{
						for (int i = flag[4]; i < flag[5]; i++)
						{
							rpa2->inc();
							rpa3->inc();
						}
					}
				}
			}
        }
    }

	*rpa1 = *rpa2 + *rpa3;
}

struct kernel_global
{
	static void func(tiled_index<BLOCK_SIZE> idx, const array_view<s1, 1> *&rpa1, const array_view<s1, 1> *&rpa1f1, const array_view<s1, 1> *&rpa2, const array_view<s1, 1> *&rpa2f1,
		const array_view<s1, 1> *&rpa3, const array_view<s1, 1> *&rpa3f1, array_view<int, 1> flag, int b1, int b2, int b3, int b4) __GPU_ONLY
    {
		s1 *p1 = !b1 ? &(*rpa1)[idx] : &(*rpa1f1)[idx] ;
		s1 *p1f1 = &(*rpa1f1)[idx];
		s1 * p2 = !b2 ? &(*rpa2f1)[idx] : &(*rpa2)[idx];
		s1 * p2f1 = &(*rpa2f1)[idx];
		s1 * p3 = !b3 ? &(*rpa3f1)[idx] : &(*rpa3)[idx];
		s1 * p3f1 = &(*rpa3f1)[idx];

		cf_test(
			b1 ? p1f1 : p1,
			b2 ? p2 : p2f1,
			b3 ? p3 : p3f1,
			flag);

		(*rpa1)[idx].x *= LOCAL_SIZE;
		(*rpa1)[idx].y *= LOCAL_SIZE;
		(*rpa1)[idx].z *= LOCAL_SIZE;
		(*rpa1)[idx].w *= LOCAL_SIZE;
    }
};

struct kernel_shared
{
	static void func(tiled_index<BLOCK_SIZE> idx, const array_view<s1, 1> *&rpa1, const array_view<s1, 1> *&rpa1f1, const array_view<s1, 1> *&rpa2, const array_view<s1, 1> *&rpa2f1,
		const array_view<s1, 1> *&rpa3, const array_view<s1, 1> *&rpa3f1, array_view<int, 1> flag, int b1, int b2, int b3, int b4) __GPU_ONLY
    {
		int local_idx = idx.local[0];

		tile_static s1 share_a1[BLOCK_SIZE];
		share_a1[local_idx] = (*rpa1)[idx.global];
		tile_static s1 share_a1f1[BLOCK_SIZE];
		share_a1f1[local_idx] = (*rpa1f1)[idx.global];

		tile_static s1 share_a2[BLOCK_SIZE];
		share_a2[local_idx] = (*rpa2)[idx.global];
		tile_static s1 share_a2f1[BLOCK_SIZE];
		share_a2f1[local_idx] = (*rpa2f1)[idx.global];

		tile_static s1 share_a3[BLOCK_SIZE];
		share_a3[local_idx] = (*rpa3)[idx.global];
		tile_static s1 share_a3f1[BLOCK_SIZE];
		share_a3f1[local_idx] = (*rpa3f1)[idx.global];

		idx.barrier.wait();

		s1 *p1 = !b1 ? &share_a1[local_idx] : &share_a1f1[local_idx] ;
		s1 *p1f1 = &share_a1f1[local_idx];
		s1 *p2 = !b2 ? &share_a2f1[local_idx] : &share_a2[local_idx];
		s1 *p2f1 = &share_a2f1[local_idx];
		s1 *p3 = !b3 ? &share_a3f1[local_idx] : &share_a3[local_idx];
		s1 *p3f1 = &share_a3f1[local_idx];

		cf_test(
			b1 ? p1f1 : p1,
			b2 ? p2 : p2f1,
			b3 ? p3 : p3f1,
			flag);

		idx.barrier.wait();

		(*rpa1)[idx].x = share_a1[local_idx].x * LOCAL_SIZE;
		(*rpa1)[idx].y = share_a1[local_idx].y * LOCAL_SIZE;
		(*rpa1)[idx].z = share_a1[local_idx].z * LOCAL_SIZE;
		(*rpa1)[idx].w = share_a1[local_idx].w * LOCAL_SIZE;
    }
};

struct kernel_local
{
	static void func(tiled_index<BLOCK_SIZE> idx, const array_view<s1, 1> *&rpa1, const array_view<s1, 1> *&rpa1f1, const array_view<s1, 1> *&rpa2, const array_view<s1, 1> *&rpa2f1,
		const array_view<s1, 1> *&rpa3, const array_view<s1, 1> *&rpa3f1, array_view<int, 1> flag, int b1, int b2, int b3, int b4) __GPU_ONLY
    {
		s1 local_a1[LOCAL_SIZE];
		s1 local_a1f1[LOCAL_SIZE];

		s1 local_a2[LOCAL_SIZE];
		s1 local_a2f1[LOCAL_SIZE];

		s1 local_a3[LOCAL_SIZE];
		s1 local_a3f1[LOCAL_SIZE];

		for (int i = 0; i < LOCAL_SIZE; i++)
		{
			local_a1[i] = (*rpa1)[idx.global];
			local_a1f1[i] = (*rpa1f1)[idx.global];

			local_a2[i] = (*rpa2)[idx.global];
			local_a2f1[i] = (*rpa2f1)[idx.global];

			local_a3[i] = (*rpa3)[idx.global];
			local_a3f1[i] = (*rpa3f1)[idx.global];
		}

		for (int i = 0; i < LOCAL_SIZE; i++)
		{
			s1 *p1 = !b1 ? &local_a1[i] : &local_a1f1[i] ;
			s1 *p1f1 = &local_a1f1[i];
			s1 *p2 = !b2 ? &local_a2f1[i] : &local_a2[i];
			s1 *p2f1 = &local_a2f1[i];
			s1 *p3 = !b3 ? &local_a3f1[i] : &local_a3[i];
			s1 *p3f1 = &local_a3f1[i];

			cf_test(
				b1 ? p1f1 : p1,
				b2 ? p2 : p2f1,
				b3 ? p3 : p3f1,
				flag);
		}

		s1 t;

		for (int i = 0; i < LOCAL_SIZE; i++)
		{
			t += local_a1[i];
		}

		(*rpa1)[idx.global] = t;
    }
};

template<typename k>
void run_mykernel(vector<s1> &a1, vector<s1> &a1f1, vector<s1> &a2, vector<s1> &a2f1, vector<s1> &a3, vector<s1> &a3f1,
    vector<int> &flag, accelerator_view av)
{
    extent<1> e(DOMAIN_SIZE);

	array_view<s1> av1(e, a1);
	array_view<s1> av1f1(e, a1f1);
	array_view<s1> av2(e, a2);
	array_view<s1> av2f1(e, a2f1);
	array_view<s1> av3(e, a3);
	array_view<s1> av3f1(e, a3f1);
	array_view<int> av_flag(e, flag);

    int b1 = 0;
    int b2 = 1;
    int b3 = 3;
    int b4 = 5;

    parallel_for_each(av, av1.get_extent().tile<BLOCK_SIZE>(), [=] (tiled_index<BLOCK_SIZE> idx) __GPU_ONLY {

		const array_view<s1> *p1 = b1 ? &av1f1 : &av1;
		const array_view<s1> *p1f1 = &av1f1;

		const array_view<s1> *p2 = b2 ? &av2 : &av2f1;
		const array_view<s1> *p2f1 = &av2f1;

		const array_view<s1> *p3 = b3 ? &av3 : &av3f1;
		const array_view<s1> *p3f1 = &av3f1;
		
		k::func(idx,
			b1 ? p1f1 : p1,
			p1f1,
			b2 ? p2 : p2f1,
			p2f1,
			b3 ? p3 : p3f1,
			p3f1,
			av_flag, b1, b2, b3, b4);
    });

    av1.synchronize();
}

template<typename k>
bool test(accelerator_view av)
{
    vector<s1> a1(DOMAIN_SIZE);
    vector<s1> a1f1(DOMAIN_SIZE);
    vector<s1> a2(DOMAIN_SIZE);
    vector<s1> a2f1(DOMAIN_SIZE);
    vector<s1> a3(DOMAIN_SIZE);
    vector<s1> a3f1(DOMAIN_SIZE);
    vector<s1> ref(DOMAIN_SIZE);
    vector<int> flag(DOMAIN_SIZE);

    init(a1, a1f1, a2, a2f1, a3, a3f1, ref, flag);

    run_mykernel<k>(a1, a1f1, a2, a2f1, a3, a3f1, flag, av);

    bool ret = VerifyStruct(a1, ref);

    return ret;
}

runall_result test_main()
{
    srand(2010);
    accelerator_view av = require_device_with_double().get_default_view();;

    runall_result ret;

    ret &= REPORT_RESULT((test<kernel_global>(av)));
    ret &= REPORT_RESULT((test<kernel_shared>(av)));
    ret &= REPORT_RESULT((test<kernel_local>(av)));

    return ret;
}

