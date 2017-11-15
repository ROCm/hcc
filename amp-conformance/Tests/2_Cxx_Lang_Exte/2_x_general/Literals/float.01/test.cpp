// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Floating literals and special values, float</summary>
#include <amptest.h>
#include <amptest_main.h>
#include <cstdint>
using namespace concurrency;
using namespace concurrency::Test;

#define DEOPT(x) x = (x + 1.0f) * 2.0f
#define STORE(n,x) res_ ## n[i++] = x;

union float_int
{
	float f;
	std::int32_t i;
};

bool compare_ulp(float a, float b, std::uint32_t max_ulp_diff)
{
	float_int fi_a, fi_b;
	fi_a.f = a;
	fi_b.f = b;
	return static_cast<std::uint32_t>(std::abs(fi_a.i - fi_b.i)) <= max_ulp_diff;
}

runall_result test_main()
{
	accelerator_view av = require_device(device_flags::NOT_SPECIFIED).get_default_view();

	std::vector<float> res_0_(5);
	std::vector<float> res_1_(14);
	std::vector<float> res_2_(3);
	std::vector<float> res_3_(20);
	std::vector<float> res_4_(8);
	std::vector<float> res_5_(4);
	array_view<float> res_0(res_0_.size(), res_0_);
	array_view<float> res_1(res_1_.size(), res_1_);
	array_view<float> res_2(res_2_.size(), res_2_);
	array_view<float> res_3(res_3_.size(), res_3_);
	array_view<float> res_4(res_4_.size(), res_4_);
	array_view<float> res_5(res_5_.size(), res_5_);

	parallel_for_each(av, extent<1>(1), [=](index<1> idx) restrict(amp)
	{
		// Exact binary32 values
		float f000 = .125f;
		float f001 = .0000016689300537109375f;
		float f002 = 1920.f;
		float f003 = 1813388729421943762059264.f;
		float f004 = 388.9375f;

		// Special values
		float f100 = 3.402823466e+38f; // largest
		float f101 = 2.00000000f; // exponent = 100...0, fraction = 00...0
		float f102 = 1.99999988f; // exponent = 011...1, fraction = 11...1
		float f103 = 1.17549435e-38f; // smallest normalized
		float f104 = 1.17549429e-38f; // largest denormalized
		float f105 = 1.40129846e-45f; // smallest denormalized
		float f106 = 0.0f; // positive zero
		float f107 = -0.0f;
		float f108 = -1.40129846e-45f;
		float f109 = -1.17549429e-38f;
		float f110 = -1.17549435e-38f;
		float f111 = -1.99999988f;
		float f112 = -2.00000000f;
		float f113 = -3.402823466e+38f;

		// Long literals
		float f200 = .3333333333333333333333333333333333333333333333333333333333333333333333333333333f;
		float f201 = 100000000000000000000000000000000000000.f;
		float f202 = 100000000000000000000000000000000000000.0000000000000000000000000000000000000001f;

		// Different syntax, same value
		float f300 = 12340000.f;
		float f301 = 12340000.F;
		float f302 = 12340000.0f;
		float f303 = 12340000.0F;
		float f304 = 123.4e5f;
		float f305 = 123.4e5F;
		float f306 = 123.4e+5f;
		float f307 = 123.4e+5F;
		float f308 = 123.4E5f;
		float f309 = 123.4E5F;
		float f310 = 123.4E+5f;
		float f311 = 123.4E+5F;
		float f312 = 1234e4f;
		float f313 = 1234e4F;
		float f314 = 1234e+4f;
		float f315 = 1234e+4F;
		float f316 = 1234E4f;
		float f317 = 1234E4F;
		float f318 = 1234E+4f;
		float f319 = 1234E+4F;

		float f400 = 123.4e-5f;
		float f401 = 123.4e-5F;
		float f402 = 123.4E-5f;
		float f403 = 123.4E-5F;
		float f404 = 1234e-6f;
		float f405 = 1234e-6F;
		float f406 = 1234E-6f;
		float f407 = 1234E-6F;

		// Special values not representable with literals
		float_int fi;
		fi.i = 0x7F800000; // infinity
		float f500 = fi.f;
		fi.i = 0xFF800000; // -infinity
		float f501 = fi.f;
		fi.i = 0x7FC00000; // QNaN
		float f502 = fi.f;
		fi.i = 0x7F800001; // SNaN
		float f503 = fi.f;

		// Prevent optimizations
		if(idx[0] == 1)
		{
			DEOPT(f000);DEOPT(f001);DEOPT(f002);DEOPT(f003);DEOPT(f004);

			DEOPT(f100);DEOPT(f101);DEOPT(f102);DEOPT(f103);DEOPT(f104);
			DEOPT(f105);DEOPT(f106);DEOPT(f107);DEOPT(f108);DEOPT(f109);
			DEOPT(f110);DEOPT(f111);DEOPT(f112);DEOPT(f113);

			DEOPT(f200);DEOPT(f201);DEOPT(f202);

			DEOPT(f300);DEOPT(f301);DEOPT(f302);DEOPT(f303);DEOPT(f304);
			DEOPT(f305);DEOPT(f306);DEOPT(f307);DEOPT(f308);DEOPT(f309);
			DEOPT(f310);DEOPT(f311);DEOPT(f312);DEOPT(f313);DEOPT(f314);
			DEOPT(f315);DEOPT(f316);DEOPT(f317);DEOPT(f318);DEOPT(f319);

			DEOPT(f400);DEOPT(f401);DEOPT(f402);DEOPT(f403);DEOPT(f404);
			DEOPT(f405);DEOPT(f406);DEOPT(f407);

			DEOPT(f500);DEOPT(f501);DEOPT(f502);DEOPT(f503);
		}

		// Store results
		unsigned i;

		// res_0 := f0xx
		i = 0;
		STORE(0, f000);STORE(0, f001);STORE(0, f002);STORE(0, f003);STORE(0, f004);

		// res_1 := f1xx
		i = 0;
		STORE(1, f100);STORE(1, f101);STORE(1, f102);STORE(1, f103);STORE(1, f104);
		STORE(1, f105);STORE(1, f106);STORE(1, f107);STORE(1, f108);STORE(1, f109);
		STORE(1, f110);STORE(1, f111);STORE(1, f112);STORE(1, f113);

		// res_2 := f2xx
		i = 0;
		STORE(2, f200);STORE(2, f201);STORE(2, f202);

		// res_3 := f3xx
		i = 0;
		STORE(3, f300);STORE(3, f301);STORE(3, f302);STORE(3, f303);STORE(3, f304);
		STORE(3, f305);STORE(3, f306);STORE(3, f307);STORE(3, f308);STORE(3, f309);
		STORE(3, f310);STORE(3, f311);STORE(3, f312);STORE(3, f313);STORE(3, f314);
		STORE(3, f315);STORE(3, f316);STORE(3, f317);STORE(3, f318);STORE(3, f319);

		// res_4 := f4xx
		i = 0;
		STORE(4, f400);STORE(4, f401);STORE(4, f402);STORE(4, f403);STORE(4, f404);
		STORE(4, f405);STORE(4, f406);STORE(4, f407);

		// res_5 := f5xx
		i = 0;
		STORE(5, f500);STORE(5, f501);STORE(5, f502);STORE(5, f503);
	});

	runall_result result;
	float_int fi;

	// Exact binary32 values
	result &= REPORT_RESULT(res_0[0] == .125f);
	result &= REPORT_RESULT(res_0[1] == .0000016689300537109375f);
	result &= REPORT_RESULT(res_0[2] == 1920.f);
	result &= REPORT_RESULT(res_0[3] == 1813388729421943762059264.f);
	result &= REPORT_RESULT(res_0[4] == 388.9375f);

	// Special values
	result &= REPORT_RESULT(res_1[ 0] == 3.402823466e+38f);
	result &= REPORT_RESULT(res_1[ 1] == 2.00000000f);
	result &= REPORT_RESULT(res_1[ 2] == 1.99999988f);
	result &= REPORT_RESULT(res_1[ 3] == 1.17549435e-38f);
	result &= REPORT_RESULT(res_1[ 4] == 1.17549429e-38f);
	result &= REPORT_RESULT(res_1[ 5] == 1.40129846e-45f);
	result &= REPORT_RESULT(res_1[ 6] == 0.0f);
	fi.f = res_1[6];
	result &= REPORT_RESULT((fi.i & 0x80000000) == 0); // positive
	result &= REPORT_RESULT(res_1[ 7] == -0.0f);
	fi.f = res_1[7];
	result &= REPORT_RESULT((fi.i & 0x80000000) != 0); // negative
	result &= REPORT_RESULT(res_1[ 8] == -1.40129846e-45f);
	result &= REPORT_RESULT(res_1[ 9] == -1.17549429e-38f);
	result &= REPORT_RESULT(res_1[10] == -1.17549435e-38f);
	result &= REPORT_RESULT(res_1[11] == -1.99999988f);
	result &= REPORT_RESULT(res_1[12] == -2.00000000f);
	result &= REPORT_RESULT(res_1[13] == -3.402823466e+38f);

	// Long literals
	result &= REPORT_RESULT(compare_ulp(res_2[0], .33333333f, 1));
	result &= REPORT_RESULT(compare_ulp(res_2[1], 1e38f, 1));
	result &= REPORT_RESULT(compare_ulp(res_2[2], 1e38f, 1));

	// Different syntax, same value
	result &= REPORT_RESULT(compare_ulp(res_3[ 0], 1234e4f, 1));
	result &= REPORT_RESULT(res_3[0] == res_3[ 1]);
	result &= REPORT_RESULT(res_3[0] == res_3[ 2]);
	result &= REPORT_RESULT(res_3[0] == res_3[ 3]);
	result &= REPORT_RESULT(res_3[0] == res_3[ 4]);
	result &= REPORT_RESULT(res_3[0] == res_3[ 5]);
	result &= REPORT_RESULT(res_3[0] == res_3[ 6]);
	result &= REPORT_RESULT(res_3[0] == res_3[ 7]);
	result &= REPORT_RESULT(res_3[0] == res_3[ 8]);
	result &= REPORT_RESULT(res_3[0] == res_3[ 9]);
	result &= REPORT_RESULT(res_3[0] == res_3[10]);
	result &= REPORT_RESULT(res_3[0] == res_3[11]);
	result &= REPORT_RESULT(res_3[0] == res_3[12]);
	result &= REPORT_RESULT(res_3[0] == res_3[13]);
	result &= REPORT_RESULT(res_3[0] == res_3[14]);
	result &= REPORT_RESULT(res_3[0] == res_3[15]);
	result &= REPORT_RESULT(res_3[0] == res_3[16]);
	result &= REPORT_RESULT(res_3[0] == res_3[17]);
	result &= REPORT_RESULT(res_3[0] == res_3[18]);
	result &= REPORT_RESULT(res_3[0] == res_3[19]);

	result &= REPORT_RESULT(compare_ulp(res_4[ 0], 1234e-6f, 1));
	result &= REPORT_RESULT(res_4[0] == res_4[ 1]);
	result &= REPORT_RESULT(res_4[0] == res_4[ 2]);
	result &= REPORT_RESULT(res_4[0] == res_4[ 3]);
	result &= REPORT_RESULT(res_4[0] == res_4[ 4]);
	result &= REPORT_RESULT(res_4[0] == res_4[ 5]);
	result &= REPORT_RESULT(res_4[0] == res_4[ 6]);
	result &= REPORT_RESULT(res_4[0] == res_4[ 7]);

	// Special values not representable with literals
	fi.f = res_5[0];
	result &= REPORT_RESULT(fi.i == 0x7F800000);
	fi.f = res_5[1];
	result &= REPORT_RESULT(fi.i == 0xFF800000);
	fi.f = res_5[2];
	result &= REPORT_RESULT(fi.i == 0x7FC00000);
	fi.f = res_5[3];
	result &= REPORT_RESULT(fi.i == 0x7F800001);

	return result;
}
