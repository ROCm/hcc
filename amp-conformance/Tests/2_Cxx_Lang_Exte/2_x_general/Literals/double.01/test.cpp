// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Floating literals and special values, double</summary>
#include <amptest.h>
#include <amptest_main.h>
#include <cstdint>
using namespace concurrency;
using namespace concurrency::Test;

#define DEOPT(x) x = (x + 1.0) * 2.0
#define STORE(n,x) res_ ## n[i++] = x;

union double_int
{
	double d;
	std::int64_t i;
};

union double_int_amp
{
	double d;
	struct // Little endian
	{
		std::int32_t low;
		std::int32_t high;
	} i;
};

bool compare_ulp(double a, double b, std::uint64_t max_ulp_diff)
{
	double_int di_a, di_b;
	di_a.d = a;
	di_b.d = b;
	return static_cast<std::uint64_t>(std::abs(di_a.i - di_b.i)) <= max_ulp_diff;
}

runall_result test_main()
{
	accelerator_view av = require_device_for<double>(device_flags::NOT_SPECIFIED, false).get_default_view();

	std::vector<double> res_0_(5);
	std::vector<double> res_1_(14);
	std::vector<double> res_2_(3);
	std::vector<double> res_3_(10);
	std::vector<double> res_4_(4);
	std::vector<double> res_5_(4);
	array_view<double> res_0(res_0_.size(), res_0_);
	array_view<double> res_1(res_1_.size(), res_1_);
	array_view<double> res_2(res_2_.size(), res_2_);
	array_view<double> res_3(res_3_.size(), res_3_);
	array_view<double> res_4(res_4_.size(), res_4_);
	array_view<double> res_5(res_5_.size(), res_5_);

	parallel_for_each(av, extent<1>(1), [=](index<1> idx) restrict(amp)
	{
		// Exact binary64 values
		double d000 = .125;
		double d001 = .0000016689300537109375;
		double d002 = 1920.;
		double d003 = 1813388729421943762059264.;
		double d004 = 388.9375;

		// Special values
		double d100 = 1.7976931348623158e+308; // largest
		double d101 = 2.0000000000000000; // exponent = 100...0, fraction = 00...0
		double d102 = 1.9999999999999998; // exponent = 011...1, fraction = 11...1
		double d103 = 2.2250738585072013e-308; // smallest normalized
		double d104 = 2.2250738585072008e-308; // largest denormalized
		double d105 = 4.94065645841246544e-324; // smallest denormalized
		double d106 = 0.0; // positive zero
		double d107 = -0.0;
		double d108 = -4.94065645841246544e-324;
		double d109 = -2.2250738585072008e-308;
		double d110 = -2.2250738585072013e-308;
		double d111 = -1.9999999999999998;
		double d112 = -2.0000000000000000;
		double d113 = -1.7976931348623158e+308;

		// Long literals
		double d200 = .3333333333333333333333333333333333333333333333333333333333333333333333333333333;
		double d201 = 100000000000000000000000000000000000000.;
		double d202 = 100000000000000000000000000000000000000.0000000000000000000000000000000000000001;

		// Different syntax, same value
		double d300 = 12340000.;
		double d301 = 12340000.0;
		double d302 = 123.4e5;
		double d303 = 123.4e+5;
		double d304 = 123.4E5;
		double d305 = 123.4E+5;
		double d306 = 1234e4;
		double d307 = 1234e+4;
		double d308 = 1234E4;
		double d309 = 1234E+4;

		double d400 = 123.4e-5;
		double d401 = 123.4E-5;
		double d402 = 1234e-6;
		double d403 = 1234E-6;

		// Special values not representable with literals
		double_int_amp di;
		di.i.high = 0x7FF00000; // infinity
		di.i.low  = 0x00000000;
		double d500 = di.d;
		di.i.high = 0xFFF00000; // -infinity
		di.i.low  = 0x00000000;
		double d501 = di.d;
		di.i.high = 0x7FF80000; // QNaN
		di.i.low  = 0x00000000;
		double d502 = di.d;
		di.i.high = 0x7FF00000; // SNaN
		di.i.low  = 0x00000001;
		double d503 = di.d;

		// Prevent optimizations
		if(idx[0] == 1)
		{
			DEOPT(d000);DEOPT(d001);DEOPT(d002);DEOPT(d003);DEOPT(d004);

			DEOPT(d100);DEOPT(d101);DEOPT(d102);DEOPT(d103);DEOPT(d104);
			DEOPT(d105);DEOPT(d106);DEOPT(d107);DEOPT(d108);DEOPT(d109);
			DEOPT(d110);DEOPT(d111);DEOPT(d112);DEOPT(d113);

			DEOPT(d200);DEOPT(d201);DEOPT(d202);

			DEOPT(d300);DEOPT(d301);DEOPT(d302);DEOPT(d303);DEOPT(d304);
			DEOPT(d305);DEOPT(d306);DEOPT(d307);DEOPT(d308);DEOPT(d309);

			DEOPT(d400);DEOPT(d401);DEOPT(d402);DEOPT(d403);

			DEOPT(d500);DEOPT(d501);DEOPT(d502);DEOPT(d503);
		}

		// Store results
		unsigned i;

		// res_0 := d0xx
		i = 0;
		STORE(0, d000);STORE(0, d001);STORE(0, d002);STORE(0, d003);STORE(0, d004);

		// res_1 := d1xx
		i = 0;
		STORE(1, d100);STORE(1, d101);STORE(1, d102);STORE(1, d103);STORE(1, d104);
		STORE(1, d105);STORE(1, d106);STORE(1, d107);STORE(1, d108);STORE(1, d109);
		STORE(1, d110);STORE(1, d111);STORE(1, d112);STORE(1, d113);

		// res_2 := d2xx
		i = 0;
		STORE(2, d200);STORE(2, d201);STORE(2, d202);

		// res_3 := d3xx
		i = 0;
		STORE(3, d300);STORE(3, d301);STORE(3, d302);STORE(3, d303);STORE(3, d304);
		STORE(3, d305);STORE(3, d306);STORE(3, d307);STORE(3, d308);STORE(3, d309);

		// res_4 := d4xx
		i = 0;
		STORE(4, d400);STORE(4, d401);STORE(4, d402);STORE(4, d403);

		// res_5 := d5xx
		i = 0;
		STORE(5, d500);STORE(5, d501);STORE(5, d502);STORE(5, d503);
	});

	runall_result result;
	double_int di;

	// Exact binary64 values
	result &= REPORT_RESULT(res_0[0] == .125);
	result &= REPORT_RESULT(res_0[1] == .0000016689300537109375);
	result &= REPORT_RESULT(res_0[2] == 1920.);
	result &= REPORT_RESULT(res_0[3] == 1813388729421943762059264.);
	result &= REPORT_RESULT(res_0[4] == 388.9375);

	// Special values
	result &= REPORT_RESULT(res_1[ 0] == 1.7976931348623158e+308);
	result &= REPORT_RESULT(res_1[ 1] == 2.0000000000000000);
	result &= REPORT_RESULT(res_1[ 2] == 1.9999999999999998);
	result &= REPORT_RESULT(res_1[ 3] == 2.2250738585072013e-308);
	result &= REPORT_RESULT(res_1[ 4] == 2.2250738585072008e-308);
	result &= REPORT_RESULT(res_1[ 5] == 4.94065645841246544e-324);
	result &= REPORT_RESULT(res_1[ 6] == 0.0);
	di.d = res_1[6];
	result &= REPORT_RESULT((di.i & 0x8000000000000000) == 0); // positive
	result &= REPORT_RESULT(res_1[ 7] == -0.0);
	di.d = res_1[7];
	result &= REPORT_RESULT((di.i & 0x8000000000000000) != 0); // negative
	result &= REPORT_RESULT(res_1[ 8] == -4.94065645841246544e-324);
	result &= REPORT_RESULT(res_1[ 9] == -2.2250738585072008e-308);
	result &= REPORT_RESULT(res_1[10] == -2.2250738585072013e-308);
	result &= REPORT_RESULT(res_1[11] == -1.9999999999999998);
	result &= REPORT_RESULT(res_1[12] == -2.0000000000000000);
	result &= REPORT_RESULT(res_1[13] == -1.7976931348623158e+308);

	// Long literals
	result &= REPORT_RESULT(compare_ulp(res_2[0], 0.33333333333333333, 1));
	result &= REPORT_RESULT(compare_ulp(res_2[1], 1e38, 1));
	result &= REPORT_RESULT(compare_ulp(res_2[2], 1e38, 1));

	// Different syntax, same value
	result &= REPORT_RESULT(compare_ulp(res_3[ 0], 1234e4, 1));
	result &= REPORT_RESULT(res_3[0] == res_3[ 1]);
	result &= REPORT_RESULT(res_3[0] == res_3[ 2]);
	result &= REPORT_RESULT(res_3[0] == res_3[ 3]);
	result &= REPORT_RESULT(res_3[0] == res_3[ 4]);
	result &= REPORT_RESULT(res_3[0] == res_3[ 5]);
	result &= REPORT_RESULT(res_3[0] == res_3[ 6]);
	result &= REPORT_RESULT(res_3[0] == res_3[ 7]);
	result &= REPORT_RESULT(res_3[0] == res_3[ 8]);
	result &= REPORT_RESULT(res_3[0] == res_3[ 9]);

	result &= REPORT_RESULT(compare_ulp(res_4[ 0], 1234e-6, 1));
	result &= REPORT_RESULT(res_4[0] == res_4[ 1]);
	result &= REPORT_RESULT(res_4[0] == res_4[ 2]);
	result &= REPORT_RESULT(res_4[0] == res_4[ 3]);

	// Special values not representable with literals
	di.d = res_5[0];
	result &= REPORT_RESULT(di.i == 0x7FF0000000000000);
	di.d = res_5[1];
	result &= REPORT_RESULT(di.i == 0xFFF0000000000000);
	di.d = res_5[2];
	result &= REPORT_RESULT(di.i == 0x7FF8000000000000);
	di.d = res_5[3];
	result &= REPORT_RESULT(di.i == 0x7FF0000000000001);

	return result;
}
