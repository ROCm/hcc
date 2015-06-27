// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Access tile_static variable from a local class.</summary>
#include <amptest.h>
#include <amptest_main.h>
using namespace concurrency;
using namespace concurrency::Test;

runall_result test_main()
{
	std::vector<int> res_(2);
	array_view<int> res(res_.size(), res_);
	parallel_for_each(extent<1>(1).tile<1>(), [=](tiled_index<1>) restrict(amp)
	{
		tile_static int ts_i;
		ts_i = 0;
		struct obj
		{
			void f() { ts_i = 1; } // Store
			int g() { return ts_i + 1; } // Load
		};
		obj().f();
		res[0] = ts_i;
		res[1] = obj().g();
	});
	
	runall_result result;
	result &= REPORT_RESULT(res[0] == 1);
	result &= REPORT_RESULT(res[1] == 2);
	return result;
}
