// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>(Negative) Create a new extent with -ve extents on GPU only</summary>

#include <amptest.h>
#include <amptest_main.h>

using namespace concurrency;

runall_result test_main()
{
	runall_result result;

    try
    {
		const int _rank = 5;
		int data[] = {1, 2, 3, 4, 0};

		extent<_rank> e1(data);
		extent<_rank> g1(e1);

		parallel_for_each (g1, [=](index<_rank> idx) restrict(amp,cpu)
		{
			g1.size();
		});
		result = runall_fail;
    }
    catch (const std::exception& e)
    {
        std::cout << "ok, Got exception as expected" << std::endl;
        result= runall_pass;
    }

	std::cout << "Testcase : " << result << std::endl;
    return result;
}

