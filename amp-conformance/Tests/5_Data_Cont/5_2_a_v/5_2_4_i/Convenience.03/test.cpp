//--------------------------------------------------------------------------------------
// File: test.cpp
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this
// file except in compliance with the License.  You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR
// CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
//
// See the Apache Version 2.0 License for specific language governing permissions
// and limitations under the License.
//
//--------------------------------------------------------------------------------------
//
/// <tags>P1</tags>
/// <summary>Use the Rank 1-index [] () operators “call” form on a rank 1 array view.</summary>

#include "amptest.h"
#include <vector>

using namespace Concurrency;
using namespace Concurrency::Test;
using std::vector;

int main()
{
	int data[2] = {0};
	vector<int> vdata(data, &data[1]);
	array_view<int, 1> t(2, data);

	// test index operators [] and ()
	t[0]= 14;
	t(1)= 14;

	bool status = (t[0] == t(0) && t(1) == t(0) && t(0) == 14);
	Log(LogType::Info, true) << " Test 1: " << (status? "Passed": "Failed") << std::endl;

	vector<int> result(1);
	result[0] = 0;
	array<int, 1> gpustatus(1, result.begin());

	parallel_for_each(t.get_extent(), [&, t](index<1> idx) __GPU
	{
		t[idx]= 15;
		t[0]= 15;
		t(1)= 15;
		gpustatus[0] = (t[0] == t(0) && t(1) == t(0) && t(0) == 15);
	});


	result = gpustatus;
	Log(LogType::Info, true) << " Test 2: " << runall_result_name(1==result[0]? true:false) << std::endl;

	status = 1==result[0] ? status : false;

	return status ? runall_pass : runall_fail;
}

