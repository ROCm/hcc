// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Test that passing an index with a smaller rank than the extent rank to contains results in a compilation error.</summary>
//#Expects: Error: test.cpp\(23\) : error C2664
//#Expects: Error: test.cpp\(28\) : error C2664

#include <amptest.h>

using namespace concurrency;
using namespace concurrency::Test;

int test() restrict(cpu,amp)
{
	{
		extent<3> ext;
		index<2> idx;
		ext.contains(idx);
	}
	{
		extent<10> ext;
		index<3> idx;
		ext.contains(idx);
	}

    return runall_pass;
}

int main(int argc, char **argv)
{
	accelerator_view av = require_device(device_flags::NOT_SPECIFIED).get_default_view();

	int result = test();
	Log(LogType::Info, true) << "Test " << runall_result_name(result) << " on host\n";
	if(result != runall_pass) return result;

	result = GPU_INVOKE(av, int, test);
	Log(LogType::Info, true) << "Test " << runall_result_name(result) << " on device\n";
	return result;
}

