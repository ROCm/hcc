// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.

/// <summary>Copying between source and destination with different rank (array view to array view)</summary>

#include <amptest.h>

using namespace Concurrency;
using namespace Concurrency::Test;

int main()
{	
	accelerator cpuDevice(accelerator::cpu_accelerator);
	
	std::vector<int> stdContSrc(10);
	array_view<int, 1> srcArrayView(10, stdContSrc);
	
	std::vector<int> stdContDest(100);
	array_view<int, 2> destArrayView(10, 10, stdContDest);

	copy(srcArrayView, destArrayView);
	
	//We are here means test falied.
	return runall_fail;
}

//#Expects: Error: error C2668
