// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.

/// <summary>Copy to a const array view</summary>

#include <amptest.h>

using namespace Concurrency;
using namespace Concurrency::Test;

int main()
{	
	accelerator cpuDevice(accelerator::cpu_accelerator);
	
	array<int, 1> srcArray(10, cpuDevice.get_default_view());
	
	array<int, 1> dataArray(10, cpuDevice.get_default_view());
	array_view<const int, 1> destArrayView(dataArray);	
	
	copy(srcArray, destArrayView);

	//We are here means test paased.
    return 1;
}

//#Expects: Error: error C2668

