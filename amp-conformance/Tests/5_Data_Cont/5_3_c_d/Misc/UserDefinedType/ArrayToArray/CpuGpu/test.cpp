// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.

/// <summary>Copy involving userdefined type</summary>

#include <amptest_main.h>
#include "./../../TestMethod.h"

using namespace Concurrency;
using namespace Concurrency::Test;

runall_result test_main()
{	
	accelerator cpuDevice(accelerator::cpu_accelerator);
	accelerator gpuDevice = require_device(Device::ALL_DEVICES);
	
	if(!CopyBetweenArrayAndArray(cpuDevice, gpuDevice)) { return runall_fail; }
	
	//We are here means test paased.
    return 0;
}
