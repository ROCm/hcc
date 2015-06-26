// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.

/// <summary>Copy with data containers of rank greater than 3</summary>

#include "./../../CopyTestFlow.h"
#include <amptest_main.h>
#include <deque>

using namespace Concurrency;
using namespace Concurrency::Test;

runall_result test_main()
{	
	accelerator_view cpu_av = accelerator(accelerator::cpu_accelerator).get_default_view();
	accelerator_view gpu_av = require_device(Device::ALL_DEVICES).get_default_view();
	
	runall_result res;
	res &= CopyAndVerifyFromArrayToArray<int, 4>(cpu_av, gpu_av, access_type_auto, access_type_auto, access_type_auto);
	res &= CopyAndVerifyFromArrayToArray<long, 6>(gpu_av, gpu_av, access_type_auto, access_type_auto, access_type_auto);
	res &= CopyAndVerifyFromArrayViewToArray<unsigned long, 7>(cpu_av, gpu_av, access_type_auto, access_type_auto, access_type_auto);
	res &= CopyAndVerifyBetweenArrayAndIterator<float, 8, std::deque>(gpu_av, access_type_auto);
	res &= CopyAndVerifyFromArrayToArrayView<unsigned int, 5>(cpu_av, gpu_av, access_type_auto, access_type_auto, access_type_auto);	
	
	return res;
}

