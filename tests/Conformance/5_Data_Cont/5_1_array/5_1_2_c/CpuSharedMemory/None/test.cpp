// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.

/// <summary>Using array with CPU access type none on accelerator supporting zero-copy</summary>
// RUN: %amp_device -D__GPU__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll && mkdir -p %t
// RUN: %llc -march=c -o %t/kernel_.cl < %t.ll
// RUN: cat %opencl_math_dir/opencl_math.cl %t/kernel_.cl > %t/kernel.cl
// RUN: pushd %t && %embed_kernel kernel.cl %t/kernel.o && popd
// RUN: %cxxamp %link %t/kernel.o %s -o %t.out && %t.out

#include "../Common.h"
#include "../../../../amp.data.h"
#include "../../../../amp.compare.h"
#include "../../../../../device.h"
using namespace Concurrency;
using namespace Concurrency::Test;
const int RANK = 3;
typedef int DATA_TYPE;

int main()
{
	accelerator device = require_device_for<DATA_TYPE>();
	
	if(!device.get_supports_cpu_shared_memory())
	{
		//WLog() << "The accelerator " << device.get_description() << " does not support zero copy: Skipping" << std::endl;
        return 2;
	}
	
	extent<RANK> arr_extent = CreateRandomExtent<RANK>(256);
		std::vector<DATA_TYPE> cont(arr_extent.size(), 100);
	
		array<DATA_TYPE, RANK> arr(arr_extent, cont.begin(), device.get_default_view(), access_type_none);
	
	if(!VerifyCpuAccessType(arr, access_type_none))
	{
		return false;
	}
	
	for(int i = 0; i < 100; i++)
	{
		parallel_for_each(device.get_default_view(), arr.get_extent(), [&](index<RANK> idx) restrict(amp)
		{
			arr[idx] += 2;
		});
	}
	
    return (VerifyAllSameValue(arr, static_cast<DATA_TYPE>(300)) == -1);
}

