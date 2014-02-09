// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.

/// <summary>Using array with CPU access type read/write on accelerator supporting zero-copy</summary>
// RUN: %amp_device -D__GPU__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll && mkdir -p %t
// RUN: %llc -march=c -o %t/kernel_.cl < %t.ll
// RUN: cat %opencl_math_dir/opencl_math.cl %t/kernel_.cl > %t/kernel.cl
// RUN: pushd %t && %embed_kernel kernel.cl %t/kernel.o && popd
// RUN: %cxxamp %link %t/kernel.o %s -o %t.out && %t.out
#include "../Common.h"
#include "../../../../amp.data.h"
#include "../../../../../device.h"
static const int runall_skip   =   2;
static const int runall_fail   =   1;
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
		array<DATA_TYPE, RANK> arr(arr_extent, device.get_default_view(), access_type_read_write);
	
	if(!VerifyCpuAccessType(arr, access_type_read_write)) { return 1; }
	
	Write<DATA_TYPE, RANK>(arr, 100);
	if(!ReadAndVerify<DATA_TYPE, RANK>(arr, 100)) { return 1; };
	
	for(int i = 1; i <= 100; i++)
	{
		parallel_for_each(device.get_default_view(), arr.get_extent(), [&](index<RANK> idx) restrict(amp)
		{
			arr[idx] += 1;
		});
		
		Increment<DATA_TYPE, RANK>(arr, 1);
	}
	
	return (ReadAndVerify<DATA_TYPE, RANK>(arr, 300));
}

