// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
// RUN: %amp_device -D__GPU__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll && mkdir -p %t
// RUN: %clamp-device %t.ll %t/kernel.cl
// RUN: pushd %t && %embed_kernel kernel.cl %t/kernel.o && popd
// RUN: %cxxamp %link %t/kernel.o %s -o %t.out && %t.out
#include <amp.h>
#include <vector>

using std::vector;
using namespace Concurrency;

#define INVOKE_TEST_FUNC_ON_CPU_AND_GPU(_func, ...) [&]() { \
	/* Invoke on cpu */ \
	int cpu_result = _func(__VA_ARGS__); \
	/* Invoke on gpu */ \
	int gpu_result; \
	concurrency::array_view<int, 1> gpu_resultv(1, &gpu_result); \
	gpu_resultv.discard_data(); \
	concurrency::parallel_for_each(gpu_resultv.get_extent() \
		, [=](concurrency::index<1> idx) restrict(amp) { \
		gpu_resultv[idx] = _func(__VA_ARGS__); \
	}); \
	gpu_resultv.synchronize(); \
	return cpu_result & gpu_result; \
}()

/// Common Section
template<typename _type>
bool test_feature() restrict(amp,cpu);

template <typename T>
bool test() restrict(amp,cpu)
{
    return test_feature<T>();
}

int main() 
{
	//accelerator_view av = require_device().get_default_view();
	int result = 1;
	result &= INVOKE_TEST_FUNC_ON_CPU_AND_GPU([]() restrict(amp,cpu){
		return test<extent<1>>();
	});

	result &= INVOKE_TEST_FUNC_ON_CPU_AND_GPU([]() restrict(amp,cpu){
		return test<extent<4>>();
	});

	result &= INVOKE_TEST_FUNC_ON_CPU_AND_GPU([]() restrict(amp,cpu){
		return test<extent<10>>();
	});																

	return !result;
}
