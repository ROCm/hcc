// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Create a section of a rank 3 array  using the convenience APIs</summary>

// RUN: %amp_device -D__GPU__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll && mkdir -p %t
// RUN: %clamp-device %t.ll %t/kernel.cl
// RUN: pushd %t && %embed_kernel kernel.cl %t/kernel.o && popd
// RUN: %cxxamp %link %t/kernel.o %s -o %t.out && %t.out
#include <amp.h>
#include "../../../../array_test.h"
using namespace Concurrency;
using namespace Concurrency::Test;

int scalar_values()
{
	ArrayTest<int, 3> original(extent<3>(2, 3, 4));
    ArrayViewTest<int, 3> section = original.section(original.arr().section(0, 1, 2, 2, 2, 2), index<3>(0, 1, 2));
    
    original.set_value(index<3>(1, 2, 2), 13);
    
    section.view()(0, 0, 0) = 15;
    section.set_known_value(index<3>(0, 0, 0), 15);
    
    return (gpu_read(original.arr(),index<3>(0, 1, 2)) == 15 && section.view()(1, 1, 0) == 13) ? 1 : 0;
}

int only_index()
{
	ArrayTest<int, 3> original(extent<3>(2, 3, 4));
    ArrayViewTest<int, 3> section = original.section(original.arr().section(index<3>(0, 1, 2)), index<3>(0, 1, 2));
    
    original.set_value(index<3>(1, 2, 2), 13);
    
    section.view()(0, 0, 0) = 15;
    section.set_known_value(index<3>(0, 0, 0), 15);
    
    return (gpu_read(original.arr(),index<3>(0, 1, 2)) == 15 && section.view()(1, 1, 0) == 13) ? 1 : 0;
}

int only_extent()
{
	ArrayTest<int, 3> original(extent<3>(2, 3, 4));
    ArrayViewTest<int, 3> section = original.section(original.arr().section(extent<3>(2, 2, 2)), index<3>(0, 0, 0));
    
    original.set_value(index<3>(1, 1, 1), 13);
    
    section.view()(0, 0, 0) = 15;
    section.set_known_value(index<3>(0, 0, 0), 15);
	
    return (gpu_read(original.arr(),index<3>(0, 0, 0)) == 15 && section.view()(1, 1, 1) == 13)? 1 : 0;
}

int main()
{
   int res = 1;
	
	res &= scalar_values();
	res &= only_index();
	res &= only_extent();
	
	return !res;
}

