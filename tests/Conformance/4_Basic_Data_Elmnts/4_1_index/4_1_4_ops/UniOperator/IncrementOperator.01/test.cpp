// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Check for Increment Operator</summary>

// RUN: %amp_device -D__GPU__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll && mkdir -p %t
// RUN: %clamp-device %t.ll %t/kernel.cl
// RUN: pushd %t && %embed_kernel kernel.cl %t/kernel.o && popd
// RUN: %cxxamp %link %t/kernel.o %s -o %t.out && %t.out

#include "../../../Helpers/IndexHelpers.h"
#include <amp.h>
using namespace Concurrency;
using std::vector;

//Post IncrementOperator
bool test_post_increment() restrict(cpu,amp)
{
    int data[] = {-100, -10, -1, 0,  1,  10, 100};
    int data_inc[] = {-99,  -9,  0, 1, 2, 11, 101};
    index<7> io(data);
    index<7> i_inc(data_inc);
    index<7> i1,ir;

	i1 = io;
	ir = i1++;

    return ((ir == io) && (i1 == i_inc));
}

//Pre IncrementOperator
bool test_pre_increment() restrict(cpu,amp)
{
    int data[] = {-100, -10, -1, 0,  1,  10, 100};
    int data_inc[] = {  -99,  -9,  0, 1, 2, 11, 101};
    index<7> io(data);
    index<7> i_inc(data_inc);
    index<7> i1,ir;

	i1 = io;
	ir = ++i1;

    return ((ir == i1) && (i1 == i_inc));
}

int main() 
{ 
    int result = 1;

    result &= test_post_increment();
    result &= test_pre_increment();

    return !result;
}
