// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Check for Decrement Operator</summary>

// RUN: %amp_device -D__GPU__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll && mkdir -p %t
// RUN: %clamp-device %t.ll %t/kernel.cl
// RUN: pushd %t && %embed_kernel kernel.cl %t/kernel.o && popd
// RUN: %cxxamp %link %t/kernel.o %s -o %t.out && %t.out

#include "../../../Helpers/IndexHelpers.h"
#include <amp.h>
using namespace Concurrency;
using std::vector;

//Post DecrementOperator
bool test_post_decrement() restrict(cpu,amp)
{
    int data[] = {-100, -10, -1, 0,  1,  10, 100};
    int data_dec[] = {-101, -11, -2, -1, 0, 9, 99};
    index<7> io(data);
    index<7> i_dec(data_dec);
    index<7> i1,ir;

	i1 = io;
	ir = i1--;

    return ((ir == io) && (i1 == i_dec));
}

//Pre DecrementOperator
bool test_pre_decrement() restrict(cpu,amp)
{
    int data[] = {-100, -10, -1, 0,  1,  10, 100};
    int data_dec[] = {-101, -11, -2, -1, 0, 9, 99};
    index<7> io(data);
    index<7> i_dec(data_dec);
    index<7> i1,ir;

	i1 = io;
	ir = --i1;

    return ((ir == i1) && (i1 == i_dec));
}

int main() 
{ 
    int result = 1;

    result &= test_post_decrement();
    result &= test_pre_decrement();

    return !result;
}
