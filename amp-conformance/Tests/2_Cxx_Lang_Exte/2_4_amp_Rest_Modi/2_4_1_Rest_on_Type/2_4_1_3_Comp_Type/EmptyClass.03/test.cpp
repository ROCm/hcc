// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Test Struct,Class with static members in GPU function. Regression test for 204819.</summary>


#include <amptest.h>

using namespace Concurrency;
using namespace Concurrency::Test;

struct A
{
    static int s;
};

struct B
{
    static int s;
};


void testEmptyStruct(A* a) __GPU
{
}

void testEmptyClass(B* b) __GPU
{
}

int test(accelerator_view &rv)
{
        A a;
        B b;	
	testEmptyStruct(&a);
	testEmptyClass(&b);
	return 0;
}

int main()
{
    accelerator device;
    if (!get_device(Device::ALL_DEVICES, device))
    {
        printf("Unable to get requested compute device\n");
        return 2;
    }
    accelerator_view rv = device.get_default_view();

    return test(rv);
}

