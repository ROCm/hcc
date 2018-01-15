// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Test Empty Struct,Class,Union definitions in GPU function </summary>


#include <amptest.h>

using namespace Concurrency;
using namespace Concurrency::Test;

typedef struct
{
}EmptyStruct;

typedef union
{
}EmptyUnion;

class EmptyClass
{
};

EmptyStruct testEmptyStruct() __GPU
{
	EmptyStruct a;
	return a;
}

EmptyClass testEmptyClass() __GPU
{
	EmptyClass a;
	return a;
}

EmptyUnion testEmptyUnion() __GPU
{
	EmptyUnion a;
	return a;
}

int test(accelerator_view &rv)
{	
	testEmptyStruct();
	testEmptyClass();
	testEmptyUnion();
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

