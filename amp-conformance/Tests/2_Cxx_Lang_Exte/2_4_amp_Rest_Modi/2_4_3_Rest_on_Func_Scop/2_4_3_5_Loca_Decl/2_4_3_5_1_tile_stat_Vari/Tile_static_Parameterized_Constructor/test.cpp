// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>C6723: tile_static variables cannot be initialized with parameterized constructor</summary>
//#Expects: Error: test.cpp\(26\) : error C3584

#include <amptest.h>

class A
{
public:
    A(int x) __GPU_ONLY : m1{x} {}

private:
    int m1;
};

void test(int x) __GPU_ONLY
{
	tile_static A a{x};
}

int main()
{
	//Execution should never reach here
	//return 1 to indicate failure
	return 1;
}

