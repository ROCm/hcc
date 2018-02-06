// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Test whether defaulted destructors have restrict(cpu) specifiers.</summary>
//#Expects: Error: test\.cpp\(26\) : .+ C3930:.*(\bA1::~A1\b)
//#Expects: Error: test\.cpp\(27\) : .+ C3930:.*(\bA2::~A2\b)
//#Expects: Error: test\.cpp\(28\) : .+ C3930:.*(\bA3::~A3\b)
//#Expects: Error: test\.cpp\(29\) : .+ C3581:.*(\bA4\b)
//#Expects: Error: test\.cpp\(30\) : .+ C3581:.*(\bA5\b)
//#Expects: Error: test\.cpp\(31\) : .+ C3930:.*(\bA6::~A6\b)
//#Expects: Error: test\.cpp\(32\) : .+ C3930:.*(\bA7::~A7\b)
//#Expects: Error: test\.cpp\(33\) : .+ C3930:.*(\bA8::~A8\b)
//#Expects: Error: test\.cpp\(34\) : .+ C3930:.*(\bA9::~A9\b)
//#Expects: Error: test\.cpp\(35\) : .+ C3930:.*(\bA10::~A10\b)
#include <amptest.h>
#include <amptest_main.h>
#include "../../common_defaulted_dtor_cpu.01.h"
using namespace concurrency;
using namespace concurrency::Test;

void test() restrict(amp)
{
	A1 a1;
	A2 a2;
	A3 a3;
	A4 a4;
	A5 a5;
	A6 a6;
	A7 a7;
	A8 a8;
	A9 a9;
	A10 a10;
}

runall_result test_main()
{
	return runall_fail; // Should not compile.
}

