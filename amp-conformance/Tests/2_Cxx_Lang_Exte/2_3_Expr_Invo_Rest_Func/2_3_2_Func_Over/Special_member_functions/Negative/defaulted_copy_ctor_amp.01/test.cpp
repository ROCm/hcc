// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Test whether defaulted copy constructors have restrict(amp) specifiers.</summary>
//#Expects: Error: test\.cpp\(25\) : .+ C3930:.*(\bA1::A1\b)
//#Expects: Error: test\.cpp\(28\) : .+ C3930:.*(\bA2::A2\b)
//#Expects: Error: test\.cpp\(31\) : .+ C3930:.*(\bA3::A3\b)
//#Expects: Error: test\.cpp\(34\) : .+ C3930:.*(\bA4::A4\b)
//#Expects: Error: test\.cpp\(37\) : .+ C3930:.*(\bA5::A5\b)
//#Expects: Error: test\.cpp\(40\) : .+ C3930:.*(\bA6::A6\b)
//#Expects: Error: test\.cpp\(43\) : .+ C3930:.*(\bA7::A7\b)
//#Expects: Error: test\.cpp\(46\) : .+ C3930:.*(\bA8::A8\b)
#include <amptest.h>
#include <amptest_main.h>
#include "../../common_defaulted_copy_ctor_amp.01.h"
using namespace concurrency;
using namespace concurrency::Test;

void test() restrict(cpu)
{
	A1* a1_ptr; // A1 does not have cpu restricted ctor
	new A1(*a1_ptr);

	A2 a2;
	A2 a2c(a2);

	A3 a3;
	A3 a3c(a3);

	A4 a4;
	A4 a4c(a4);

	A5 a5;
	A5 a5c(a5);

	A6 a6;
	A6 a6c(a6);

	A7 a7;
	A7 a7c(a7);

	A8* a8_ptr; // A8 does not have cpu restricted ctor
	new A8(*a8_ptr);
}

runall_result test_main()
{
	return runall_fail; // Should not compile.
}

