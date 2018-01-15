// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Declaring multiple destructors results in error</summary>
//#Expects: Error: test\.cpp\(19\) : .+ C2535:.*(\bA::~A\(void\))
//#Expects: Error: test\.cpp\(27\) : .+ C3935:.*(\bf::B::~B\b)
//#Expects: Error: test\.cpp\(38\) : .+ C2535:.*(\bg::C::~g::C\(void\) restrict\(amp\))
#include <amptest.h>
#include <amptest_main.h>
using namespace concurrency;
using namespace concurrency::Test;

class A
{
	~A() restrict(cpu);
	~A() restrict(amp); // Error
};

void f() restrict(cpu,amp)
{
	struct B
	{
		~B() restrict(amp) {}
		~B() restrict(cpu,amp) {} // Error
	};
}

void g() restrict(amp)
{
	[]
	{
		union C
		{
			~C() restrict(amp) {}
			~C() restrict(cpu) {} // Error
		};
	};
}

runall_result test_main()
{
	return runall_fail; // Should not compile.
}

