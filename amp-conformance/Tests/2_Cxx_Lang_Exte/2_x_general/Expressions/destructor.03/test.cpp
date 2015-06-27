// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>This testcase tests that invoking delete operator in amp functions results in compilation error</summary>
//#Expects: Error: error C3930
//#Expects: Error: error C3930
//#Expects: Error: error C3930
//#Expects: Error: error C3930
//#Expects: Error: error C3930
//#Expects: Error: error C3930
//#Expects: Error: error C3930
//#Expects: Error: error C3930
//#Expects: Error: error C3930
//#Expects: Error: error C3930
//#Expects: Error: error C3930
//#Expects: Error: error C3930


#include <amptest.h>
#include <amptest_main.h>

using namespace concurrency;
using namespace concurrency::Test;

class A
{
    int m;
    int n[1000];
    public:
	A() restrict(cpu):m(-1){}
	A() restrict(amp):m(0) {}
};

void test_nullptr(accelerator_view av)
{
    extent<1> ext(10);
    parallel_for_each(av,ext,[=](index<1> idx) restrict(amp){	    	
              A *localPtrObject = NULL; //  Valid stmt only.
	       // Delete,New operators are not allowed in AMP. Even 'deletion' of NULL Ptr is not allowed.
    	      delete localPtrObject;
              delete []localPtrObject;

	      localPtrObject = new A();
	      delete localPtrObject;
    });

}

/*
   Compiler reports errors errors at regions #1,#2,#3,#4
*/

void function1() restrict(cpu,amp)
{
	A *localObj = NULL;
	delete localObj;  // #1
	delete []localObj;// #2

	localObj = new A(); //#3
	delete localObj;    //#4
}

void function2() restrict(amp)
{
	A *localObj = NULL;
	delete localObj;  // #1
	delete []localObj;// #2

	localObj = new A(); //#3
	delete localObj;    //#4
}

void function3() restrict(cpu)
{
	A *localObj = NULL;
	delete localObj;
	delete []localObj;

	localObj = new A();
	delete localObj;
}

runall_result test_main()
{
	accelerator a;
	accelerator_view av = a.get_default_view();
	test_nullptr(av);

 	return runall_fail;

}
