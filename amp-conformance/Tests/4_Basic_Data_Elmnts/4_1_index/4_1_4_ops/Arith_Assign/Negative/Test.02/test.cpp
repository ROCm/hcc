// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>(Negative) Check that applying the operator *,/ on index objects of same ranks results in a compilation error</summary>
//#Expects: Error: test.cpp\(35\) : error C2679
//#Expects: Error: test.cpp\(35\) : error C2088
//#Expects: Error: test.cpp\(36\) : error C2679
//#Expects: Error: test.cpp\(36\) : error C2088
//#Expects: Error: test.cpp\(38\) : error C2679
//#Expects: Error: test.cpp\(38\) : error C2088
//#Expects: Error: test.cpp\(39\) : error C2679
//#Expects: Error: test.cpp\(39\) : error C2088

#include <amptest.h>

using namespace Concurrency;
using namespace Concurrency::Test;
using std::vector;


bool test1() __GPU
{
    index<2> i1(5,15);
    index<2> i2(10,30);
    index<2> i3;

    i1 *= 10; // Multiplication, Division using const value are valid
    i1 /= 10;

    i1 *= i2; // Compilation error expected at this line
    i1 /= i2; // Compilation error expected at this line

    i3 = i1 * i2; // Compilation error expected at this line
    i3 = i1 / i2; // Compilation error expected at this line

    return false;
}

runall_result test_main()
{
      return runall_fail;
}

