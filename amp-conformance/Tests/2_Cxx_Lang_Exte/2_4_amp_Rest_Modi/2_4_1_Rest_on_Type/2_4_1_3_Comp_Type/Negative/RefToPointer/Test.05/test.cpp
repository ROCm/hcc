// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P2</tags>
/// <summary>reference to pointer cannot be capture.</summary>

#include <amptest.h>

using std::vector;
using namespace Concurrency;
using namespace Concurrency::Test;

void test()
{
    double *p;
    double *&rp = p;
    auto a = [=]() __GPU {double d = *rp;};
    auto a2 = [&]() __GPU {double d = *rp;};
}

//#Expects: Error: test.cpp\(19\) : error C3596
//#Expects: Error: test.cpp\(20\) : error C3590

