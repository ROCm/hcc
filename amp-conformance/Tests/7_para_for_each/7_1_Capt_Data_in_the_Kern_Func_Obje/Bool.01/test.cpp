// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Lambda expression captures bool type</summary>
// Not a negative test anymore. We now allow capturing bool type

#include <iostream>
#include <amptest.h>

using std::vector;
using namespace Concurrency;
using namespace Concurrency::Test;

int main()
{
    const int size = 2048;
    vector<int> c(size);
    Concurrency::extent<1> e(size);
    array<int, 1> ac(e);

    bool hostSideBool = true;

    parallel_for_each(ac.get_extent(), [&, hostSideBool](index<1> idx) __GPU { //error lambda cannot capture bool
        if (hostSideBool)
        {
            ac[idx] = 1;
        }
        else
        {
            ac[idx] = 2;
        }
    });

    c = ac;

    bool passed = true;
    for(int i=0; i<size; ++i)
    {
        int expectedResult = hostSideBool? 1 : 2;
        if (c[i] != expectedResult)
        {
            std::cout << "c[" << i << "] = " << c[i] << " expected:" << expectedResult << std::endl;
            passed = false;
            break;
        }
    }

    std::cout << "lambda test: " << (passed ? "pass" : "fail") << std::endl;

    return passed ? 0 : 1;
}

