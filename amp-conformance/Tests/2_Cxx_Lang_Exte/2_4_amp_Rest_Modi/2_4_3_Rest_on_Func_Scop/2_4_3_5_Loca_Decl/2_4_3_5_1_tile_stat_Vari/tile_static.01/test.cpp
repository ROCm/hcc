// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Check that tile_static cannot be initialized</summary>
//#Expects: Error: error C3584

#include <amptest.h>
#include <iostream>

using namespace Concurrency;
using namespace std;

class StaticClass
{
    public:
    static int foo(int a, int b) __GPU_ONLY
    {
        tile_static int c = a + b;
        return c;
    }
};



// Main entry point
int main(int argc, char **argv)
{
    bool passed = true;

    Concurrency::extent<1> ex(1);
    Concurrency::array<int, 1> arr(ex);

    parallel_for_each(arr.get_extent(), [&](index<1> idx) __GPU_ONLY {

       arr[idx] = StaticClass::foo(1, 2);

    });

    vector<int> v = arr;

    if(v[0] != 3)
    {
        passed = false;
        cout << "Failed\n" << endl;
    }
    else
    {
        cout << "Passed\n" << endl;
    }

    return passed ? 0 : 1;
}

