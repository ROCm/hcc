// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Global function with default arguments, returns an object which has the same name as the new function modifier</summary>

#include <amptest.h>
#include <iostream>

using namespace Concurrency;
using namespace std;

class restrict
{

public:
    int data;

    restrict(int d) __GPU
    {
        data = d;
    }
};

restrict test1(restrict a = 1)
{
    restrict b = a;

    b.data++;

    return b;
}

// Main entry point
int main(int argc, char **argv)
{
    bool passed = true;
    int x = 10;

    cout << "Test: Global function with default arguments, returns an object which has the same name as the new function modifier" << endl;
    if(test1(x).data == x + 1)
    {
        cout << "Passed!" << endl;
    }
    else
    {
        passed = false;
        cout << "Failed!" << endl;
    }

    return passed ? 0 : 1;
}
