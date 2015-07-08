// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P2</tags>
/// <summary>This test checks that a function modifier can be used as a template parameter</summary>

#include <amptest.h>
#include <iostream>

using namespace std;
using namespace Concurrency;

template<typename restrict>
class Foo
{
    restrict data;

public:

    Foo(restrict d)
    {
        data = d;
    }

    restrict GetData()
    {
       return data;
    }
};


// Main entry point
int main(int argc, char **argv)
{
    cout<< "Test: declare template parameter with the same name as a function modifier" << endl;

    bool passed = true;
    int x = 10;

    Foo<int> *restrict = new Foo<int>(x);
    passed = (restrict->GetData() == x);

    delete restrict;

    cout << (passed? "Passed!" : "Failed!") << endl;

    return passed ? 0 : 1;
}

