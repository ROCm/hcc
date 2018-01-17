// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P2</tags>
/// <summary>This test checks that a function modifier can be used as a namespace name</summary>

#include <amptest.h>
#include <iostream>

using namespace std;
using namespace Concurrency;

namespace restrict
{
    class Foo
    {
    public:
        double data;

        explicit Foo(double d) __GPU
        {
            data = d;
        }
    };
}

restrict::Foo func(double d)
{
    using namespace restrict;

    Foo foo = Foo(d);

    return foo;
}

// Main entry point
int main(int argc, char **argv)
{
    cout << "Test: declare function modifier as a namespace" << endl;
    double d = 445.344;
    bool passed = true;

    restrict::Foo foo = restrict::Foo(d);
    passed &= (foo.data == d);

    restrict::Foo bar = func(d);
    passed &= (bar.data == d);

    cout << (passed? "Passed!" : "Failed!") << endl;

    return passed ? 0 : 1;
}



