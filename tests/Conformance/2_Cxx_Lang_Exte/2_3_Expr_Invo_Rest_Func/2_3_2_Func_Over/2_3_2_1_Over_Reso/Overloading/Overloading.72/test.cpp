// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>pointer to member -> const pointer to member or add __GPU</summary>

// RUN: %cxxamp %s -o %t.out && %t.out

#include <stdio.h>

class c2
{};

class c
{
public:
    int f(const int c2::*) 
    {
        return 0;
    }

    int f(int c2::*) restrict(amp,cpu)
    {
        return 1;
    }
};

bool test()
{
    bool passed = true;

    c o;

    int c2::* p = NULL;

    int v = o.f(p);

    if (v != 1)
        passed = false;

    return passed;
}

int main(int argc, char **argv)
{
    bool passed = true;

    passed = test();

    return passed ? 0 : 1;

}

