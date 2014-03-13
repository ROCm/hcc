// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>(negative)caller: __GPU, class static; callee: non __GPU, class static</summary>
//#Expects: Error: error C3930

#include <amptest.h>

class C1
{
public:
    static void foo(int &flag) {flag = 1;}
};

class C2
{
public:
    static void foo(int &flag) __GPU {C1::foo(flag);}
};

static bool test() __GPU
{
    int flag = 0;

    C2::foo(flag);

    if (flag == 1)
    {
        return true;
    }
    else
    {
        return false;
    }
}

int main(int argc, char **argv)
{
    return test() ? 0 : 1;
}

