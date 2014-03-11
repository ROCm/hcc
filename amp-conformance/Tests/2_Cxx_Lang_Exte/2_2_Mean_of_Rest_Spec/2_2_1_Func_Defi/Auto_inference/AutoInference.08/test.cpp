// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P2</tags>
/// <summary>Auto inference, caller: Member function , callee: Static member function </summary>

#include <amptest.h>

class c2
{
public:
    void f(int &flag) __GPU
    {
        class c
        {
        public:
            static void f(int &flag)
            {
                flag = 1;
            }
        };

        c::f(flag);
    }
};

class c3
{
public:
    void f(int &flag) __GPU
    {
        class c
        {
        public:
            static void f(int &flag)
            {
                flag = 1;
            }
        };

        c o;
        o.f(flag);
    }
};

bool test() __GPU
{
    int flag = 0;

    bool passed = true;

    c2 o1;
    o1.f(flag);

    if (flag != 1)
    {
        return false;
    }

    flag = 0;

    c3 o2;

    o2.f(flag);

    if (flag != 1)
    {
        return false;
    }

    return passed;
}

int main(int argc, char **argv)
{
    return test() ? 0 : 1;
}

