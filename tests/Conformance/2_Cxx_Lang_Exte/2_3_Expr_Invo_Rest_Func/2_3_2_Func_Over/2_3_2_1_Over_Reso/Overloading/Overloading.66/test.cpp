// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Select __GPU function over non __GPU</summary>

// RUN: %cxxamp %s -o %t.out && %t.out

class c2 {};

class c1
{
public:

    operator c2() restrict(amp)
    {
        flag = 1;
        c2 o;
        return o;
    }

    operator c2() 
    {
        flag = 2;
        c2 o;
        return o;
    }

    int flag;
};

bool test()
{
    bool passed = true;

    c1 o1;

    c2 o2 = o1;

    if (o1.flag != 2)
    {
        return false;
    }

    return passed;
}

int main(int argc, char **argv)
{
    return test() ? 0 : 1;
}

