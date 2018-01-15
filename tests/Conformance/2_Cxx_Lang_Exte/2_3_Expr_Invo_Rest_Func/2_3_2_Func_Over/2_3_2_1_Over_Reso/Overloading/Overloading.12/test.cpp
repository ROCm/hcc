// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>pointer -> const pointer</summary>

// RUN: %cxxamp %s -o %t.out && %t.out

class c
{
public:
    int f(int *) 
    {
        return 0;
    }

    int f(const int *) restrict(amp,cpu)
    {
        return 1;
    }
};

bool test()
{
    int flag = 0;
    bool passed = true;
    
    c o;

    int *p = 0;

    flag = o.f(p);

    if (flag != 0)
    {
        return false;
    }

    return passed;
}

int main(int argc, char **argv)
{
    return test() ? 0 : 1;
}

