// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P2</tags>
/// <summary>Negative: in GPU function, define int func(int) __GPU; then int (&pfn)(int) __GPU = (int (&)(int) __GPU)func; call pfn</summary>
//#Expects: Error: error C3581

#include <amptest.h>

int func(int) __GPU
{
    return 1;
}

bool test() __GPU
{
    int flag = 0;

    bool passed = true;

    int v = 0;

    int (&pfn)(int) __GPU = (int (&)(int) __GPU)func;

    flag = pfn(v);

    if (flag != 1)
    {
        return false;
    }

    return passed;
}


int main(int argc, char **argv)
{
    test();
    return 1;
}

