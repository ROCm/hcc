// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Declares a function accepting a reference to a restrict(amp) function</summary>

#include "amptest/restrict.h"
#include "amptest/runall.h"

int amp_function(int x, int y) __GPU_ONLY
{
    return x + y;
}

int test(int (&p)(int, int) __GPU_ONLY)
{
    // can't call an amp function through a pointer
    return 1;
}

int main()
{
    test(amp_function);
    return runall_pass;
}

//#Expects: Error: error C3939
