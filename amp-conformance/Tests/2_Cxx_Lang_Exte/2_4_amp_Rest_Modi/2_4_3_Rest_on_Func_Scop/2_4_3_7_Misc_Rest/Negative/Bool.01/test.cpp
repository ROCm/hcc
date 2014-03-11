// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>C3599: cannot perform pointer arithmetic on pointer to bool in amp restricted code</summary>
//#Expects: Error: error C3599

// Verifies fix to Bug 309184: Disallow pointer arithmetic on non 4-byte aligned types

#include <amptest.h>

void BoolPointerArithmeticNotSupported(bool x) __GPU_ONLY
{
    bool *y = &x;
    bool *z = y+1;
}

int main()
{
    //Execution should never reach here
    //return 1 to indicate failure
    return 1;
}

