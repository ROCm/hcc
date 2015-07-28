// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Supported extern "C" and explicit "C++" linkage specifications</summary>

#include "amptest/runall.h"

extern "C++" void foo() restrict(amp, cpu); // ok this is C++ linkage

extern "C++" {
    void boo() restrict(amp, cpu); // same as above
}

extern "C" void hoo() restrict(cpu); // ok, single restriction modifier

extern "C" void poo(); // default restriction modifier

extern "C" void qoo() restrict(amp); // ok, single restriction modifier

int main()
{
    return runall_pass;
}
