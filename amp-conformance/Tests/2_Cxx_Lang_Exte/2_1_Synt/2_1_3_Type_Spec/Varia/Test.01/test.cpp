// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Using a typeid in an explicit template instantiation</summary>

#include "amptest/restrict.h"

template <typename T>
void func(T arg)
{};

template <typename T>
struct S
{};

//explicit instantiations
template void func<int (*)(int, int) restrict(cpu) throw()>(int (*arg)(int, int) restrict(cpu) throw());

template struct S<const int && (int) restrict(cpu)>;

int main()
{
    // if this compiles it passes
    return 0;
}

