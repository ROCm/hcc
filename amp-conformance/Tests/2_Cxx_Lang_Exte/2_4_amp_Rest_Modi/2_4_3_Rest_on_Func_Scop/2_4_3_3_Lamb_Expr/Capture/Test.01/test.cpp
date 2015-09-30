// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Capture an array by reference and array_view by value (no  default capture mode)</summary>

#include "amptest.h"
#include <vector>

using namespace Concurrency;
using namespace Concurrency::Test;


int main()
{
    array<int, 1> a(1);

    std::vector<int> v(1);
    array_view<int, 1> av(1, v);

    auto l = [&a, av] () __GPU {};

    return runall_pass;
}
