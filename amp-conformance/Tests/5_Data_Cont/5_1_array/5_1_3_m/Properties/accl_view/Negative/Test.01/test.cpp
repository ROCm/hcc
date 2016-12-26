// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Verify array accelerator_view property is copied by value</summary>
//#Expects: Error: error C2774

#include "./../../../../member.h"

template<typename _type, int _rank>
bool test_feature()
{
    int edata[_rank];
    for (int i = 0; i < _rank; i++)
        edata[i] = 3;
    const extent<_rank> e1(edata);

    accelerator device(accelerator::default_accelerator);

    accelerator_view av = device.create_view(queuing_mode_immediate);
	Concurrency::array<_type, _rank> src(e1, av);

    src.get_accelerator_view() = device.get_default_view();

    return false;
}

int main(int argc, char **argv)
{
    int passed = test_feature<float, 5>() ? runall_pass : runall_fail;

    printf("%s\n", (passed == runall_pass) ? "Passed!" : "Failed!");

    return passed;
}

