// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Verify function data() returning valid data for CPU accelerators on CPU</summary>

#include <iterator>
#include "./../../member.h"

template<typename _type, int _rank>
bool test_feature()
{
    int edata[_rank];
    for (int i = 0; i < _rank; i++)
        edata[i] = i+1;
    extent<_rank> e1(edata);

    std::vector<_type> data(e1.size());
    for (unsigned int i = 0; i < e1.size(); i++)
        data[i] = static_cast<_type>(i+1);

    accelerator device(accelerator::cpu_accelerator);

    {
        array<_type, _rank> src(e1, data.begin(), data.end(), device.get_default_view());

        _type* dst_data = src.data();

        if (dst_data == NULL)
            return false;

        for(unsigned int i = 0; i < src.get_extent().size(); i++)
        {
            if (dst_data[i] != data[i])
                return false;
        }
    }

    {
        array<_type, _rank> src(e1, data.begin(), data.end(), device.get_default_view());

        const _type* dst_data = src.data();

        if (dst_data == NULL)
            return false;

        for(unsigned int i = 0; i < src.get_extent().size(); i++)
        {
            if (dst_data[i] != data[i])
                return false;
        }
    }

    {
        const array<_type, _rank> src(e1, data.begin(), data.end(), device.get_default_view());

        const _type* dst_data = src.data();

        if (dst_data == NULL)
            return false;

        for(unsigned int i = 0; i < src.get_extent().size(); i++)
        {
            if (dst_data[i] != data[i])
                return false;
        }
    }

    return true;
}

int main()
{
    int passed = test_feature<int, 5>() && test_feature<float, 7>() &&
                    test_feature<double, 7>() && test_feature<unsigned int, 5>()
            ? runall_pass : runall_fail;

    printf("%s\n", (passed == runall_pass) ? "Passed!" : "Failed!");

    return passed;
}

