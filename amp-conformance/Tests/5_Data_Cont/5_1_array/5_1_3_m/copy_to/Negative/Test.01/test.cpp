// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Verify copy_to function where src space is more than dst space</summary>

#include "./../../../member.h"

template<typename _type, int _rank>
bool test_feature()
{
    int edata_src[_rank];
    int edata_dst[_rank];
    for (int i = 1; i < _rank+1; i++) // extent < 1 is not supported
    {
        edata_src[i-1] = i+1;
        printf("src %d %d\n", i, edata_src[i-1]);
        edata_dst[i-1] = i;
        printf("dst %d %d\n", i, edata_dst[i-1]);
    }
    extent<_rank> esrc(edata_src);
    extent<_rank> edst(edata_dst);

    std::vector<_type> data_src(esrc.size());
    for (unsigned int i = 0; i < esrc.size(); i++)
        data_src[i] = (_type)rand();

    std::vector<_type> vsrc;
    std::vector<_type> vdst;

    {
        array<_type, _rank> src(esrc, data_src.begin());
        array<_type, _rank> dst(edst);

        src.copy_to(dst);

        vsrc = src;
        vdst = dst;

        if (vdst.size() != vdst.size())
            return false;

        for (size_t i = 0; i < vdst.size(); i++)
        {
            if (vdst[i] != vsrc[i])
                return false;
        }
    }

    return true;
}

int main()
{
    try
    {
        test_feature<float, 5>();
    }
    catch (runtime_exception &ex)
    {
	return runall_pass;
    }
    catch (std::exception e)
    {
	return runall_fail;
    }

    printf("Failed!");

    return runall_fail;
}

