// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Create an array using copy assignment</summary>
// RUN: %amp_device -D__GPU__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll && mkdir -p %t
// RUN: %clamp-device %t.ll %t/kernel.cl
// RUN: pushd %t && %embed_kernel kernel.cl %t/kernel.o && popd
// RUN: %cxxamp %link %t/kernel.o %s -o %t.out && %t.out
#include "./../../../member.h"
#include "../../../../../amp.compare.h"
using namespace Concurrency::Test;
template<typename _type, int _rank>
bool test_feature()
{
    int *edata = new int[_rank];
    for (int i = 0; i < _rank; i++)
        edata[i] = 3;
    extent<_rank> e1(edata);

    {
        std::vector<_type> data;
        for (unsigned int i = 0; i < e1.size(); i++)
            data.push_back((_type)rand());
        array<_type, _rank> src(e1, data.begin());

        array<_type, _rank> dst(e1);

        // Copy assignment use
        dst = src;

        if (!VerifyDataOnCpu<_type, _rank>(src, dst))
        {
            return false;
        }
    }
    
    {
        std::vector<_type> data;
        for (unsigned int i = 0; i < e1.size(); i++)
            data.push_back((_type)rand());
        const array<_type, _rank> src(e1, data.begin());

        array<_type, _rank> dst(e1);

        // Copy assignment - src is const
        dst = src;

        if (!VerifyDataOnCpu<_type, _rank>(src, dst))
        {
            return false;
        }
    }

    return true;
}

int main()
{
    int result = 1;

    result &= ((test_feature<int, 1>()));
    result &= ((test_feature<int, 2>()));
    result &= ((test_feature<int, 5>()));
    result &= ((test_feature<float, 1>()));
    result &= ((test_feature<float, 2>()));
    result &= ((test_feature<float, 5>()));

    return !result;
}
