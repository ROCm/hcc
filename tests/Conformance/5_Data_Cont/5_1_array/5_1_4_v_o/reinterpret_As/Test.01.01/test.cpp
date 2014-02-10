// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Reinterpret an Array of unsigned int as int (GPU)</summary>
// RUN: %amp_device -D__GPU__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll && mkdir -p %t
// RUN: %clamp-device %t.ll %t/kernel.cl
// RUN: pushd %t && %embed_kernel kernel.cl %t/kernel.o && popd
// RUN: %cxxamp %link %t/kernel.o %s -o %t.out && %t.out
#include <amp.h>
#include "../../../../amp.compare.h"
#include "../../../../data.h"
using namespace Concurrency;
using namespace Concurrency::Test;

int main()
{
    std::vector<unsigned int> v(10);
    Fill(v);
    
    array<unsigned int, 1> arr_uint(static_cast<int>(v.size()), v.begin());
    array_view<unsigned int, 1> av_uint(arr_uint);// Created for verification.
	
    // reinterpret on the GPU and copy back
    std::vector<int> results_v(v.size());
    array_view<int, 1> results(static_cast<int>(results_v.size()), results_v);
    results.discard_data();
    parallel_for_each(arr_uint.get_extent(),[=,&arr_uint](index<1> i) restrict(amp,cpu) {
        array_view<int, 1> av_int = arr_uint.reinterpret_as<int>();
        results[i] = av_int[i];
    });
    
    return Verify<int>(reinterpret_cast<int *>(av_uint.data()), results.data(), v.size()) ? 0 : 1;
}

