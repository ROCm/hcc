// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Test that copy_to const array_view<T,N> is allowed</summary>
// RUN: %amp_device -D__GPU__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll && mkdir -p %t
// RUN: %clamp-device %t.ll %t/kernel.cl
// RUN: pushd %t && %embed_kernel kernel.cl %t/kernel.o && popd
// RUN: %cxxamp %link %t/kernel.o %s -o %t.out && %t.out
#include "../../../../amp.compare.h"
#include "../../../../data.h"
using namespace concurrency;
using namespace Concurrency::Test;
using std::vector;

template<unsigned int rank>
int test(extent<rank> e, accelerator_view acc_view)
{
    vector<int> src_v(e.size());
    Fill<int>(src_v);
    array<int, rank> src(e, src_v.begin(), acc_view);

    vector<int> dst_v(e.size());
    const array_view<int, rank> dst_av(e, dst_v);

    src.copy_to(dst_av);
    dst_av.synchronize();
    return Verify(dst_v, src_v);
}

int main()
{
    accelerator def;
    accelerator_view acc_view = def.get_default_view();
    //accelerator_view acc_view = require_device().create_view();

    int result = 1;

    extent<1> e1(10);
    extent<2> e2(1, 20);
    extent<3> e3(10, 2, 3);
    
    result &= test<1>(e1, acc_view);
    result &= test<2>(e2, acc_view);
    result &= test<3>(e3, acc_view);
    
    return !result;
}
