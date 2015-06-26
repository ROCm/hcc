// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>test for restrictions on any class functions/variables in parallel_for_each </summary>
//#Expects: Error: error C3581
//#Expects: Error: error C3930

#include "./../../../dpc_array.h"

template<typename _type>
bool test_feature()
{
    const int _rank = 5;
    int edata[_rank];
    for (int i = 0; i < _rank; i++)
        edata[i] = i+1;
    extent<_rank> e1(edata);

#ifdef _NOT_GPU_
    parallel_for_each(e1, [&, _rank, e1] (index<_rank> idx) __GPU
    {
#endif

#ifdef _NOT_GPU_ONLY_
    parallel_for_each(e1, [&, _rank, e1] (index<_rank> idx) __GPU_ONLY
    {
#endif

#ifdef _CONSTRUCTOR_EXTENT_ONLY_
    array<_type, _rank> arr(e1);
#endif

#ifdef _CONSTRUCTOR_1D_ONLY_
    array<_type, 1> arr(10);
#endif

#ifdef _CONSTRUCTOR_2D_ONLY_
    array<_type, 2> arr(10, 10);
#endif

#ifdef _CONSTRUCTOR_3D_ONLY_
    array<_type, 3> arr(10, 10, 10);
#endif

#ifdef _CONSTRUCTOR_EXTENT_ACCLVW_
    array<_type, _rank> arr(e1, accelerator().get_default_view());
#endif

#ifdef _CONSTRUCTOR_1D_ACCLVW_
    array<_type, 1> arr(10, accelerator().get_default_view());
#endif

#ifdef _CONSTRUCTOR_2D_ACCLVW_
    array<_type, 2> arr(10, 10, accelerator().get_default_view());
#endif

#ifdef _CONSTRUCTOR_3D_ACCLVW_
    array<_type, 3> arr(10, 10, 10, accelerator().get_default_view());
#endif


#ifdef _CONSTRUCTOR_EXTENT_ACCL_STAGING_
    array<_type, _rank> arr(e1, accelerator(), accelerator());
#endif

#ifdef _CONSTRUCTOR_EXTENT_ACCLVW_STAGING_
    array<_type, _rank> arr(e1, accelerator().get_default_view(), accelerator().get_default_view());
#endif

#ifdef _CONSTRUCTOR_1D_ACCLVW_STAGING_
    array<_type, 1> arr(10, accelerator().get_default_view(), accelerator().get_default_view());
#endif

#ifdef _CONSTRUCTOR_2D_ACCLVW_STAGING_
    array<_type, 2> arr(10, 10, accelerator().get_default_view(), accelerator().get_default_view());
#endif

#ifdef _CONSTRUCTOR_3D_ACCLVW_STAGING_
    array<_type, 3> arr(10, 10, 10, accelerator().get_default_view(), accelerator().get_default_view());
#endif

#ifdef _NOT_GPU_
    });
#endif
#ifdef _NOT_GPU_ONLY_
    });
#endif
    return false;
}

int main(int argc, char **argv)
{
    test_feature<int>();

    printf("Failed!");

    return runall_fail;
}

