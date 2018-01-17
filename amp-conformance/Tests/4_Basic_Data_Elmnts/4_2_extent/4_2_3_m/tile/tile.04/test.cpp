// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Test tile() const member function on const objects</summary>

#include "./../tile.h"

template<typename _type>
bool test_tile() restrict(amp,cpu)
{
    // tile 1D 20 extent extent by 5
    {
        const int extsize = 256;
        const int tileby = 8;
        const int dim = 1;
        extent<dim> e1(extsize);
        const extent<dim> g1(e1);

        const tiled_extent<tileby> t1 = g1.tile<tileby>();
    }

    // tile 2D 512 extent extent by 16
    {
        const int extsize = 512;
        const int tileby = 16;
        const int dim = 2;
        extent<dim> e2(extsize, extsize);
        const extent<dim> g2(e2);

        const tiled_extent<tileby, tileby> t2 = g2.tile<tileby, tileby>();
    }

    // tile 3D 1024 extent extent by 16
    {
        const int extsize = 1024;
        const int tileby = 32;
        const int dim = 3;
        extent<dim> e3(extsize, extsize, extsize);
        const extent<dim> g3(e3);

        const tiled_extent<tileby, tileby, tileby> t3 = g3.tile<tileby, tileby, tileby>();
    }

    // No validation here - For any change in const member function compiler will complain
    return true;
}

runall_result test_main()
{
	accelerator_view av = require_device(device_flags::NOT_SPECIFIED).get_default_view();

	runall_result result;
	result &= INVOKE_TEST_FUNC_ON_CPU_AND_GPU(av, []() restrict(amp,cpu)->bool{return test_tile<extent<1>>();});
	return result;
}

