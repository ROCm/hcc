// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Test tile() member function on different extents of extent(2D) and tile by 2D.</summary>

#include "./../tile.h"

template<typename _type>
bool test_tile() restrict(amp,cpu)
{
            // Squared extent
    return  (test_tile_2d<_type, 1, 1>() && test_tile_2d<_type, 11, 11>() &&
            test_tile_2d<_type, 1000, 1000>() && test_tile_2d<_type, 16, 16>() &&
            // rectangled extent
            test_tile_2d<_type, 1, 11>() && test_tile_2d<_type, 1000, 16>() &&
            test_tile_2d<_type, 1000, 1>() && test_tile_2d<_type, 16, 11>());
}

runall_result test_main()
{
	accelerator_view av = require_device(device_flags::NOT_SPECIFIED).get_default_view();

	runall_result result;
	result &= INVOKE_TEST_FUNC_ON_CPU_AND_GPU(av, []() restrict(amp,cpu)->bool{return test_tile<extent<2>>();});
	return result;
}

