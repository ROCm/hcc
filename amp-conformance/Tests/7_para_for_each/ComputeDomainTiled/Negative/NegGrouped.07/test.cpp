//--------------------------------------------------------------------------------------
// File: test.cpp
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this
// file except in compliance with the License.  You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR
// CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
//
// See the Apache Version 2.0 License for specific language governing permissions
// and limitations under the License.
//
//--------------------------------------------------------------------------------------
//
/// <tags>P1</tags>
/// <summary>Test that tile extent dimension exceeding limit results in a compilation error. Test for 2D.</summary>
//#Expects: Error: error C3574
//#Expects: Error: test\.cpp\(39\).?:.*tiled_extent
//#Expects: Error: test\.cpp\(42\).?:.*tiled_extent
//#Expects: Error: test\.cpp\(45\).?:.*tiled_extent
//#Expects: Error: test\.cpp\(48\).?:.*tiled_extent

#include <amptest.h>
#include <climits>
using namespace Concurrency;

#define P_F_E(ext, D0, D1) parallel_for_each(ext, [=](tiled_index<D0, D1>) restrict(amp){int y = x;(void)y;})

int main()
{
	int x;

	tiled_extent<1, 1025> ext1 = extent<2>(2, 2050).tile<1, 1025>();
	P_F_E(ext1, 1, 1025);

	tiled_extent<1025, 1> ext2 = extent<2>(2050, 2).tile<1025, 1>();
	P_F_E(ext2, 1025, 1);

	tiled_extent<25, 41> ext3 = extent<2>(50, 82).tile<25, 41>(); //25*41=1025
	P_F_E(ext3, 25, 41);

	tiled_extent<INT_MAX, INT_MAX> ext4 = extent<2>(INT_MAX, INT_MAX).tile<INT_MAX, INT_MAX>();
	P_F_E(ext4, INT_MAX, INT_MAX);

	return runall_fail; // Should not have compiled.
}

