//--------------------------------------------------------------------------------------
// File: Utils.h
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
// This header file contains common helpers for the ComputeDomainTiled Negative tests covering exceptions thrown by the parallel_for_each.
#pragma once
#include <amptest.h>

// Runs a p_f_e on a given accelerator view with a given extent and expects an invalid_compute_domain exception with a given message.
template <int _Dim0, int _Dim1, int _Dim2>
runall_result expect_exception(const concurrency::accelerator_view& av, const concurrency::tiled_extent<_Dim0, _Dim1, _Dim2>& ext)
{
	using namespace concurrency;
	using namespace concurrency::Test;

	try
	{
		int x;
		parallel_for_each(av, ext, [=](tiled_index<_Dim0, _Dim1, _Dim2>) restrict(amp) { int y = x; (void)y; });
	}
	catch(const invalid_compute_domain& e)
	{
		return runall_pass;
	}

	return runall_fail;
}

