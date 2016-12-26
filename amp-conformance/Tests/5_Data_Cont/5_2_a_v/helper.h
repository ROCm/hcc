//--------------------------------------------------------------------------------------
// File: helper.h
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
//
// Helper functions to compare data in a vector to data in an array_view
// These are needed because the .data() member which serializes data out of
// on array is not available on a 2D and 3D array_view

#pragma once

#include <amptest.h>
#include <vector>

using namespace Concurrency;
using namespace Concurrency::Test;
using std::vector;

template<typename T, typename U>
bool compare(const vector<T>& vec, const array_view<U, 1>& av)
{
	for(int i = 0; i < av.get_extent()[0]; i++)
	{
		if(vec[i] != av(i))
		{
			Log(LogType::Error, true) << compose_incorrect_element_message(i, vec[i], av(i)) << std::endl;
			return false;
		}
	}

	return true;
}

template<typename T, typename U>
bool compare(const vector<T>& vec, const array_view<U, 2>& av)
{
	for(int i = 0; i < av.get_extent()[1]; i++)
	{
		for(int j = 0; j < av.get_extent()[0]; j++)
		{
			if(vec[i * av.get_extent()[1] + j] != av(i,j))
			{
				Log(LogType::Error, true) << compose_incorrect_element_message(index<2>(i,j), vec[i * av.get_extent()[1] + j], av(i,j)) << std::endl;
				return false;
			}
		}
	}

	return true;
}

template<typename T, typename U>
bool compare(const vector<T>& vec, const array_view<U, 3>& av)
{
	for(int i = 0; i < av.get_extent()[0]; i++)
	{
		for(int j = 0; j < av.get_extent()[1]; j++)
		{
			for(int k = 0; k < av.get_extent()[2]; k++)
			{
				if(vec[i * av.get_extent()[1] * av.get_extent()[2] + j * av.get_extent()[2] + k] != av(i,j,k))
				{
					Log(LogType::Error, true) << compose_incorrect_element_message(index<3>(i,j,k), vec[i * av.get_extent()[1] * av.get_extent()[2] + j * av.get_extent()[2] + k], av(i,j,k)) << std::endl;
					return false;
				}
			}
		}
	}

	return true;
}

///<summary> Verifies that the extent of an array view matches expected values </summary>
template <typename TValue, int Rank>
bool verify_extent(concurrency::array_view<TValue, Rank> actual, concurrency::extent<Rank> expected)
{
	using Concurrency::Test::operator<<;

	if(actual.get_extent() != expected)
	{
		Log(LogType::Error, true) << "Extent doesn't match expected "
			<< "Expected: " << expected << " Actual: " << actual.get_extent() << std::endl;
		return false;
	}

	return true;
}
