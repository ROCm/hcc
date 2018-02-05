// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>(Negative) Create a new extent with -ve extents and verify size</summary>

#include "./../../size.h"

template<typename _type>
bool test_size() restrict(amp,cpu)
{
	const int _rank = 5;
    int data[] = {-1, 2, 3, 4, 5};

    extent<_rank> e1(data);
    extent<_rank> g1(e1);

	int correct_size = 1;
	for (int i = 0; i < g1.rank; i++)
	{
            correct_size *= data[i];
	}
			
	return ( correct_size == g1.size());
}

// MAIN function is located in 4_Basic_Data_Elmnts/extentbase.h

