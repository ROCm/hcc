// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.

#include "./../../extentbase.h"

template<typename _type>
bool test_size() restrict(cpu,amp);

template<typename _type>
bool test_feature() __GPU
{
    return test_size<_type>();
}

template<typename _type>
bool test_positive_size() __GPU
{
    const int rank = _type::rank;

    // all positive elements
    int data1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    {
        int correct_size = 1;
        extent<rank> e1(data1);
        _type g1(e1);

        for (int i = 0; i < g1.rank; i++)
            correct_size *= data1[i];
        if (correct_size != g1.size())
            return false;
    }

    {
        int correct_size = 1;
        extent<rank> e1(data1);
        _type g1(e1);

        for (int i = 0; i < g1.rank; i++)
            correct_size *= data1[i];
        if (correct_size != g1.size())
            return false;
    }

    return true;
}

template<typename _type>
bool test_size_const_attribute() __GPU
{
    const int rank = _type::rank;
    int data1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    {
        int correct_size = 1;
        extent<rank> e1(data1);
        const _type g1(e1);

        for (int i = 0; i < g1.rank; i++)
            correct_size *= data1[i];
        if (correct_size != g1.size())
            return false;
    }

    {
        int correct_size = 1;
        extent<rank> e1(data1);
        const _type g1(e1);

        for (int i = 0; i < g1.rank; i++)
            correct_size *= data1[i];
        if (correct_size != g1.size())
            return false;
    }

    {
        int correct_size = 0; // default construtor
        const _type g1;

        if (correct_size != g1.size())
            return false;
    }

    return true;
}

