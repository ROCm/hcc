// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.

#include <amptest.h>
#include <amptest_main.h>
#include <vector>

using std::vector;
using namespace Concurrency;
using namespace Concurrency::Test;

template<typename _type>
bool test_feature() restrict(amp,cpu)
{
    return false;
}

template<typename _type, int _tile_x>
bool test_tile_1d() restrict(amp,cpu)
{
    const int rank = _type::rank;
    // 2 elements are passed to extent to reuse this function for negative test case

    // tile 1D extent of extent 1 by tile_x
    {
        int data1[] = {1};
        extent<rank> e1(data1);
        _type g1(e1);

        tiled_extent<_tile_x> t1 = g1.template tile<_tile_x>();

        if ((t1.tile_dim0 != _tile_x) || (t1.size() != g1.size()))
        {
            return false;
        }
    }

    // tile 1D extent with extent
    {
        _type g3;

        tiled_extent<_tile_x> t3 = g3.template tile<_tile_x>();

        if ((t3.tile_dim0 != _tile_x) || (t3.size() != g3.size()))
        {
            //return false;
        }
    }

    // tile 1D extent with extent set to 32
    {
        int data4[] = {32};
        extent<rank> e4(data4);
        _type g4(e4);

        tiled_extent<_tile_x> t4 = g4.template tile<_tile_x>();
        if ((t4.tile_dim0 != _tile_x) || (t4.size() != g4.size()))
        {
            return false;
        }
    }

    // tile 1D extent with extent set to prime number
    {
        int edata6[] = {91};
        extent<rank> e6(edata6);
        _type g6(e6);

        tiled_extent<_tile_x> t6 = g6.template tile<_tile_x>();

        if ((t6.tile_dim0 != _tile_x) || (t6.size() != g6.size()))
        {
            return false;
        }
    }

    return true;
}

template<typename _type, int _tile_x, int _tile_y>
bool test_tile_2d() restrict(amp,cpu)
{
    const int rank = _type::rank;

	// tile 2D extent of extent 1 by tile_x
    {
        int data1[] = {1, 1};
        extent<rank> e1(data1);
        _type g1(e1);

        tiled_extent<_tile_y, _tile_x> t1 = g1.template tile<_tile_y, _tile_x>();

        if ((t1.tile_dim1 != _tile_x) || (t1.tile_dim0 != _tile_y) || (t1.size() != g1.size()))
        {
            return false;
        }
    }

    // tile 2D extent
    {
        _type g3;

        tiled_extent<_tile_y, _tile_x> t3 = g3.template tile<_tile_y, _tile_x>();

        if ((t3.tile_dim1 != _tile_x) || (t3.tile_dim0 != _tile_y) || (t3.size() != g3.size()))
        {
            return false;
        }
    }

    // tile 2D extent with extent set to 32
    {
        int data4[] = {32, 32};
        extent<rank> e4(data4);
        _type g4(e4);

        tiled_extent<_tile_y, _tile_x> t4 = g4.template tile<_tile_y, _tile_x>();

        if ((t4.tile_dim1 != _tile_x) || (t4.tile_dim0 != _tile_y) || (t4.size() != g4.size()))
        {
            return false;
        }
    }

    // tile 2D extent with extent set to prime number
    {
        int edata6[] = {91, 91};
        extent<rank> e6(edata6);
        _type g6(e6);

        tiled_extent<_tile_y, _tile_x> t6 = g6.template tile<_tile_y, _tile_x>();

        if ((t6.tile_dim1 != _tile_x) || (t6.tile_dim0 != _tile_y) || (t6.size() != g6.size()))
        {
            return false;
        }
    }

    return  true;
}


template<typename _type, int _tile_x, int _tile_y, int _tile_z>
bool test_tile_3d() restrict(amp,cpu)
{
    const int rank = _type::rank;

	// tile 3D extent of extent(1)
    {
        int data1[] = {1, 1, 1};
        extent<rank> e1(data1);
        _type g1(e1);

        tiled_extent<_tile_z, _tile_y, _tile_x> t1 = g1.template tile<_tile_z, _tile_y, _tile_x>();

        if ((t1.tile_dim2 != _tile_x) || (t1.tile_dim1 != _tile_y) ||
            (t1.tile_dim0 != _tile_z) || (t1.size() != g1.size()))
        {
            return false;
        }
    }

    // tile 3D extent
    {
        _type g3;

        tiled_extent<_tile_z, _tile_y, _tile_x> t3 = g3.template tile<_tile_z, _tile_y, _tile_x>();

        if ((t3.tile_dim2 != _tile_x) || (t3.tile_dim1 != _tile_y) ||
            (t3.tile_dim0 != _tile_z) || (t3.size() != g3.size()))
        {
            return false;
        }
    }

    // tile 3D extent with extent set to 32
    {
        int data4[] = {32, 32, 32};
        extent<rank> e4(data4);
        _type g4(e4);

        tiled_extent<_tile_z, _tile_y, _tile_x> t4 = g4.template tile<_tile_z, _tile_y, _tile_x>();

        if ((t4.tile_dim2 != _tile_x) || (t4.tile_dim1 != _tile_y) ||
            (t4.tile_dim0 != _tile_z) || (t4.size() != g4.size()))
        {
            return false;
        }
    }

    // tile 3D extent with extent set to prime number
    {
        int edata6[] = {91, 91, 91};
        extent<rank> e6(edata6);
        _type g6(e6);

        tiled_extent<_tile_z, _tile_y, _tile_x> t6 = g6.template tile<_tile_z, _tile_y, _tile_x>();

        if ((t6.tile_dim2 != _tile_x) || (t6.tile_dim1 != _tile_y) ||
            (t6.tile_dim0 != _tile_z) || (t6.size() != g6.size()))
        {
            return false;
        }
    }

    return  true;
}

template<typename _type, int _tile_x>
bool test_tile_1d_negative_incorrect_template_param() restrict(amp,cpu)
{
    const int rank = _type::rank;
    {
        int data1[] = {1, 1};
        extent<rank> e1(data1);
        _type g1(e1);

        tiled_extent<_tile_x> t1 = g1.template tile<_tile_x>();
    }

    return false;
}

template<typename _type, int _tile_x, int _tile_y>
bool test_tile_2d_negative_incorrect_template_param() restrict(amp,cpu)
{
    const int rank = _type::rank;
    {
        int data1[] = {1, 1, 1};
        extent<rank> e1(data1);
        _type g1(e1);

        tiled_extent<_tile_y, _tile_x> t1 = g1.template tile<_tile_y, _tile_x>();
    }
    return false;
}

template<typename _type, int _tile_x, int _tile_y, int _tile_z>
bool test_tile_3d_negative_incorrect_template_param() restrict(amp,cpu)
{
    const int rank = _type::rank;

    {
        int data1[] = {1, 1, 1, 1};
        extent<rank> e1(data1);
        _type g1(e1);

        tiled_extent<_tile_z, _tile_y, _tile_x> t1 = g1.template tile<_tile_z, _tile_y, _tile_x>();
    }

    return false;
}

