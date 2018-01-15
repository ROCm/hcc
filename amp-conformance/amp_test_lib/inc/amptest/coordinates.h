//--------------------------------------------------------------------------------------
// File: coordinates.h
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

#pragma once

#include <amp.h>
#include <iterator>
#include <memory>
#include <cstddef>

namespace Concurrency
{
namespace Test
{
    index<4> make_index(int i0, int i1, int i2, int i3) restrict(cpu,amp)
    {
        int subscripts[4];
        subscripts[0] = i0;
        subscripts[1] = i1;
        subscripts[2] = i2;
        subscripts[3] = i3;
        return index<4>(subscripts);
    };

    index<5> make_index(int i0, int i1, int i2, int i3, int i4) restrict(cpu,amp)
    {
        int subscripts[5];
        subscripts[0] = i0;
        subscripts[1] = i1;
        subscripts[2] = i2;
        subscripts[3] = i3;
        subscripts[4] = i4;
        return index<5>(subscripts);
    };

    extent<4> make_extent(int i0, int i1, int i2, int i3) restrict(cpu,amp)
    {
        int subscripts[4];
        subscripts[0] = i0;
        subscripts[1] = i1;
        subscripts[2] = i2;
        subscripts[3] = i3;
        return extent<4>(subscripts);
    };

    extent<5> make_extent(int i0, int i1, int i2, int i3, int i4) restrict(cpu,amp)
    {
        int subscripts[5];
        subscripts[0] = i0;
        subscripts[1] = i1;
        subscripts[2] = i2;
        subscripts[3] = i3;
        subscripts[4] = i4;
        return extent<5>(subscripts);
    };

    ///<summary>
    /// An abstract base class for classes mapping an index of a certain rank to another
    /// potentially different rank
    ///</summary>
    template<int rank, int original_rank = rank>
    class coordinate_nest
    {
    public:
        virtual index<original_rank> get_absolute(index<rank> i) const = 0;

        virtual unsigned int get_linear(index<rank> i) const = 0;
    };

    template<int rank>
    class extent_coordinate_nest : public coordinate_nest<rank, rank>
    {
    public:
        extent_coordinate_nest(extent<rank> ex) :
            data_extent(ex)
        {
        };

        virtual index<rank> get_absolute(index<rank> i) const
        {
            return i;
        }

        virtual unsigned int get_linear(index<rank> i) const
        {
            auto absolute_index = this->get_absolute(i);

            unsigned int stride = 1;
            unsigned int linear = 0;
            for (int i = rank - 1; i >= 0; i--)
            {
                linear += absolute_index[i] * stride;
                stride *= data_extent[i];
            }

            return linear;
        };

    private:
        extent<rank> data_extent;
    };

    template<int rank, int original_rank = rank>
    class offset_coordinate_nest : public coordinate_nest<rank, original_rank>
    {
    public:
        offset_coordinate_nest(std::shared_ptr<coordinate_nest<rank, original_rank>> n, index<rank> o) :
            next(n),
            offset(o)
        {
        };

        virtual index<original_rank> get_absolute(index<rank> i) const
        {
            return next.get()->get_absolute(i + offset);
        }

        virtual unsigned int get_linear(index<rank> i) const
        {
            return next.get()->get_linear(i + offset);
        };

    private:
        std::shared_ptr<coordinate_nest<rank, original_rank>> next;
        index<rank> offset;
    };

    template<int rank, int next_rank = rank + 1, int original_rank = next_rank>
    class projected_coordinate_nest : public coordinate_nest<rank, original_rank>
    {
    public:
        projected_coordinate_nest(std::shared_ptr<coordinate_nest<next_rank, original_rank>> n, index<next_rank - rank> origin) :
        next(n),
        origin(origin)
        {
        };

        virtual index<original_rank> get_absolute(index<rank> i) const
        {
            return next.get()->get_absolute(this->get_relative_index(i));
        };

        virtual unsigned int get_linear(index<rank> i) const
        {
            return next.get()->get_linear(this->get_relative_index(i));
        };

    private:
        index<next_rank> get_relative_index(index<rank> projected_index) const
        {
            int subscripts[next_rank];
            for (int i = 0; i < next_rank - rank; i++)
            {
                subscripts[i] = origin[i];
            }
            for (int i = 0; i < rank; i++)
            {
                subscripts[i + next_rank - rank] = projected_index[i];
            }

            return index<next_rank>(subscripts);
        };

        std::shared_ptr<coordinate_nest<next_rank, original_rank>> next;
        index<original_rank - rank> origin;
    };

    template<int rank, int original_rank>
    class reshaped_coordinate_nest : public coordinate_nest<rank, original_rank>
    {
    public:
        reshaped_coordinate_nest(std::shared_ptr<coordinate_nest<1, original_rank>> n, extent<rank> ex) :
            next(n),
            original(ex)
        {
        };

        virtual index<1> get_absolute(index<rank> i) const
        {
            return next.get()->get_absolute(index<1>(original.get_linear(i)));
        };

        virtual unsigned int get_linear(index<rank> i) const
        {
            return next.get()->get_linear(index<1>(original.get_linear(i)));
        };

    private:
        std::shared_ptr<coordinate_nest<1, original_rank>> next;
        extent_coordinate_nest<rank> original;
    };

    template<int rank>
    class index_iterator
    {
    public:

        typedef std::input_iterator_tag iterator_category;
        typedef index<rank> value_type;
        typedef std::ptrdiff_t difference_type;
        typedef index<rank>* pointer;
        typedef index<rank>& reference;

        index_iterator(extent<rank> extent) :
            _position(0),
            _extent(extent)
        {
            int stride = 1;
            for (int i = rank - 1; i >= 0; i--)
            {
                _strides[i] = stride;
                stride *= _extent[i];
            }
        }

        index_iterator& operator++()
        {
            _position++;
            return *this;
        };

        index_iterator& operator++(int)
        {
            _position++;
            return *this;
        };


        index<rank> operator*()
        {
            int linear = _position;
            int subscripts[rank];
            for (int i = 0; i < rank; i++)
            {
                subscripts[i] = linear / _strides[i];
                linear %= _strides[i];
            }
            return index<rank>(subscripts);
        };

        bool operator==(const index_iterator<rank> &other)
        {
            return _position == other._position && _extent == other._extent;
        };

        bool operator!=(const index_iterator<rank> &other)
        {
            return !(*this == other);
        };

        index_iterator<rank> begin()
        {
            index_iterator other(*this);
            other._position = 0;
            return other;
        };

        index_iterator<rank> end()
        {
            index_iterator other(*this);
            other._position = other._extent.size();
            return other;
        };

    private:
        int _position;
        extent<rank> _extent;
        int _strides[rank];
    };

	template<int _rank>
	unsigned int flatten(index<_rank> idx, extent<_rank> ext) restrict(cpu,amp)
	{
		int result = idx[0];
		for(int i = 1; i < _rank; i++)
		{
			result = result * ext[i]  + idx[i];
		}
		return result;
	}

    template<>
    unsigned int flatten(index<1> idx, extent<1>) restrict(cpu,amp)
    {
        return idx[0];
    }

    template<>
    unsigned int flatten(index<2> idx, extent<2> ex) restrict(cpu,amp)
    {
        return idx[0] * ex[1] + idx[1];
    }

    template<>
    unsigned int flatten(index<3> idx, extent<3> ex) restrict(cpu,amp)
    {
        return idx[0] * ex[1] * ex[2] + idx[1] * ex[2] + idx[2];
    }
}
}

