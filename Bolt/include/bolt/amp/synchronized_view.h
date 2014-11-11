/***************************************************************************
*   Copyright 2012 - 2013 Advanced Micro Devices, Inc.
*
*   Licensed under the Apache License, Version 2.0 (the "License");
*   you may not use this file except in compliance with the License.
*   You may obtain a copy of the License at
*
*       http://www.apache.org/licenses/LICENSE-2.0
*
*   Unless required by applicable law or agreed to in writing, software
*   distributed under the License is distributed on an "AS IS" BASIS,
*   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*   See the License for the specific language governing permissions and
*   limitations under the License.

***************************************************************************/

/*! \file bolt/amp/synchronized_view.h
    \brief  Override the CPU implementation of array indexes.
*/

#if !defined( BOLT_AMP_SYNCVIEW_H )
#define BOLT_AMP_SYNCVIEW_H


#include <amp.h>
#pragma once

namespace bolt {

	// Experimental type to see if we can override the CPU implementation of array indexes to eliminate
	// the cpu-side "if synchronized" checks since these are costing a significant amount of performance.
	// This prototype only supports array dimensions of 2.
	template <typename _Value_type, int _Rank = 1>
	class synchronized_view : public concurrency::array_view<_Value_type, _Rank>
	{
	public:
		typedef typename _Value_type value_type;


		template <typename _Container>
		explicit synchronized_view(int _E0, int _E1,  _Container& _Src) :
		concurrency::array_view(_E0 , _E1, _Src)
		{};


		explicit synchronized_view(int _E0, int _E1,  _Value_type * _Src) :
		concurrency::array_view<_Value_type, _Rank> (_E0 , _E1, _Src)
		{};



		//amprt.h: 1817
		_Ret_ void * _Access(_Access_mode _Requested_mode, const concurrency::index<_Rank>& _Index) const __CPU_ONLY
		{
			static_assert(_Rank == 2, "value_type& array_view::operator()(int,int) is only permissible on array_view<T, 2>");
			int * _Ptr = reinterpret_cast<int *>(_M_buffer_descriptor._M_data_ptr);
			// This only works for array_dim = 2, we couldn't call flatten_helper without private access to Index._M_base.
			return &_Ptr[_M_total_linear_offset + ((sizeof(_Value_type)/sizeof(int)) * (_M_array_multiplier[0] * _Index[0] + _Index[1]))];
			//return &_Ptr[                   0   + ((sizeof(_Value_type)/sizeof(int)) * (_M_array_multiplier[0] * _Index[0] + _Index[1]))];
			//return &_Ptr[_M_total_linear_offset + (_Element_size * _Flatten_helper::func(_M_array_multiplier._M_base, _Index._M_base))];
		}

		_Ret_ void * _Access(_Access_mode _Requested_mode, const concurrency::index<_Rank>& _Index) const __GPU_ONLY
		{

			UNREFERENCED_PARAMETER(_Requested_mode);

			int * _Ptr = reinterpret_cast<int *>(_M_buffer_descriptor._M_data_ptr);
			return &_Ptr[_M_total_linear_offset + ((sizeof(_Value_type)/sizeof(int)) * (_M_array_multiplier[0] * _Index[0] + _Index[1]))];
			//return &_Ptr[_M_total_linear_offset + (_Element_size * _Flatten_helper::func(_M_array_multiplier._M_base, _Index._M_base))];
		}

		// amp.h: 2309, amprt.h:1756
		value_type& operator() (const concurrency::index<_Rank>& _Index) const __GPU
		{
			void * _Ptr = _Access(_Read_write_access, _Index);
			return *reinterpret_cast<_Value_type*>(_Ptr);
		}

		_Value_type& operator() (int _I0, int _I1) const __GPU
		{
			static_assert(_Rank == 2, "value_type& array_view::operator()(int,int) is only permissible on array_view<T, 2>");
			return this->operator()(concurrency::index<2>(_I0,_I1));
		}

	};
};

#endif
