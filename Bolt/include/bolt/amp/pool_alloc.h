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

/*! \file bolt/amp/pool_alloc.h
    \brief Pool allocator.
*/
#if !defined( BOLT_AMP_PARALLEL_ALLOC_H )
#define BOLT_AMP_PARALLEL_ALLOC_H


#include <amp.h>
#pragma once

namespace bolt {

	// FIXME - this bool should be independent of type, should just allocate raw bytes.
	template<typename T>
	class ArrayPool {
	public:
		struct PoolEntry {
			enum State {e_New, e_Created, e_Reserved};
			PoolEntry() : _state(e_New), _dBuffer(NULL), _stagingBuffer(NULL) {};

			State _state;
			concurrency::array<T> *_dBuffer;	// storage on accelerator
			concurrency::array<T> *_stagingBuffer;
		};

		PoolEntry &alloc(concurrency::accelerator_view av,  int size)
		{
			using namespace concurrency;
			// FIXME - need atomic acccess here to make this thread-safe
			if (pool[0]._state == PoolEntry::e_New) {
				pool[0]._state =  PoolEntry::e_Reserved;
				accelerator cpuAccelerator = accelerator(accelerator::cpu_accelerator);
				pool[0]._stagingBuffer =  new  array<T,1>(size, cpuAccelerator.default_view, av);  // cpu memory
				pool[0]._dBuffer =  new  concurrency::array<T,1>(size, av);
			} else if ( pool[0]._state == PoolEntry::e_Reserved) {
				throw("OclBufferpool[0] full!");
			}

			pool[0]._state = PoolEntry::e_Reserved;
			return pool[0];
		};

#if 0
		concurrency::array<T,1> &alloc(int size)
		{
			// FIXME - need atomic acccess here to make this thread-safe
			if (pool[0]._state == PoolEntry::e_New) {
				pool[0]._state =  PoolEntry::e_Reserved;
				pool[0]._dBuffer =  new  concurrency::array<T,1>(size);
			} else if ( pool[0]._state == PoolEntry::e_Reserved) {
				throw("OclBufferpool[0] full!");
			}

			pool[0]._state = PoolEntry::e_Reserved;
			return *(pool[0]._dBuffer);
		};

		void free(/*cl::Buffer m*/)
		{
			// FIXME , need to find the entry, etc.
			assert (pool[0]._state == PoolEntry::e_Reserved);
			pool[0]._state = PoolEntry::e_Created;
		};
#endif

		void free(PoolEntry &poolEntry)
		{
			// FIXME , need to find the entry, etc.
			assert (poolEntry._state == PoolEntry::e_Reserved);
			poolEntry._state = PoolEntry::e_Created;
		};

	private:
		PoolEntry pool[1];
	};
};

#endif
