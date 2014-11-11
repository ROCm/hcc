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

#include <assert.h>


class OclBufferPool {
public:
	struct PoolEntry {
		enum State {e_New, e_Created, e_Reserved};
		PoolEntry() : _state(e_New) {};

		State _state;
		cl::Buffer _buffer;
		cl_mem _bufferCl;
	};



	cl::Buffer alloc(cl_mem_flags flags, int size)
	{
		// FIXME - need atomic acccess here to make this thread-safe
		if (pool._state == PoolEntry::e_New) {
			pool._state =  PoolEntry::e_Reserved;
			pool._buffer =  cl::Buffer(flags, size); 
		} else if ( pool._state == PoolEntry::e_Reserved) {
			throw("OclBufferPool full!");
		} 

		pool._state = PoolEntry::e_Reserved;
		return pool._buffer;
	};
	
	void free(cl::Buffer m)
	{
		// FIXME , need to find the entry, etc.
		assert (pool._state == PoolEntry::e_Reserved);
		pool._state = PoolEntry::e_Created;
	};

	cl_mem allocCl(cl_context context, cl_mem_flags flags, int size)
	{
		// FIXME - need atomic acccess here to make this thread-safe
		if (pool._state == PoolEntry::e_New) {
			pool._state =  PoolEntry::e_Reserved;
			int err;
			pool._bufferCl =  clCreateBuffer(context, flags, size, NULL, &err);
		} else if ( pool._state == PoolEntry::e_Reserved) {
			throw("OclBufferPool full!");
		} 
		pool._state = PoolEntry::e_Reserved;
		return pool._bufferCl;
	};


	void free(cl_mem m)
	{
		// FIXME , need to find the entry, etc.
		assert (pool._state == PoolEntry::e_Reserved);
		pool._state = PoolEntry::e_Created;
	};

private:
	PoolEntry pool;
};

