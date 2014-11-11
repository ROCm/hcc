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

//#pragma OPENCL EXTENSION cl_amd_printf : enable

#define _REDUCE_STEP(_LENGTH, _IDX, _W) \
    if ((_IDX < _W) && ((_IDX + _W) < _LENGTH)) {\
      T mine = scratch[_IDX];\
      T other = scratch[_IDX + _W];\
      scratch[_IDX] = (*userFunctor)(mine, other); \
    }\
    barrier(CLK_LOCAL_MEM_FENCE);

template< typename iTypePtr, typename iTypeIter, typename binary_function,typename T >
kernel void reduceTemplate(
    global iTypePtr*    input_ptr, 
    iTypeIter input_iter,
    const int length,
    global binary_function* userFunctor,
    global T*    result,
    local T*     scratch
)
{
    int gx = get_global_id (0);
    int gloId = gx;
    input_iter.init( input_ptr );

    //  Initialize the accumulator private variable with data from the input array
    //  This essentially unrolls the loop below at least once
    T accumulator;
    if(gloId < length){
       accumulator = (T) input_iter[gx];
       gx += get_global_size(0);
    }

    // Loop sequentially over chunks of input vector, reducing an arbitrary size input
    // length into a length related to the number of workgroups
    while (gx < length)
    {
        typename iTypeIter::value_type element = input_iter[gx];
        accumulator = (*userFunctor)(accumulator, element);
        gx += get_global_size(0);
    }

    //  Initialize local data store
    int local_index = get_local_id(0);
    scratch[local_index] = accumulator;
    barrier(CLK_LOCAL_MEM_FENCE);

    //  Tail stops the last workgroup from reading past the end of the input vector
    uint tail = length - (get_group_id(0) * get_local_size(0));

    // Parallel reduction within a given workgroup using local data store
    // to share values between workitems - 256 is good to achieve high occupancy
    _REDUCE_STEP(tail, local_index, 128);
    _REDUCE_STEP(tail, local_index, 64);
    _REDUCE_STEP(tail, local_index, 32);
    _REDUCE_STEP(tail, local_index, 16);
    _REDUCE_STEP(tail, local_index,  8);
    _REDUCE_STEP(tail, local_index,  4);
    _REDUCE_STEP(tail, local_index,  2);
    _REDUCE_STEP(tail, local_index,  1);
 
     //  Abort threads that are passed the end of the input vector
    if( gloId >= length )
        return;

    //  Write only the single reduced value for the entire workgroup
    if (local_index == 0) {
        result[get_group_id(0)] = scratch[0];
    }
};
