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

// BOUNDARY_CHECK = 0,1
// BURST_SIZE = 1,2...64

// 1 thread per element
template <typename oType, typename Generator,  typename iIterType>
__kernel
void generate_I(
    global oType * restrict dst,
	 iIterType input_iter,
    const int numElements,
    global Generator * restrict genPtr)
{
    input_iter.init(dst);

    int gloIdx = get_global_id(0);
#if BOUNDARY_CHECK
    if (gloIdx < numElements)
#endif

	input_iter[gloIdx] = (*genPtr)();
}


// ideal threads
template <typename oType, typename Generator>
kernel
void generate_II(
    global oType * restrict dst,
    const int numElements,
    global Generator * restrict genPtr)
{
    __private Generator gen = *genPtr;
    __private const oType val = gen();
    for (
        unsigned int i = get_global_id(0);
        i < numElements;
        i += get_global_size(0) )
    {
      dst[i] = val;
      //dst[i] = gen();
      //dst[i] = (*genPtr)();
    }
}


// ideal threads w/ Burst
#define STRIDE 16
template <typename oType, typename Generator>
kernel
void generate_III(
    global oType * restrict dst,
		const int numElements,
		global Generator * restrict genPtr)
{
    __private Generator gen = *genPtr;
    __private const oType val = gen();
    for (
        unsigned int i = get_global_id(0)*STRIDE;
        i < numElements;
        i += get_global_size(0)*STRIDE )
    {
        for (unsigned int j = 0; j < STRIDE; j++)
            // dst[i+j] = val;
       
            // dst[i] = val;//gen();//(*genPtr)();
#if STRIDE>0
      dst[i+ 0] = val;
#endif
#if STRIDE>1
      dst[i+ 1] = val;
#endif
#if STRIDE>2
      dst[i+ 2] = val;
#endif
#if STRIDE>3
      dst[i+ 3] = val;
#endif
#if STRIDE>4
      dst[i+ 4] = val;
#endif
#if STRIDE>5
      dst[i+ 5] = val;
#endif
#if STRIDE>6
      dst[i+ 6] = val;
#endif
#if STRIDE>7
      dst[i+ 7] = val;
#endif
#if STRIDE>8
      dst[i+ 8] = val;
#endif
#if STRIDE>9
      dst[i+ 9] = val;
#endif
#if STRIDE>10
      dst[i+10] = val;
#endif
#if STRIDE>11
      dst[i+11] = val;
#endif
#if STRIDE>12
      dst[i+12] = val;
#endif
#if STRIDE>13
      dst[i+13] = val;
#endif
#if STRIDE>14
      dst[i+14] = val;
#endif
#if STRIDE>15
      dst[i+15] = val;
#endif
#if STRIDE>16
      dst[i+16] = val;
#endif
#if STRIDE>17
      dst[i+ 17] = val;
#endif
#if STRIDE>18
      dst[i+18] = val;
#endif
#if STRIDE>19
      dst[i+19] = val;
#endif
#if STRIDE>20
      dst[i+20] = val;
#endif
#if STRIDE>21
      dst[i+21] = val;
#endif
#if STRIDE>22
      dst[i+22] = val;
#endif
#if STRIDE>23
      dst[i+23] = val;
#endif
#if STRIDE>24
      dst[i+24] = val;
#endif
#if STRIDE>25
      dst[i+25] = val;
#endif
#if STRIDE>26
      dst[i+26] = val;
#endif
#if STRIDE>27
      dst[i+27] = val;
#endif
#if STRIDE>28
      dst[i+28] = val;
#endif
#if STRIDE>29
      dst[i+29] = val;
#endif
#if STRIDE>30
      dst[i+30] = val;
#endif
#if STRIDE>31
      dst[i+31] = val;
#endif
     
    }
}