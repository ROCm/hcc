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


// 1 thread / element: 166 GB/s
template < typename iType, typename iIterType, typename oType, typename oIterType >
__kernel
void copy_I(
    global iType * restrict src,
	iIterType input_iter,
    global oType * restrict dst,
	oIterType output_iter,
    const uint numElements) 
{
    input_iter.init( src );
    output_iter.init( dst );

    size_t gloIdx = get_global_id( 0 );
    if( gloIdx >= numElements) return; // on SI this doesn't mess-up barriers

	output_iter[ gloIdx ] = input_iter[ gloIdx ];

};


// 1 thread / element / BURST
// BURST,bandwidth: 1:150, 2:166, 4:151, 8:157, 12:90, 16:77
template < typename iType, typename oType >
__kernel
void copy_II(
    global iType * restrict src,
    global oType * restrict dst,
    const uint numElements )
{
    int offset = get_global_id(0)*BURST_SIZE;
    __global iType *threadSrc = &src[ offset ];
    __global oType *threadDst = &dst[ offset ];
    iType tmp[BURST_SIZE];
    
    for ( int i = 0; i < BURST_SIZE; i++)
    {
//#if BOUNDARY_CHECK
//        if (offset+i < numElements)
//#endif
            tmp[i] = threadSrc[i];
    }
    for ( int j = 0; j < BURST_SIZE; j++)
    {
//#if BOUNDARY_CHECK
//        if (offset+j < numElements)
//#endif
            threadDst[j] = tmp[j];
    }
    
};



// ideal num threads: 144 GB/s
template < typename iType, typename oType >
__kernel
void copy_III(
    global iType * restrict src,
    global oType * restrict dst,
      const uint numElements )
{
    for (
        unsigned int i = get_global_id(0);
        i < numElements;
        i += get_global_size( 0 ) )
    {
        dst[i] = src[i];
    }
};


// ideal num threads w/ BURST
// BURST Bandwidth
// 1     6 GB/s
// 2     9 GB/s
// 4    14 GB/s
// 8    19 GB/s
template < typename iType, typename oType >
__kernel
void copy_IV(
    global iType * restrict src,
    global oType * restrict dst,
    const uint numElements )
{
    const int numMyElements = numElements / get_global_size(0);
    const int start = numMyElements * get_global_id(0);
    const int stop = (numElements < start+numMyElements) ? numElements : start+numMyElements;

    __private iType tmp[BURST_SIZE];

    for (
        int i = start;
        i < stop;
        i += BURST_SIZE )
    {
        
        for ( int j = 0; j < BURST_SIZE; j++)
        {
            tmp[j] = src[i+j];
        }
        for ( int k = 0; k < BURST_SIZE; k++)
        {
            dst[i+k] = tmp[k];
        }
    }
};

#if 0
#define ITER 16

__attribute__((reqd_work_group_size(256,1,1)))
__kernel
void copy_VInstantiated(
    global const float4 * restrict src,
    global float4 * restrict dst,
      const uint numElements )
{
    //unsigned int offset = get_global_id(0)*BURST_SIZE;
    __global const float4 *srcPtr = &src[get_global_id(0)*BURST_SIZE];
    __global float4 *dstPtr = &dst[get_global_id(0)*BURST_SIZE];
    __private float4 tmp[BURST_SIZE];



#if BURST_SIZE >  0
                tmp[ 0] = srcPtr[ 0];
#endif                            
#if BURST_SIZE >  1                    
                tmp[ 1] = srcPtr[ 1];
#endif                            
#if BURST_SIZE >  2                    
                tmp[ 2] = srcPtr[ 2];
#endif                            
#if BURST_SIZE >  3                    
                tmp[ 3] = srcPtr[ 3];
#endif                            
#if BURST_SIZE >  4                    
                tmp[ 4] = srcPtr[ 4];
#endif                            
#if BURST_SIZE >  5                    
                tmp[ 5] = srcPtr[ 5];
#endif                            
#if BURST_SIZE >  6                    
                tmp[ 6] = srcPtr[ 6];
#endif                            
#if BURST_SIZE >  7                    
                tmp[ 7] = srcPtr[ 7];
#endif                            
#if BURST_SIZE >  8                    
                tmp[ 8] = srcPtr[ 8];
#endif                            
#if BURST_SIZE >  9                    
                tmp[ 9] = srcPtr[ 9];
#endif                            
#if BURST_SIZE > 10                    
                tmp[10] = srcPtr[10];
#endif                            
#if BURST_SIZE > 11                    
                tmp[11] = srcPtr[11];
#endif                            
#if BURST_SIZE > 12                    
                tmp[12] = srcPtr[12];
#endif                            
#if BURST_SIZE > 13                    
                tmp[13] = srcPtr[13];
#endif                            
#if BURST_SIZE > 14                    
                tmp[14] = srcPtr[14];
#endif                            
#if BURST_SIZE > 15                    
                tmp[15] = srcPtr[15];
#endif


#if BURST_SIZE >  0
                dstPtr[ 0] = tmp[ 0];
#endif                            
#if BURST_SIZE >  1                    
                dstPtr[ 1] = tmp[ 1];
#endif                            
#if BURST_SIZE >  2                    
                dstPtr[ 2] = tmp[ 2];
#endif                            
#if BURST_SIZE >  3                    
                dstPtr[ 3] = tmp[ 3];
#endif                            
#if BURST_SIZE >  4                    
                dstPtr[ 4] = tmp[ 4];
#endif                            
#if BURST_SIZE >  5                    
                dstPtr[ 5] = tmp[ 5];
#endif                            
#if BURST_SIZE >  6                    
                dstPtr[ 6] = tmp[ 6];
#endif                            
#if BURST_SIZE >  7                    
                dstPtr[ 7] = tmp[ 7];
#endif                            
#if BURST_SIZE >  8                    
                dstPtr[ 8] = tmp[ 8];
#endif                            
#if BURST_SIZE >  9                    
                dstPtr[ 9] = tmp[ 9];
#endif                            
#if BURST_SIZE > 10                    
                dstPtr[10] = tmp[10];
#endif                            
#if BURST_SIZE > 11                    
                dstPtr[11] = tmp[11];
#endif                            
#if BURST_SIZE > 12                    
                dstPtr[12] = tmp[12];
#endif                            
#if BURST_SIZE > 13                    
                dstPtr[13] = tmp[13];
#endif                            
#if BURST_SIZE > 14                    
                dstPtr[14] = tmp[14];
#endif                            
#if BURST_SIZE > 15                    
                dstPtr[15] = tmp[15];
#endif


};
#endif