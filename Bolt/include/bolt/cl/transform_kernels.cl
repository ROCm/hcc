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

template< typename iNakedType1, typename iIterType1, typename iNakedType2, typename iIterType2, typename oNakedType, 
    typename oIterType, typename binary_function >
kernel
void transformTemplate (
            global iNakedType1* A_ptr,
            iIterType1 A_iter,
            global iNakedType2* B_ptr,
            iIterType2 B_iter,
            global oNakedType* Z_ptr,
            oIterType Z_iter,
			const uint length,
            global binary_function* userFunctor )
{
    int gx = get_global_id( 0 );
	if (gx >= length)
		return;

    A_iter.init( A_ptr );
    B_iter.init( B_ptr );
    Z_iter.init( Z_ptr );
    
    iNakedType1 aa = A_iter[ gx ];
    iNakedType2 bb = B_iter[ gx ];
    
    Z_iter[ gx ] = (*userFunctor)( aa, bb );
}

template< typename iNakedType1, typename iIterType1, typename iNakedType2, typename iIterType2, typename oNakedType, 
    typename oIterType, typename binary_function >
kernel
void transformNoBoundsCheckTemplate (
            global iNakedType1* A_ptr,
            iIterType1 A_iter,
            global iNakedType2* B_ptr,
            iIterType2 B_iter,
            global oNakedType* Z_ptr,
            oIterType Z_iter,
			const uint length,
            global binary_function* userFunctor)
{
    int gx = get_global_id( 0 );
    A_iter.init( A_ptr );
    B_iter.init( B_ptr );
    Z_iter.init( Z_ptr );

    iNakedType1 aa = A_iter[ gx ];
    iNakedType2 bb = B_iter[ gx ];

    Z_iter[ gx ] = (*userFunctor)( aa, bb );
}

template <typename iNakedType, typename iIterType, typename oNakedType, typename oIterType, typename unary_function >
kernel
void unaryTransformTemplate(
            global iNakedType* A_ptr,
            iIterType A_iter,
            global oNakedType* Z_ptr,
            oIterType Z_iter,
			const uint length,
            global unary_function* userFunctor)
{
    int gx = get_global_id( 0 );
	if (gx >= length)
		return;

    A_iter.init( A_ptr );
    Z_iter.init( Z_ptr );

    iNakedType aa = A_iter[ gx ];
    Z_iter[ gx ] = (*userFunctor)( aa );
}

template <typename iNakedType, typename iIterType, typename oNakedType, typename oIterType, typename unary_function >
kernel
void unaryTransformNoBoundsCheckTemplate(
            global iNakedType* A_ptr,
            iIterType A_iter,
            global oNakedType* Z_ptr,
            oIterType Z_iter,
			const uint length,
            global unary_function* userFunctor)
{
    int gx = get_global_id( 0 );

    A_iter.init( A_ptr );
    Z_iter.init( Z_ptr );

    iNakedType aa = A_iter[ gx ];
    Z_iter[ gx ] = (*userFunctor)( aa );
}

#define BURST_SIZE 16

template <typename iType, typename oType, typename unary_function>
kernel
void unaryTransformA (
    global iType* input,
    global oType* output,
    const uint numElements,
    const uint numElementsPerThread,
    global unary_function *userFunctor )
{
	// global pointers
    // __global const iType  *inputBase =  &input[get_global_id(0)*numElementsPerThread];
    // __global oType *const outputBase = &output[get_global_id(0)*numElementsPerThread];

    __private iType  inReg[BURST_SIZE];
    //__private oType outReg[BURST_SIZE];
    //__private unary_function f = *userFunctor;

    // for each burst
    for (int offset = 0; offset < numElementsPerThread; offset+=BURST_SIZE)
    {
        // load burst
        for( int i = 0; i < BURST_SIZE; i++)
        {
            inReg[i]=input[get_global_id(0)*numElementsPerThread+offset+i];
        }
        // compute burst
        //for( int j = 0; j < BURST_SIZE; j++)
        //{
        //    inReg[j]=(*userFunctor)(inReg[j]);
        //}
        // write burst
        for( int k = 0; k < BURST_SIZE; k++)
        {
            output[get_global_id(0)*numElementsPerThread+offset+k]=inReg[k];
        }

    }
    // output[get_global_id(0)] = inReg[BURST_SIZE-1];
}
