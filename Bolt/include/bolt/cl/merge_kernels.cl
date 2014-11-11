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

template< typename iTypePtr1,typename iTypeIter1,typename comp_function >
int binary_search1( iTypePtr1 value, iTypeIter1  a,int start, int length,global comp_function *userFunctor)
{
	int low  = start;
	int high = length;
	while( low < high )
	{
		long mid = ( low + high ) / 2;
        iTypePtr1 val1 = a[ mid ];
		if ((*userFunctor)( val1,value ))
		    low  = mid + 1;
		else
			high = mid;

	}
	return high;
}

template< typename iTypePtr1,typename iTypeIter1,typename comp_function >
int binary_search2( iTypePtr1 value, iTypeIter1  a,int start, int length,global comp_function *userFunctor)
{
	int low  = start;
	int high = length;
	while( low < high )
	{
		long mid = ( low + high ) / 2;
        iTypePtr1 val1 = a[ mid ];
		if ((*userFunctor)( value,val1 ))
              high = mid;
		else
		    low  = mid + 1;

	}
	return high;
}


template< typename iTypePtr1, typename iTypeIter1,typename iTypePtr2, typename iTypeIter2,
    typename riTypeIter,typename oTypePtr, typename comp_function>
__kernel void mergeTemplate(
    global iTypePtr1*    input_ptr1,
    iTypeIter1 input_iter1,
    const int length1,
    global iTypePtr2*    input_ptr2,
    iTypeIter2 input_iter2,
    const int length2,
    global oTypePtr* result,
    riTypeIter riter,
    global comp_function* userFunctor
)
{
    int gx = get_global_id (0);
    int pos1,pos2;

    input_iter1.init( input_ptr1 );
    input_iter2.init( input_ptr2 );
    riter.init(result);


    if(gx < length1 )
    {
        iTypePtr1 val = input_iter1[gx];
        pos1 = binary_search1(val , input_iter2,input_iter2.m_StartIndex, length2, userFunctor);
        if (input_iter2[pos1 -1] == val  && pos1 !=0 )
             riter[pos1 + gx -1] = val;
        else
            riter[pos1 + gx] = val;

    }

    if(gx < length2 )
    {
         iTypePtr2 val = input_iter2[gx];
        pos2 = binary_search2(val, input_iter1,input_iter1.m_StartIndex,length1,userFunctor);
        riter[pos2 + gx ] = val;
    }
}
