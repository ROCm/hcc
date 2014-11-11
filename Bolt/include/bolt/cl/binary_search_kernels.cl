/***************************************************************************
* Copyright 2012 - 2013 Advanced Micro Devices, Inc.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
***************************************************************************/
template < typename iType, typename T, typename iIterType, typename StrictWeakOrdering >
__kernel void binarysearch_kernel(
 global iType * src,
 iIterType input_iter,
 const T val,
 const uint numElements,
 global StrictWeakOrdering * comp,
 global int * result,
 const uint startIndex,
 const uint endIndex )
 {
    input_iter.init(src);
    size_t gloId = get_global_id( 0 );
    size_t mid;
    size_t low = startIndex + gloId * numElements;
    size_t high = low + numElements;
    size_t resultIndex = startIndex/numElements;

    if(high > endIndex)
        high = endIndex;

    int found = 0;
    
    //printf("\n\nstartIndex =%d gloId=%d low = %d, high = %d", startIndex, gloId, low, high );
    
    while(low < high)
    {	
        mid = (low + high) / 2;
        
        iType midVal = input_iter[mid];
        iType firstVal = input_iter[low];
        //printf("\nlow = %d, high = %d midVal=%d val =%d", low, high, midVal, val);
        if( (!(*comp)(midVal, val)) && (!(*comp)(val, midVal)) )
        {
		    //printf("\n Element found!");
            found = 1;
            break;
        }
        else if ( (*comp)(midVal, val) ) /*if true, midVal comes before val hence adjust low*/
            low = mid + 1;
        else	/*else val comes before midVal, hence adjust high*/
            high = mid;
    }
    //printf ("\n Setting found to = %d at index = %d", found, resultIndex+gloId);
    
    result[resultIndex+gloId] = found;
};
