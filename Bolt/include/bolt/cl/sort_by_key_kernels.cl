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

#pragma OPENCL EXTENSION cl_amd_printf : enable
//#pragma OPENCL EXTENSION cl_khr_fp64 : enable 
// //FIXME - this was added to support POD with bolt::cl::greater data types
// template<typename T>
// struct greater
// {
    // bool operator()(const T &lhs, const T &rhs) const  {return (lhs > rhs);}
// };
// template<typename T>
// struct less 
// {
    // bool operator()(const T &lhs, const T &rhs) const  {return (lhs < rhs);}
// };

template<typename iKeysType, typename iKeysIter, typename iValType, typename iValIter, typename Compare >
kernel
void BitonicSortByKeyTemplate(
        global iKeysType* keys_ptr, 
        iKeysIter        keys_iter, 
        global iValType* values_ptr, 
        iValIter         values_iter, 
        const uint stage,
        const uint passOfStage,
        global Compare* userComp)
{
    uint threadId = get_global_id(0);
    uint pairDistance = 1 << (stage - passOfStage);
    uint blockWidth   = 2 * pairDistance;
    uint temp;
    uint leftId = (threadId % pairDistance) 
                       + (threadId / pairDistance) * blockWidth;
    bool compareResult;
    uint rightId = leftId + pairDistance;
    
    keys_iter.init( keys_ptr );
    values_iter.init( values_ptr );
    
    iKeysType leftElement = keys_iter[leftId];
    iKeysType rightElement = keys_iter[rightId];
    iValType leftValue = values_iter[leftId];
    iValType rightValue = values_iter[rightId];

    uint sameDirectionBlockWidth = 1 << stage;
    
    if((threadId/sameDirectionBlockWidth) % 2 == 1)
    {
        temp = rightId;
        rightId = leftId;
        leftId = temp;
    }

    compareResult = (*userComp)(leftElement, rightElement);

    if(compareResult)
    {
        keys_iter[rightId] = rightElement;
        keys_iter[leftId]  = leftElement;
        values_iter[rightId] = rightValue;
        values_iter[leftId]  = leftValue;
    }
    else
    {
        keys_iter[rightId] = leftElement;
        keys_iter[leftId]  = rightElement;
        values_iter[rightId] = leftValue;
        values_iter[leftId]  = rightValue;
    }
}
