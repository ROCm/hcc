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

// #pragma OPENCL EXTENSION cl_amd_printf : enable
// #pragma OPENCL EXTENSION cl_khr_fp64 : enable 

template <typename T, typename Compare>
kernel
void selectionSortLocalTemplate(global const T * in, 
                           global T       * out, 
                           global Compare * userComp, 
                           local  T       * scratch, 
                           const int        buffSize)
{
  int          i  = get_local_id(0); // index in workgroup
  int numOfGroups = get_num_groups(0); // index in workgroup
  int groupID     = get_group_id(0);
  int         wg  = get_local_size(0); // workgroup size = block size
  int n;
  
  int offset = groupID * wg;
  int same=0;
  in  += offset; 
  out += offset;
  n = (groupID == (numOfGroups-1))? (buffSize - wg*(numOfGroups-1)) : wg;
  T iData, jData;
  if(i < n)
  {
      iData = in[i];
      scratch[i] = iData;
      barrier(CLK_LOCAL_MEM_FENCE);
  
      int pos = 0;
      for (int j=0;j<n;j++)
      {
          jData = scratch[j];
          if((*userComp)(jData, iData)) 
              pos++;
          else 
          {
              if((*userComp)(iData, jData))
                  continue;
              else 
              {
                  // iData and jData are same
                  same++;
              }
          }
      }
      for (int j=0; j< same; j++)      
         out[pos + j] = iData;
  }
  return;
}

template <typename T, typename Compare>
kernel
void selectionSortFinalTemplate(global const T * in, 
                           global T       * out, 
                           global Compare * userComp,
                           local  T       * scratch, 
                           const int        buffSize)
{
  int          i  = get_local_id(0); // index in workgroup
  int numOfGroups = get_num_groups(0); // index in workgroup
  int groupID     = get_group_id(0);
  int         wg  = get_local_size(0); // workgroup size = block size
  int pos = 0, same = 0;
  int remainder;
  int offset = get_group_id(0) * wg;
  T iData, jData;
  iData = in[groupID*wg + i];
  if((offset + i ) >= buffSize)
      return;
  
  remainder = buffSize - wg*(numOfGroups-1);
  
  for(int j=0; j<numOfGroups-1; j++ )
  {
     for(int k=0; k<wg; k++)
     {
        jData = in[j*wg + k];
        if(((*userComp)(iData, jData)))
           break;
        else
        {
           //Increment only if the value is not the same. 
           //Two elements are same if (*userComp)(jData, iData)  and (*userComp)(iData, jData) are both false
           if( ((*userComp)(jData, iData)) )
              pos++;
           else 
              same++;
        }
     }
  }
  
  for(int k=0; k<remainder; k++)
  {
     jData = in[(numOfGroups-1)*wg + k];
        if(((*userComp)(iData, jData)))
           break;
        else
        {
           //Don't increment if the value is the same. 
           //Two elements are same if (*userComp)(jData, iData)  and (*userComp)(iData, jData) are both false
           if(((*userComp)(jData, iData)))
              pos++;
           else 
              same++;
        }
  }  
  for (int j=0; j< same; j++)      
      out[pos + j] = iData;  
}


template <typename iPtrType, typename iIterType, typename Compare>
kernel
void BitonicSortTemplate(
                    global iPtrType *input_ptr,
                    iIterType       input_iter,
                 const uint stage,
                 const uint passOfStage,
                 global Compare *userComp)
{
    uint threadId = get_global_id(0);
    uint pairDistance = 1 << (stage - passOfStage);
    uint blockWidth   = 2 * pairDistance;
    uint temp;
    uint leftId = (threadId & (pairDistance -1)) 
                       + (threadId >> (stage - passOfStage) ) * blockWidth;
    bool compareResult;
    input_iter.init( input_ptr );
    
    uint rightId = leftId + pairDistance;
    
    iPtrType greater, lesser;
	iPtrType leftElement, rightElement;
    leftElement = input_iter[leftId];
    rightElement = input_iter[rightId];
    
    uint sameDirectionBlockWidth = threadId >> stage;
    uint sameDirection = sameDirectionBlockWidth & 0x1;
    
    temp    = sameDirection?rightId:temp;
    rightId = sameDirection?leftId:rightId;
    leftId  = sameDirection?temp:leftId;

    compareResult = (*userComp)(leftElement, rightElement);

    greater = compareResult?rightElement:leftElement;
    lesser  = compareResult?leftElement:rightElement;

    input_iter[leftId]  = lesser;
    input_iter[rightId] = greater;

}


