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
template < typename T, typename Type, typename iIterType >
__kernel
void fill_kernel(
    const T src,
    global Type * dst,
	iIterType input_iter,
    const uint numElements )
{
    input_iter.init(dst);

    size_t gloId = get_global_id( 0 );
    if( gloId >= numElements ) return; // on SI this doesn't mess-up barriers
    
	input_iter[ gloId ] = src;
};
