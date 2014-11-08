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

#include <bolt/unicode.h>
#include <bolt/cl/device_vector.h>

int _tmain( int argc, _TCHAR* argv[ ] )
{
    const size_t vecSize = 10;
    bolt::cl::device_vector< int > dV( vecSize );

    bolt::cl::device_vector< int >::iterator myIter = dV.begin( );
    std::cout << "\nDevice vector EXAMPLE \n";
    //  Iterator arithmetic supported
    *myIter = 1;
    ++myIter;
    *myIter = 2;
    myIter++;
    *myIter = 3;
    myIter += 1;
    *(myIter + 0) = 4;
    *(myIter + 1) = 5;
    myIter += 1;

    //  Operator [] on the container suported
    dV[ 5 ] = 6;
    dV[ 6 ] = 7;

    //  The .data() method maps internal GPU buffer to host accessible memory, and keeps the memory mapped
    bolt::cl::device_vector< int >::pointer pdV = dV.data( );

    //  These are fast writes to host accessible memory
    pdV[ 7 ] = 8;
    pdV[ 8 ] = 9;
    pdV[ 9 ] = 10;

    //  Unmaps the GPU buffer, updating the contents of GPU memory
    pdV.reset( );

    std::cout << "Device Vector contents: " << std::endl;
    for( size_t i = 0; i < vecSize; ++i )
    {
        std::cout << dV[ i ] << ", ";
    }
    return 0;
}
