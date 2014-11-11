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
#include <bolt/cl/scan.h>

#include <vector>
#include <numeric>

int _tmain( int argc, _TCHAR* argv[ ] )
{
    size_t length = 1024;
    //Create device_vector and initialize it to 1
    std::cout << "\nScan EXAMPLE \n";
    bolt::cl::device_vector< int > boltInput( length, 1 );
    std::cout << "\n\nInclusive Scan of device_vector with integers of length " << length << " elements with bolt Functor. ...\n";
    bolt::cl::device_vector< int >::iterator boltEnd = bolt::cl::inclusive_scan( boltInput.begin( ), boltInput.end( ), boltInput.begin( ) );
    std::cout << "COMPLETED. ...\n";

    //Create std vector and initialize it to 1
    std::cout << "\n\nInclusive Scan of device_vector with integers " << length << " elements with bolt Functor. ...\n";
    std::vector< int > stdInput( length, 1 );
    std::vector< int >::iterator stdEnd  = bolt::cl::inclusive_scan( stdInput.begin( ), stdInput.end( ), stdInput.begin( ) );
    std::cout << "COMPLETED. ...\n";

	return 0;
}
