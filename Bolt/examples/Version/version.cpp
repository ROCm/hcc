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
#include <bolt/BoltVersion.h>
#include <bolt/cl/bolt.h>
#include <iostream>
#include <iomanip>

int _tmain( int argc, _TCHAR* argv[ ] )
{
    cl_uint libMajor = 0, libMinor = 0, libPatch = 0;
    cl_uint appMajor = 0, appMinor = 0, appPatch = 0;
    
    // These version numbers come directly from the Bolt header files, and represent the version of header that the app is compiled against
    appMajor = BoltVersionMajor;
    appMinor = BoltVersionMinor;
    appPatch = BoltVersionPatch;
    
    // These version numbers come from the Bolt library, and represent the version of headers that the lib is compiled against
    bolt::cl::getVersion( libMajor, libMinor, libPatch );
    
    std::cout << std::setw( 35 ) << std::right << "Application compiled with Bolt: " << "v" << appMajor << "." << appMinor << "." << appPatch << std::endl;
    std::cout << std::setw( 35 ) << std::right << "Bolt library compiled with Bolt: " << "v" << libMajor << "." << libMinor << "." << libPatch << std::endl;
    
    return 0;
}
