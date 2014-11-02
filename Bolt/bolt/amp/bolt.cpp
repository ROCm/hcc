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


#include <iostream>
#include <fstream>
#include <streambuf>
#ifdef _WIN32
#include <direct.h>  //windows CWD for error message
#endif
#ifdef UNICODE
#include <tchar.h>
#endif
#include <algorithm>
#include <vector>
#include <set>

#include "bolt/amp/bolt.h"
#include "bolt/unicode.h"


namespace bolt {
namespace amp {

void getVersion( unsigned int& major, unsigned int& minor, unsigned int& patch )
{
    major	= static_cast<unsigned int>( BoltVersionMajor );
    minor	= static_cast<unsigned int>( BoltVersionMinor );
    patch	= static_cast<unsigned int>( BoltVersionPatch );
}

// for printing errors
const std::string hr = "###############################################################################";
const std::string es = "ERROR: ";



/*
void wait(const bolt::amp::control &ctl, ::amp::Event &e) 
{
    
};
*/
    

}; //namespace bolt::amp
}; // namespace bolt
