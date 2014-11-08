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

#include <bolt/cl/bolt.h>
#include <bolt/cl/transform.h>
#include <algorithm>

#include "utils.h"

#include "common/stdafx.h"
#include "common/myocl.h"

// Instantiate host-side version of the saxpy functor:
#include "saxpy_functor.h"

std::string mycode = bolt::cl::fileToString("saxpy_functor.h");
BOLT_CREATE_TYPENAME(SaxpyFunctor);
BOLT_CREATE_CLCODE(SaxpyFunctor, mycode);



void readFromFileTest()
{
    std::string fName = __FUNCTION__ ;
    fName += ":";



    const int sz=2000;

    SaxpyFunctor s(100);
    std::vector<float> x(sz); // initialization not shown
    std::vector<float> y(sz); // initialization not shown
    std::vector<float> z(sz);
    bolt::cl::transform(x.begin(), x.end(), y.begin(), z.begin(), s);

    std::vector<float> stdZ(sz);
    std::transform(x.begin(), x.end(), y.begin(), stdZ.begin(), s);

    checkResults(fName, stdZ.begin(), stdZ.end(), z.begin());
};


int _tmain(int argc, _TCHAR* argv[])
{
    readFromFileTest();
}
