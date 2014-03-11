// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
#pragma once
/**********************************************************************************
* amptest\dpctest_lib.h
* 
* Attaches the appropriate dpctest library.
**********************************************************************************/

#ifndef AMP_TEST_LIB_BUILD

    #if defined(_MT) && !defined(_DEBUG) && !defined(_DLL)
        #pragma comment(lib, "libamptest-MT.lib")
    #elif defined(_MT) && !defined(_DEBUG) && defined(_DLL)
        #pragma comment(lib, "libamptest-MD.lib")
    #elif defined(_MT) && defined(_DEBUG) && !defined(_DLL)
        #pragma comment(lib, "libamptest-MTd.lib")
    #elif defined(_MT) && defined(_DEBUG) && defined(_DLL)
        #pragma comment(lib, "libamptest-MDd.lib")
    #else
        #error DpcTest library requires /MT, /MD, /MTd or /MDd compiler switch.
    #endif

#endif

