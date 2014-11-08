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
#if defined(_WIN32)

#ifndef MINIDUMP_H_
#define MINIDUMP_H_
#pragma once

#if defined( _WIN32 )
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
    #include <dbghelp.h>
#endif
#include <string>
#include "unicode.h"

// Function prototype for dynamically discovering dbghelp.dll!MiniDumpWriteDump( ).
typedef BOOL (WINAPI* FNMINIDUMPWRITEDUMP)(
    HANDLE hProcess,
    DWORD ProcessId,
    HANDLE hFile,
    MINIDUMP_TYPE DumpType,
    PMINIDUMP_EXCEPTION_INFORMATION ExceptionParam,
    PMINIDUMP_USER_STREAM_INFORMATION UserStreamParam,
    PMINIDUMP_CALLBACK_INFORMATION CallbackParam
    );

namespace bolt
{
    //	This class is not yet thread-safe (static variable creation is not thread safe); 
    //	the user must ensure that the singleton is used in a thread-safe manner.
    class miniDumpSingleton
    {
    public:
        enum minidumpVerbosity { noVerbose, Verbose };

        //  This sets up the exception filter and enables the generation of minidumps.
        static miniDumpSingleton& enableMiniDumps( minidumpVerbosity verbosity = noVerbose )
        {
            #if defined( _WIN32 )
                static miniDumpSingleton singleton( verbosity );
                return singleton;
            #endif
        }

        static LONG WINAPI ExceptionFilter( PEXCEPTION_POINTERS pExceptionInfo );

    private:
        HMODULE hDbgHelp;
        LPTOP_LEVEL_EXCEPTION_FILTER topFilterFunc;
        minidumpVerbosity   m_Verbosity;

        miniDumpSingleton( minidumpVerbosity verbosity );
        miniDumpSingleton( const miniDumpSingleton& );
        miniDumpSingleton& operator=( const miniDumpSingleton& );
        ~miniDumpSingleton( );

        static FNMINIDUMPWRITEDUMP fnMiniDumpWriteDump;
        static bolt::tstring exePath;
        static bolt::tstring exeName;
    };
}

#endif /* CLAMDBLAS_H_ */
#endif

