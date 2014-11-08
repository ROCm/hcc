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

#include <bolt/miniDump.h>
#ifdef UNICODE
#include <tchar.h>
#endif

//  Good info used to make this minidump object can be found at: http://www.debuginfo.com/articles/effminidumps.html
using namespace bolt;

//	Initialize static data to NULL
FNMINIDUMPWRITEDUMP miniDumpSingleton::fnMiniDumpWriteDump = NULL;
tstring miniDumpSingleton::exePath;
tstring miniDumpSingleton::exeName;

BOOL CALLBACK MiniDumpCallBack( PVOID pParam, const PMINIDUMP_CALLBACK_INPUT pInput, PMINIDUMP_CALLBACK_OUTPUT pOutput )
{
    BOOL retValue = FALSE;

    if( ( pInput == NULL ) || ( pOutput == NULL ) )
        return retValue;

    switch( pInput->CallbackType )
    {
        case ModuleCallback:
            retValue = TRUE;
            break;
        case ThreadCallback:
            retValue = TRUE;
            break;
        case ThreadExCallback:
            retValue = TRUE;
            break;
        case IncludeThreadCallback:
            retValue = TRUE;
            break;
        case IncludeModuleCallback:
            retValue = TRUE;
            break;
        case MemoryCallback:
            break;
        case CancelCallback:
            break;
    }

    return retValue;
}

//	miniDumpSingleton::miniDumpSingleton() is a singleton constructor, which guarantees that the
//	logic included here is only executed once
miniDumpSingleton::miniDumpSingleton( minidumpVerbosity verbosity ): hDbgHelp( NULL ), topFilterFunc( NULL ), m_Verbosity( verbosity )
{
    // Since this code is written to be loaded in a library, we take a conservative approach to replacing
    //	the exception filter and intercepting exceptions.  As an .exe may have their own minidump logic that
    //	we don't want to interfere with, we only load dbghelp.dll if it has been copied side-by-side with
    //	the application.
    TCHAR osPath[ MAX_PATH ];
    if( ::GetModuleFileName( NULL, osPath, MAX_PATH ) )
    {
        tstring thisPath( osPath );
        tstring::size_type pos = thisPath.find_last_of( _T( "\\" ) );

        if( pos != tstring::npos )
        {
            exePath	= thisPath.substr( 0, pos );
            exeName = thisPath.substr( pos, bolt::tstring::npos );
            tstring newPath = exePath + _T( "\\dbghelp.dll" );

            if( m_Verbosity )
            {
                tout << _T( "<<<Bolt.MinidumpGenerator>>> Attempting to load DbgHelp.dll first from => " ) << newPath << std::endl;
            }
            hDbgHelp = ::LoadLibrary( newPath.c_str( ) );
            if( hDbgHelp == NULL )
            {
                if( m_Verbosity )
                {
                    tout << _T( "<<<Bolt.MinidumpGenerator>>> Attempting to load DbgHelp.dll from system paths" ) << std::endl;
                }
                hDbgHelp = ::LoadLibrary( _T( "dbghelp.dll" ) );
            }
        }
    }

    if( hDbgHelp != NULL )
    {
        if( m_Verbosity )
        {
            tout << _T( "<<<Bolt.MinidumpGenerator>>> Successfully loaded DbgHelp.dll" ) << std::endl;
        }

        fnMiniDumpWriteDump = reinterpret_cast< FNMINIDUMPWRITEDUMP >( ::GetProcAddress( hDbgHelp, "MiniDumpWriteDump" ) );
        if( (fnMiniDumpWriteDump == NULL) && m_Verbosity )
        {
            terr << _T( "<<<Bolt.MinidumpGenerator>>> Could not retrieve the address of callable function MiniDumpWriteDump" ) << std::endl;
        }

        // Set our Filter function override, and save off the previous filter function
        topFilterFunc = ::SetUnhandledExceptionFilter( ExceptionFilter );
    }

}

miniDumpSingleton::~miniDumpSingleton( )
{ 
    fnMiniDumpWriteDump	= NULL;
    if( m_Verbosity )
    {
        tout << _T( "<<<Bolt.MinidumpGenerator>>> Inside miniDumpSingleton::~miniDumpSingleton\n" ) << fnMiniDumpWriteDump << std::endl;
    }
};

LONG WINAPI miniDumpSingleton::ExceptionFilter( PEXCEPTION_POINTERS pExceptionInfo )
{
    long exceptionReturn = EXCEPTION_CONTINUE_SEARCH;

    miniDumpSingleton& mds = enableMiniDumps( );

    if( mds.m_Verbosity )
    {
        tout << _T( "<<<Bolt.MinidumpGenerator>>> Entering miniDumpSingleton::ExceptionFilter: " ) <<  fnMiniDumpWriteDump << std::endl;
    }

    if( fnMiniDumpWriteDump )
    {
        //  Create the base name of the minidump file name that we want to generate, including the last period
        tstring::size_type pos = exeName.find_last_of( _T( "." ) );
        tstring dmpBaseFileName = exeName.substr( 0, pos + 1 );
        tstring miniDumpFilePath;

        unsigned int nAttempts = 0;
        HANDLE hDmpFile = INVALID_HANDLE_VALUE;
        DWORD gle = ERROR_FILE_EXISTS;

        //  Keep looping until we can find a filename that we have not used yet; if we can not find a suitiable file name within
        //  100 attempts, abort
        while( (hDmpFile == INVALID_HANDLE_VALUE) && (gle == ERROR_FILE_EXISTS) && (nAttempts < 100) )
        {
            //  Append an increasing digit to the file name to attempt to find a unique filename
            tstring dmpTmpFileName = dmpBaseFileName;
            tstringstream ssTmp;
            ssTmp << nAttempts;
            dmpTmpFileName += ssTmp.str( );

            dmpTmpFileName += _T( ".dmp" );
            miniDumpFilePath = exePath + dmpTmpFileName;

            //	Attempt to create a file right next to the application, to make it easy to find
            hDmpFile = ::CreateFile( miniDumpFilePath.c_str( ), 
                                    GENERIC_WRITE,
                                    FILE_SHARE_READ,
                                    NULL,
                                    CREATE_NEW,
                                    FILE_ATTRIBUTE_NORMAL,
                                    NULL );

            gle = ::GetLastError( );
            ++nAttempts;
        }

        if( (hDmpFile == INVALID_HANDLE_VALUE) || (nAttempts == 100))
        {
            if( mds.m_Verbosity )
            {
                tout << _T( "<<<Bolt.MinidumpGenerator>>> Failed to create output file at => " ) << miniDumpFilePath << std::endl;
            }

            //	Couldn't open a file right next to the application
            //	Attempt to create the dmp file in the system temporary directory.
            //	If no temp path is defined, abort
            TCHAR tmpPath[ MAX_PATH ];
            if( GetTempPath( MAX_PATH, tmpPath ) == 0 )
                return (mds.topFilterFunc)( pExceptionInfo );

            nAttempts = 0;
            hDmpFile = INVALID_HANDLE_VALUE;
            gle = ERROR_FILE_EXISTS;

            //  Keep looping until we can find a filename that we have not used yet; if we can not find a suitiable file name within
            //  100 attempts, abort
            while( (hDmpFile == INVALID_HANDLE_VALUE) && (gle == ERROR_FILE_EXISTS) && (nAttempts < 100) )
            {
                //  Append an increasing digit to the file name to attempt to find a unique filename
                tstring dmpTmpFileName = dmpBaseFileName;
                tstringstream ssTmp;
                ssTmp << nAttempts;
                dmpTmpFileName += ssTmp.str( );

                dmpTmpFileName += _T( ".dmp" );
                miniDumpFilePath = tmpPath + dmpTmpFileName;

                //	Attempt to create a file right next to the application, to make it easy to find
                hDmpFile = ::CreateFile( miniDumpFilePath.c_str( ), 
                                        GENERIC_WRITE,
                                        FILE_SHARE_READ,
                                        NULL,
                                        CREATE_NEW,
                                        FILE_ATTRIBUTE_NORMAL,
                                        NULL );

                gle = ::GetLastError( );
                ++nAttempts;
            }

            if( (hDmpFile == NULL ) || (hDmpFile == INVALID_HANDLE_VALUE) )
            {
                //	Can't create a .dmp file in 2 locations, we abort our attempt to create a minidump of exception
                if( mds.m_Verbosity )
                {
                    tout << _T( "<<<Bolt.MinidumpGenerator>>> Failed to create output file at... " ) << miniDumpFilePath << std::endl;
                    tout << _T( "<<<Bolt.MinidumpGenerator>>> Aborting miniDump generation for failure to create file at 2 locations " ) << std::endl;
                }
                //  Be a good citizen, and pass the exception to the previously registered filter
                return (mds.topFilterFunc)( pExceptionInfo );
            }
            else
            {
                tout << _T( "<<<Bolt.MinidumpGenerator>>> Creating minidump at => " ) << miniDumpFilePath << std::endl;
            }
        }
        else
        {
            tout << _T( "<<<Bolt.MinidumpGenerator>>> Creating minidump at => " ) << miniDumpFilePath << std::endl;
        }

        MINIDUMP_EXCEPTION_INFORMATION mdei;

        mdei.ThreadId			= ::GetCurrentThreadId( );
        mdei.ExceptionPointers	= pExceptionInfo;
        mdei.ClientPointers		= FALSE;

        MINIDUMP_CALLBACK_INFORMATION	mdci;

        mdci.CallbackParam		= 0;
        mdci.CallbackRoutine	= MiniDumpCallBack;

        MINIDUMP_TYPE mdt = static_cast< MINIDUMP_TYPE >( MiniDumpWithPrivateReadWriteMemory 
                                                            | MiniDumpWithDataSegs
                                                            | MiniDumpWithHandleData
                                                            | MiniDumpWithFullMemoryInfo
                                                            | MiniDumpWithThreadInfo
                                                            | MiniDumpWithUnloadedModules );

        BOOL miniDumpOK = fnMiniDumpWriteDump( ::GetCurrentProcess( ), ::GetCurrentProcessId( ), hDmpFile, mdt, &mdei, NULL, &mdci );

        if( miniDumpOK )
        {
            exceptionReturn = EXCEPTION_EXECUTE_HANDLER;
            if( mds.m_Verbosity )
            {
                tout << _T( "<<<Bolt.MinidumpGenerator>>> fnMiniDumpWriteDump successfully called; minidump successfully created" ) << std::endl;
            }
        }

        ::CloseHandle( hDmpFile );
    }

    //  We have dumped our minidump, we are done
    //  Be a good citizen, and pass the exception to the previously registered filter to handle
    return (mds.topFilterFunc)( pExceptionInfo );
}
