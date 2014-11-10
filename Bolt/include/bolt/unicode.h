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

#pragma once
#if !defined( BOLT_UNICODE_H )
#define BOLT_UNICODE_H

#include <string>
#include <iostream>
#include <sstream>

//	These macros help linux cope with the conventions of windows tchar.h file.
#if defined( _WIN32 )
	#include <tchar.h>
#else
    // Relax for clang
	#if 1//defined( __GNUC__ )
		typedef char TCHAR;
		typedef char _TCHAR;
		#define _tmain main

		#if defined( UNICODE )
			#define _T(x)	L ## x
		#else
			#define _T(x)	x
		#endif 
	#endif
#endif

namespace bolt
{
#if defined( _UNICODE )
	typedef std::wstring		tstring;
	typedef std::wstringstream	tstringstream;
	typedef std::wifstream		tifstream;
	typedef std::wofstream		tofstream;
	typedef std::wfstream		tfstream;
	typedef std::wostream		tstream;
	static std::wostream&	tout	= std::wcout;
	static std::wostream&	terr	= std::wcerr;
	static std::wostream&	tlog	= std::wclog;
#else
	typedef std::string tstring;
	typedef std::stringstream tstringstream;
	typedef std::ifstream		tifstream;
	typedef std::ofstream		tofstream;
	typedef std::fstream		tfstream;
	typedef std::ostream		tstream;
	static std::ostream&	tout	= std::cout;
	static std::ostream&	terr	= std::cerr;
	static std::ostream&	tlog	= std::clog;
#endif 
}

#endif