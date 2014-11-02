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



//#include "stdafx.h"  // not present in the BoltCL dir, but don't really need it 




#include <iostream>
#include <fstream>
#include <streambuf>
#include <direct.h>  //windows CWD for error message
#include <tchar.h>

//	TODO:  Find an appropriate place for this to live
//	Typedefs to support unicode and ansii compilation
#if defined( _UNICODE )
	typedef std::wstring		tstring;
	typedef std::wstringstream	tstringstream;
	typedef std::wifstream		tifstream;
	typedef std::wofstream		tofstream;
	typedef std::wfstream		tfstream;
	static std::wostream&	tout	= std::wcout;
	static std::wostream&	terr	= std::wcerr;
#else
	typedef std::string tstring;
	typedef std::stringstream tstringstream;
	typedef std::ifstream		tifstream;
	typedef std::ofstream		tofstream;
	typedef std::fstream		tfstream;
	static std::ostream&	tout	= std::cout;
	static std::ostream&	terr	= std::cerr;
#endif 

namespace oclcpp {

	std::string fileToString(const std::string &fileName)
	{
		std::ifstream infile (fileName);
		if (infile.fail() ) {
#if defined( _WIN32 )
			TCHAR osPath[ MAX_PATH ];

			//	If loading the .cl file fails from the specified path, then make a last ditch attempt (purely for convenience) to find the .cl file right to the executable,
			//	regardless of what the CWD is
			//	::GetModuleFileName( ) returns TCHAR's (and we define _UNICODE for windows); but the fileName string is char's, 
			//	so we needed to create an abstraction for string/wstring
			if( ::GetModuleFileName( NULL, osPath, MAX_PATH ) )
			{
				tstring thisPath( osPath );
				tstring::size_type pos = thisPath.find_last_of( _T( "\\" ) );

				tstring newPath;
				if( pos != tstring::npos )
				{
					tstring exePath	= thisPath.substr( 0, pos + 1 );	// include the \ character

					//	Narrow to wide conversion should always work, but beware of wide to narrow!
					tstring convName( fileName.begin( ), fileName.end( ) );
					newPath = exePath + convName;
				}

				infile.open( newPath.c_str( ) );
			}
#endif
			if (infile.fail() ) {
				TCHAR cCurrentPath[FILENAME_MAX];
				if (_tgetcwd(cCurrentPath, sizeof(cCurrentPath) / sizeof(TCHAR))) {
					tout <<  _T( "CWD=" ) << cCurrentPath << std::endl;
				};
				std::cout << "error: failed to open file '" << fileName << std::endl;
				throw;
			} 
		}

		std::string str((std::istreambuf_iterator<char>(infile)),
			std::istreambuf_iterator<char>());
		return str;
	};



	cl::Kernel compileFunctor(const std::string &kernelCodeString, const std::string kernelName)
	{
		cl::Program mainProgram(kernelCodeString, false);
		try
		{
			mainProgram.build("-x clc++");

		} catch(cl::Error e) {
			std::cout << "Code         :\n" << kernelCodeString << std::endl;
			std::cout << "Build Status: " << mainProgram.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(cl::Device::getDefault()) << std::endl;
			std::cout << "Build Options:\t" << mainProgram.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(cl::Device::getDefault()) << std::endl;
			std::cout << "Build Log:\t " << mainProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cl::Device::getDefault()) << std::endl;
			throw e;
		}

		return cl::Kernel(mainProgram, kernelName.c_str());
	}



	 void constructAndCompile(cl::Kernel *masterKernel, const std::string &apiName, const std::string instantiationString, std::string userCode, std::string valueTypeName,  std::string functorTypeName) {

		//FIXME, when this becomes more stable move the kernel code to a string in bolt.cpp
		// Note unfortunate dependency here on relative file path of run directory and location of boltcl dir.
		std::string templateFunctionString = boltcl::fileToString( apiName + "_kernels.cl"); 

		std::string codeStr = userCode + "\n\n" + templateFunctionString +   instantiationString;

		if (0) {
			std::cout << "Compiling: '" << apiName << "'" << std::endl;
			std::cout << "ValueType  ='" << valueTypeName << "'" << std::endl;
			std::cout << "FunctorType='" << functorTypeName << "'" << std::endl;

			std::cout << "code=" << std::endl << codeStr;
		}

		*masterKernel = boltcl::compileFunctor(codeStr, apiName + "Instantiated");
	};



};

