// University of Illinois/NCSA
// Open Source License
// 
// Copyright (c) 2013, Advanced Micro Devices, Inc.
// All rights reserved.
// 
// Developed by:
// 
//     Runtimes Team
// 
//     Advanced Micro Devices, Inc
// 
//     www.amd.com
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal with
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is furnished to do
// so, subject to the following conditions:
// 
//     * Redistributions of source code must retain the above copyright notice,
//       this list of conditions and the following disclaimers.
// 
//     * Redistributions in binary form must reproduce the above copyright notice,
//       this list of conditions and the following disclaimers in the
//       documentation and/or other materials provided with the distribution.
// 
//     * Neither the names of the LLVM Team, University of Illinois at
//       Urbana-Champaign, nor the names of its contributors may be used to
//       endorse or promote products derived from this Software without specific
//       prior written permission.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH THE
// SOFTWARE.
//===----------------------------------------------------------------------===//

#ifndef FILEUTILS_H
#define FILEUTILS_H
#include <string>
#include <iostream>
#include <fstream>

	char *readFile(std::string source_filename, size_t& size)
	{
		FILE *fp = ::fopen( source_filename.c_str(), "rb" );
		long length;
		size_t offset = 0;
		char *ptr;
		
		if (!fp) {
			return NULL;
		}
		
		// obtain file size.
		::fseek (fp , 0 , SEEK_END);
		length = ::ftell (fp);
		::rewind (fp);
		
		ptr = reinterpret_cast<char*>(malloc(offset + length + 1));
		if (length != fread(&ptr[offset], 1, length, fp))
		{
			free(ptr);
			return NULL;
		}
		
		ptr[offset + length] = '\0';
		size = offset + length;
		::fclose(fp);
		return ptr;
	}
	
	//later move this to the helper file
	void writeToFile(const void *buf, size_t length, char* filename)
	{
		FILE *fp;
		fp=fopen(filename, "wb");
		
		size_t bytes = fwrite(buf, sizeof(char), length, fp);
		
		fclose(fp);
	}

    // substring replacement within a string
	void replaceAll( std::string &s, const std::string &search, const std::string &replace ) {
		for( size_t pos = 0; ; pos += replace.length() ) {
			// Locate the substring to replace
			pos = s.find( search, pos );
			if( pos == std::string::npos ) break;
			// Replace by erasing and inserting
			s.erase( pos, search.length() );
			s.insert( pos, replace );
		}
	}


#endif //FILEUTILS_H
