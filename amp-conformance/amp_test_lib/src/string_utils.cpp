// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
#include <amptest/string_utils.h>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <codecvt>
#include <cstdlib>
#include <cstring>

namespace Concurrency
{
	namespace Test
	{
		std::string convert_to_utf8(const std::wstring& str) { return convert_to_utf8(str.c_str()); };

		/// <summary>Converts a null terminated wchar_t string to UTF-8.</summary>
		/// <param name="str">null-terminated wchar_t string</param>
		/// <return>null-terminated UTF-8 encoded char*.  If str is null, an empty std::string will be returned.</return>
		std::string convert_to_utf8(const wchar_t* str)
		{
			// return an empty string for the null or empty case.
			if ((str == nullptr) || (wcslen(str) == 0)) { return std::string(); }
			
			std::wstring wstr(str);
			std::wstring_convert<std::codecvt_utf8<wchar_t> > to_utf8;
			std::string utf8_str = to_utf8.to_bytes(wstr);
			
			return utf8_str;
		}
		
		std::wstring convert_to_wchar_t(const std::string& str)
		{

#pragma warning(disable:4996)
			size_t wlen = mbstowcs(nullptr, str.c_str(), 0);
			if (wlen == (size_t)-1)
			{
				return std::wstring(L"[convert_to_wchar_t() failed]");
			}
			
			std::unique_ptr<wchar_t[]> wstr(new wchar_t[wlen + 1]);
			memset(wstr.get(), 0, sizeof(wchar_t) * (wlen + 1));
			
			wlen = mbstowcs(wstr.get(), str.c_str(), wlen);
#pragma warning(default:4996)
	
			return wstr.get();
		}
		
		std::wstring convert_to_utf16(const std::string& str)
		{
			return convert_to_wchar_t(str);
		}
		
		std::wstring convert_to_utf16(const char* str)
		{
			return convert_to_wchar_t(str);
		}

		std::string AMP_TEST_API remove_quote(const std::string& str)
		{
			if(str.length() >= 2 &&
				*str.begin() == '\"' &&
				*(--str.end()) == '\"')
			{
				return std::string(++str.begin(), --str.end());
			}

			return str;
		}

		std::wstring AMP_TEST_API remove_quote(const std::wstring& str)
		{
			if(str.length() >= 2 &&
				*str.begin() == L'\"' &&
				*(--str.end()) == L'\"')
			{
				return std::wstring(++str.begin(), --str.end());
			}

			return str;
		}
	}
}

