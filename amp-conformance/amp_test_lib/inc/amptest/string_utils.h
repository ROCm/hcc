// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
#pragma once
#include <algorithm>
#include <locale>
#include <string>
#include <utility>
#include <amptest/platform.h>

namespace Concurrency
{
	namespace Test
	{
		#pragma region Unicode Conversions & Helper Methods

		/// <summary>Removes '\"' from beginning and end of a string if they exist.  Otherwise, str is returned.</summary>
		std::string  AMP_TEST_API remove_quote(const std::string& str);
		std::wstring AMP_TEST_API remove_quote(const std::wstring& str);

		/// <summary>Trim leading and trailing whitespace from a string</summary>
		/// <remarks>World Readiness: depends on the std::locale::global setting (defaults to the "C" locale)</remarks>
		template<typename T> std::basic_string<T> trim(const std::basic_string<T>& str);

		/// Converts a null terminated wchar_t* string to a UTF-8 encoded std::string
		std::string AMP_TEST_API convert_to_utf8(const wchar_t* str);
		std::string AMP_TEST_API convert_to_utf8(const std::wstring& str);

		/// Converts a multi-byte encoded string to wchar_t encoded string.
		/// <remarks>Note: this function is not properly named.</remarks>
		std::wstring AMP_TEST_API convert_to_wchar_t(const std::string& str);
		std::wstring AMP_TEST_API convert_to_utf16(const std::string& str);
		std::wstring AMP_TEST_API convert_to_utf16(const char* str);

		/// <summary>Trim leading and trailing whitespace from a string</summary>
		/// <remarks>World Readiness: depends on the std::locale::global setting (defaults to the "C" locale)</remarks>
		template<typename T>
		std::basic_string<T> trim(const std::basic_string<T>& str)
		{
			auto start = std::find_if(str.begin(), str.end(), [](T c)-> bool { return !std::isspace(c, std::locale()); });
			auto end = std::find_if(str.rbegin(), str.rend(), [](T c)-> bool { return !std::isspace(c, std::locale()); }).base();
			if(start < end)
			{
				return std::basic_string<T>(start, end);
			}
			else
			{
				return std::basic_string<T>();
			}
		}
	}
}

