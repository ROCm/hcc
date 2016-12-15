//--------------------------------------------------------------------------------------
// File: logging.h
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this
// file except in compliance with the License.  You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR
// CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
//
// See the Apache Version 2.0 License for specific language governing permissions
// and limitations under the License.
//
//--------------------------------------------------------------------------------------
//

#pragma once

#include <amptest/platform.h>
#include <amptest/runall.h>
#include <iostream>
#include <iomanip>
#include <string>

namespace Concurrency
{
    namespace Test
    {
        enum class LogType
        {
            Silent = 0,
            Error = 1,
            Warning = 2,
            Info = 3
        };

		/// Gets the current timestamp from the system and formats it to a string
		/// The format used has the timestamp surrounded with square brackets (e.g. "[...]").
		std::string AMP_TEST_API get_timestamp(bool include_date);

		/// C++-style streaming operators.  Convert output to a UTF-8 encoded byte stream.
		std::ostream& operator<<(std::ostream& os, const wchar_t* str);

		/// C++-style streaming operators.  Convert output to a UTF-8 encoded byte stream.
		std::ostream& operator<<(std::ostream& os, const std::wstring& str);

        ///<summary>Logs the "logtype" and returns an ostream for further logging</summary>
        std::ostream& AMP_TEST_API Log(LogType type, bool print_line_prefix);
        std::ostream& AMP_TEST_API LogStream();

        inline std::ostream& Log(runall_result result) {
			if(result.get_is_pass() || result.get_is_no_value()) {
				return Log(LogType::Info, true);
			} else if(result.get_is_skip()) {
				return Log(LogType::Warning, true);
			} else { // Any of the fail states
				return Log(LogType::Error, true);
			}
		}

		///<summary>Logs the "logtype" and returns an wostream for further logging</summary>
        inline std::ostream& AMP_TEST_API WLog(LogType type = LogType::Info, bool print_line_prefix = true) { return Log(type, print_line_prefix); }

		void AMP_TEST_API Log_writeline(LogType type, const char *msg, ...);
        void AMP_TEST_API Log_writeline(const char *msg, ...);

		// Prints a new line character to the specified log.
        void AMP_TEST_API Log_writeline(LogType type);

        void AMP_TEST_API SetVerbosity(LogType level);
        LogType AMP_TEST_API GetVerbosity();

		/// Reports a pass/fail result.
		inline bool report_result(const std::string& description, const bool& passed) {
			Log(passed ? LogType::Info : LogType::Error, true)
				<< description << ": " << runall_result_name(passed) << std::endl << std::flush;
			return passed;
		}

		/// Reports a runall_result result.
		inline runall_result report_result(const std::string& description, const runall_result& result) {
			Log(result) << description << ": " << result << std::endl << std::flush;
			return result;
		}

		/// Reports a runall constant result.
		inline runall_result report_result(const std::string& description, const int& result) {
			return report_result(description, (runall_result)result);
		}

		/// Skips the test if a condition is not met by logging
		/// calling exit(runall_skip) if contition is false.
		inline void skip_if(const std::string& description, bool condition) {
			if(condition) {
				Log(LogType::Info, true) << description << ": Skipping..." << std::endl << std::flush;
				exit(runall_skip);
			}
		}

		// Gets a user-friendly name for the type specified by ti.
		std::string AMP_TEST_API get_type_name(const std::type_info& ti);

		/// Safely gets the type of an expression.
		/// This was created as a result of trying to write
		template <typename T>
		inline std::string get_type_name(const T& val) {
			return get_type_name(typeid(val));
		}

		/// Safely gets the type of an expression.
		/// This was created as a result of trying to write
		template <typename T>
		inline std::string get_type_name() {
			return get_type_name(typeid(T));
		}

		#pragma region format_as_code, format_as_hex, ...

		namespace details {
			template <typename T>
			inline void stream_value_as_code(std::ostream& s, const T& v) {
				// By default, we'll just use the type's stream operator
				s << v;
			}
			template <> inline void stream_value_as_code<unsigned int>(std::ostream& s, const unsigned int& v) { s << v << 'U'; }
			template <> inline void stream_value_as_code<float>  (std::ostream& s, const float& v)  { s << v << 'f'; }
			// We need to specially handle when T is char or unsigned char. When the value is (char)0 it causes the stream to be not printed out
			template <> inline void stream_value_as_code<char>    (std::ostream& s, const char& v)    { s << static_cast<int>(v); }
			template <> inline void stream_value_as_code<unsigned char>    (std::ostream& s, const unsigned char& v)    { s << static_cast<unsigned int>(v); }

			// Unfortunately, the signature of this function cannot be 'const T&'. Otherwise, I'd need to implement my own _Smanip
			template <typename T>
			inline void stream_value_as_code_wrapper(std::ios_base& s, T v) {
				stream_value_as_code<T>(dynamic_cast<std::ostream&>(s), v);
			}



			template <typename T>
			inline void stream_value_as_hex(std::ostream& s, const T& v) {
				s << "0x" << std::hex << v << std::dec;
			}
			template <> inline void stream_value_as_hex(std::ostream& s, const char& v) { stream_value_as_hex(s, static_cast<int>(v) & 0xFF); }
			template <> inline void stream_value_as_hex(std::ostream& s, const unsigned char& v) { stream_value_as_hex(s, static_cast<int>(v) & 0xFF); }
			template <> inline void stream_value_as_hex(std::ostream& s, const float& v) {
				unsigned int* ptr = reinterpret_cast<unsigned int*>(const_cast<float*>(&v));
				stream_value_as_hex<unsigned int>(s, *ptr);
			}
			template <> inline void stream_value_as_hex(std::ostream& s, const double& v) {
				unsigned long long* ptr = reinterpret_cast<unsigned long long*>(const_cast<double*>(&v));
				stream_value_as_hex<unsigned long long>(s, *ptr);
			}
			template <> inline void stream_value_as_hex(std::ostream& s, const runall_result& v) { stream_value_as_hex<int>(s, v.get_exit_code()); }

			// Unfortunately, the signature of this function cannot be 'const T&'. Otherwise, I'd need to implement my own _Smanip
			template <typename T>
			inline void stream_value_as_hex_wrapper(std::ios_base& s, T v) {
				stream_value_as_hex<T>(dynamic_cast<std::ostream&>(s), v);
			}
		}

		template <typename T>
		inline T format_as_code(const T& v) {
			return v;
		}

		template <typename T>
		inline T format_as_hex(const T& v) {
			return v;
		}

		#pragma endregion

		namespace details
		{
			void AMP_TEST_API amptest_initialize_logging();
		}

    }
}

// The following macros add support for using a code expression as the text to use for logging
// They also allow for not needing to specify the full namespace to the underlying function.

// Reports the result of the specified expression using the expression code as the description.
#define REPORT_RESULT(_testExp) Concurrency::Test::report_result(#_testExp, (_testExp))
#define SKIP_IF(_testExp) Concurrency::Test::skip_if(#_testExp, (_testExp))

// Streams the formatted expected and actual values to a common string representation
#define STREAM_EXPECTED_ACTUAL(_exp, _act) "expected = " << (_exp) << ", actual = " << (_act)


