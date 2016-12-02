//--------------------------------------------------------------------------------------
// File: logging.cpp
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

#include <sstream>
#include <iostream>
#include <amptest/logging.h>
#include <amptest/string_utils.h>
#include <amptest/context.h>

namespace Concurrency
{
	namespace Test
	{

		namespace details {

            ///<summary>A basic_streambuf implementation that doesn't log anything</summary>
            template <class cT, class traits = std::char_traits<cT> >
            class noop_streambuf : public std::basic_streambuf<cT, traits> {

                typename traits::int_type overflow(typename traits::int_type c)
                {
                    return traits::not_eof(c); // indicate success
                }
            };

            ///<summary>A basic_ostream implementation that doesn't log anything</summary>
            template <class cT, class traits = std::char_traits<cT> >
            class basic_noop_ostream: public std::basic_ostream<cT, traits> {

                public:
                    basic_noop_ostream():
                        std::basic_ios<cT, traits>(&m_sbuf),
                        std::basic_ostream<cT, traits>(&m_sbuf)
                    {
                        this->init(&m_sbuf);
                    }

                private:
                    noop_streambuf<cT, traits> m_sbuf;
            };

            static basic_noop_ostream<char> noop_ostream;

            // Controls verbosity of wrapper, default is Info level
            LogType g_verbose = LogType::Info;

			static inline std::ostream& get_raw_log_stream(LogType type) {

				switch (type)
				{
				case LogType::Info:
					return amptest_context.get_raw_stdout_stream();
				case LogType::Warning:
				case LogType::Error:
					return amptest_context.get_raw_stderr_stream();
				case LogType::Silent: return noop_ostream;
				default: throw new std::invalid_argument("Invalid LogType argument value.");
				}
			}

			/// Streams the current timestamp and the "AMPTEST:" line prefix.
			static inline void stream_line_prefix(std::ostream& log_stream, LogType type) {
				if(type == LogType::Silent) return;

				// Compose the prefix as a string so it won't get broken up
				std::stringstream ss;
				ss << get_timestamp(false) << " AMPTest: ";
				if(type == LogType::Warning) {
					ss << "Warning: ";
				} else if(type == LogType::Error) {
					ss << "Error: ";
				}

				log_stream << ss.str();
			}

		    void AMP_TEST_API amptest_initialize_logging() {
				std::ostream& info_log = get_raw_log_stream(LogType::Info);
				info_log.precision(12);	// To help display floats more accurately
				info_log << std::boolalpha;	// Make bool values print out as true/false rather than 1/0
				//info_log << std::unitbuf;	// Force flushing after each insertion

				std::ostream& err_log = get_raw_log_stream(LogType::Error);
				err_log.precision(12);	// To help display floats more accurately
				err_log << std::boolalpha;	// Make bool values print out as true/false rather than 1/0
				//err_log << std::unitbuf;	// Force flushing after each insertion

				std::ostream& warn_log = get_raw_log_stream(LogType::Warning);
				warn_log.precision(12);	// To help display floats more accurately
				warn_log << std::boolalpha;	// Make bool values print out as true/false rather than 1/0
				//warn_log << std::unitbuf;	// Force flushing after each insertion
			}

		}

		/// Gets the current timestamp from the system and formats it to a string
		/// The format used has the timestamp surrounded with square brackets (e.g. "[...]").
		std::string AMP_TEST_API get_timestamp(bool include_date) {
			static const unsigned int MAX_STR_LEN = 32; // Plenty of space for either strings but also small enough to keep on the stack
			time_t this_t = time( NULL );
			if (this_t == (time_t)-1)
			{
			   return std::string("[Unknown: time() returned -1]");
			}

#pragma warning(disable:4996)
			tm* this_tm = localtime(&this_t);
#pragma warning(default:4996)

			// Determine which format to use
			const char* format_cstr = include_date ? "[%m/%d/%Y %H:%M:%S]" : "[%H:%M:%S]";

			char time_cstr[MAX_STR_LEN] = {0};
			if(0 != strftime(time_cstr, MAX_STR_LEN, format_cstr, this_tm)) { // strftime returns 0 on failure
			   return std::string(time_cstr);
			}

			// Always return something
			return std::string("[Unknown]");
		}

		#pragma region ostream& operator<<() overloads for wchar_t & std::wstring
		/// C++-style streaming operators.  Convert output to a UTF-8 encoded byte stream.
		std::ostream& operator<<(std::ostream& os, const std::wstring& str)
		{
			os << Concurrency::Test::convert_to_utf8(str.c_str());
			return os;
		}

		/// C++-style streaming operators.  Convert output to a UTF-8 encoded byte stream.
		std::ostream& operator<<(std::ostream& os, const wchar_t* str)
		{
			os << Concurrency::Test::convert_to_utf8(str);
			return os;
		}
		#pragma endregion

		std::ostream& AMP_TEST_API Log(LogType type, bool print_line_prefix)
        {
            if (type > details::g_verbose)
            {
                // message does not have required verbosity
                return details::noop_ostream;
            }

			std::ostream& log_stream = details::get_raw_log_stream(type);

			if(print_line_prefix) {
				details::stream_line_prefix(log_stream, type);
			}

			return log_stream;
        }

		void AMP_TEST_API SetVerbosity(LogType level)
        {
            details::g_verbose = level;
        }

        LogType AMP_TEST_API GetVerbosity()
        {
            return details::g_verbose;
        }

		void AMP_TEST_API Log_writeline(LogType type, const char *msg, ...) {
            if (type > details::g_verbose) {
                return; // message does not have required verbosity
            }

            va_list args;
            va_start(args, msg);

#pragma warning(disable:4996)
			int len = vsnprintf(nullptr, 0, msg, args);
#pragma warning(default:4996)

			va_end(args);

			va_start(args, msg);

			std::unique_ptr<char[]> c_msg(new char[len+1]);
			memset(c_msg.get(), 0, sizeof(char) * (len+1));
#pragma warning(disable:4996)
			int actual_len = vsnprintf(c_msg.get(), len+1, msg, args);
#pragma warning(default:4996)
			Log(type, true) << c_msg.get() << std::endl;

            va_end(args);

			if (len > actual_len)
			{
				// The code above should ensure this doesn't happen.  However, I'd prefer to fail fast if we
				// do ever see it.
				Log(LogType::Warning, true) << "The previous message was unexpectedly truncated" << std::endl;
				throw amptest_exception("Log_Writeline() message was unexpectedly truncated");
			}
        }

        void AMP_TEST_API Log_writeline(const char *msg, ...) {
            va_list args;
            va_start(args, msg);
			Log_writeline(LogType::Info, msg, args);
            va_end(args);
		}

		void AMP_TEST_API Log_writeline(LogType type) {
            if (type > details::g_verbose) {
                return; // message does not have required verbosity
            }

			Log(type, true) << std::endl;
		}

		std::string AMP_TEST_API get_type_name(const std::type_info& ti) {

			if(ti == typeid(std::string)) {
				return "string";
			}

			std::string tname = ti.name();

			static const std::string filters_to_rem[6] = {
				"class Concurrency::graphics::", "",	// This must be before the next line so they get removed first
				"class Concurrency::", "",
				"class std::", "std::", // For std:: types, just remove the word 'class' because we want to be able to distinguish between types like std::array and concurrency::array.
			};

			for(int i = 0; i < sizeof(filters_to_rem)/sizeof(filters_to_rem[0]); i+=2) {
				const std::string& to_rem = filters_to_rem[i];
				const std::string& replacement = filters_to_rem[i+1];
				for(size_t pos = tname.find(to_rem); pos != std::string::npos; pos = tname.find(to_rem)) {
					if(pos == 0)
						tname = replacement + tname.substr(pos + to_rem.length());
					else
						tname = tname.substr(0,pos) + replacement + tname.substr(pos + to_rem.length());
				}
			}

			return tname;
		}

		std::ostream& AMP_TEST_API LogStream()
        {
			return amptest_context.get_raw_stdout_stream();
        }
	}
}

