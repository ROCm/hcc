// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
#pragma once
/********************************************************************************************
* amptest\context.h
*
* Provides an interface for determining the context in which a test is running.
*
* The context is controlled via environment variables. This allows the greatest flexability
* when running multiple tests and for controlling the context via
* env.lst lines.
*
* Environment variables may have the following types (and values):
* 	bool - Accepted values: 1/true/TRUE=true; 0/false/FALSE=false
*
* The following are the supported environment variables:
* AMPTEST_ENABLE_DEBUGGING (bool, default=false)
*    When true, this setting will turn off all unhandled exception handling so as to enable
*    a debugger to be attached when/if an unexpected exception occurs.
********************************************************************************************/


// Attach the dpctest.lib
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>
namespace Concurrency {
    namespace Test {

		class amptest_context_t {
		private:
			std::shared_ptr<std::ofstream> _cerr_logfile_override;
			std::shared_ptr<std::ofstream> _cout_logfile_override;
			std::string _cerr_logfile_path;
			std::string _cout_logfile_path;

			std::map<std::string, std::string> _env_cache;
			bool _using_env_cache;

			int _argc;
			char** _argv;

		public:

			amptest_context_t();
			amptest_context_t(int argc, char** argv);

			std::string get_environment_variable(const std::string& name) const;
			bool get_environment_variable(const std::string& name, bool default_value) const;

			bool is_buffer_aliasing_forced() const;

			#pragma region log redirection to file
			/// Gets path to file used to redirect stderr or stdout.
			const std::string& get_stderr_logfile_path() const;
			const std::string& get_stdout_logfile_path() const;

			/// Sets the logfile path for STDERR or STDOUT.
			/// <remarks>
			/// if stdout_filename or stderr_filename are empty or are whitespace only, then
			/// the configuration will be ignored for that stream.
			///
			/// Logging to the console can be restored by calling close_logfiles().
			///</remarks>
			void set_stdout_logfile_path(const std::string& stdout_filename);
			void set_stderr_logfile_path(const std::string& stderr_filename);

			/// Gets the STDERR or STDOUT logfile.  If logging output has not been redirected
			/// to a file, nullptr will be returned.
			std::ostream& get_raw_stderr_stream() const;
			std::ostream& get_raw_stdout_stream() const;

			// Close logfiles and restore logging to the console streams
			void close_logfiles();

			#pragma endregion

			/// Gets whether the environment cache is being used
			bool using_environment_variable_cache() const;

			/// Once the load command is called environment variables will only be read from the cache.  To re-enable reading
			/// variables form the system, call destroy_environment_variable_cache().
			///
			/// Reading multiple environment files is not supported and will result in an amptest_exception being thrown.
			int load_environment_variable_cache_from_file(const std::string& file_path);

			/// Writes contents of the Environment Variable Cache to Log(LogType::Info, true) and returns the number of elements in the cache.
			///
			/// If the environment variable cache does not exist, then -1 will be returned.
			int dump_environment_variable_cache() const;

			/// Clears the environment cache and re-enables reading environment variables from the system.
			void destroy_environment_variable_cache();
		};

		/// The context for the currently running test.
		extern amptest_context_t amptest_context;			/// Gets or sets the execution mode.  This should be set by framework code only.
    }
}



