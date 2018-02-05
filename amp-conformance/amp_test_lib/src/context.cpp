// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
#include <amptest/context.h>
#include <amptest/logging.h>
#include <amptest/string_utils.h>
#include <sstream>

namespace Concurrency
{
	namespace Test
	{

		int AMP_TEST_API __initialize_globals()
		{
			details::amptest_initialize_logging();
			return 0;
		}

		amptest_context_t amptest_context; // The context
        int __dummy_initialize_globals = __initialize_globals();

		amptest_context_t::amptest_context_t() :
			_cerr_logfile_override(nullptr),
			_cout_logfile_override(nullptr),
			_cerr_logfile_path(""),
			_cout_logfile_path(""),
			_using_env_cache(false),
			_argc(0),
			_argv(nullptr)
        {
		}

		amptest_context_t::amptest_context_t(int argc, char** argv) :
			_cerr_logfile_override(nullptr),
			_cout_logfile_override(nullptr),
			_cerr_logfile_path(),
			_cout_logfile_path(),
			_using_env_cache(false),
			_argc(argc),
			_argv(argv)
        {
		}

		const std::string& amptest_context_t::get_stderr_logfile_path() const
		{
			return _cerr_logfile_path;
		}

		const std::string& amptest_context_t::get_stdout_logfile_path() const
		{
			return _cout_logfile_path;
		}

		void amptest_context_t::close_logfiles()
		{
			_cerr_logfile_path.clear();
			_cout_logfile_path.clear();

			_cerr_logfile_override.reset();
			_cout_logfile_override.reset();
		}

		void amptest_context_t::set_stderr_logfile_path(const std::string& stderr_filename)
		{
			if (stderr_filename.empty())
			{
				throw amptest_exception("set_stderr_logfile_path() stderr_filename was an empty string");
			}

			if (_cerr_logfile_path != stderr_filename)
			{
				_cerr_logfile_path = stderr_filename;

				// First determine if the new filestream already exists for stdout.
				// If so, just copy it.
				if (_cout_logfile_path == stderr_filename)
				{
					_cerr_logfile_override = _cout_logfile_override;
				}
				else
				{
					_cerr_logfile_override = std::make_shared<std::ofstream>(stderr_filename);
				}
			}
		}

		void amptest_context_t::set_stdout_logfile_path(const std::string& stdout_filename)
		{
			if (stdout_filename.empty())
			{
				throw amptest_exception("set_stdout_logfile_path() stdout_filename was an empty string");
			}

			if (_cout_logfile_path != stdout_filename)
			{
				_cout_logfile_path = stdout_filename;

				// First determine if the new filestream already exists for stderr.
				// If so, just copy it.
				if (_cerr_logfile_path == stdout_filename)
				{
					_cout_logfile_override = _cerr_logfile_override;
				}
				else
				{
					_cout_logfile_override = std::make_shared<std::ofstream>(stdout_filename);
				}
			}
		}

		std::ostream& amptest_context_t::get_raw_stderr_stream() const
		{
			return (_cerr_logfile_override.get() == nullptr) ? std::cerr : *_cerr_logfile_override;
		}

		std::ostream& amptest_context_t::get_raw_stdout_stream() const
		{
			return (_cout_logfile_override.get() == nullptr) ? std::cout : *_cout_logfile_override;
		}

		void check_wgetenv_error_code(int error_code)
		{
			// 0 is success.
			if (error_code == 0) { return; }
			else if (error_code == EINVAL) { throw amptest_exception("wgetenv_s() returned EINVAL"); }

			std::stringstream ss;
			ss << "wgetenv_s() returned unexpected error (error code = " << error_code << ")";
			throw amptest_exception(ss.str());
		}

		std::string amptest_context_t::get_environment_variable(const std::string& name) const {

			std::string val_str;

			if (!_using_env_cache)
			{
#pragma warning(disable:4996)
				char* env_value = getenv(name.c_str());
#pragma warning(default:4996)

				if (env_value != nullptr)
				{
					val_str = env_value;
				}
			}
			else
			{
				auto val = _env_cache.find(name);
				if (val != _env_cache.end())
				{
					val_str = val->second;
				}
			}

			return val_str;
		}

		bool amptest_context_t::get_environment_variable(const std::string& name, bool default_value) const {
			std::string val_str = get_environment_variable(name);

			if(val_str.empty()) {
				return default_value;
			} else if(val_str == "1" || val_str == "true" || val_str == "TRUE") {
				return true;
			} else if(val_str == "0" || val_str == "false" || val_str == "FALSE") {
				return false;
			}

			// Unknown value, throw exception and exit
            std::stringstream ss;
			ss << "Environment variable " << name << " has invalid bool value '" << val_str << "'. Accepted values: 1,0,true,TRUE,false,FALSE.";
			std::string errmsg = ss.str();
			throw amptest_exception(errmsg.c_str());
		}

		bool amptest_context_t::is_buffer_aliasing_forced() const {

			return get_environment_variable("CPPAMP_FORCE_ALIASED_SHADER", false);
		}

		int amptest_context_t::load_environment_variable_cache_from_file(const std::string& file_path)
		{
			int count = 0;

			// Open file to import environment variables.  Exceptions are not being enabled, since hitting EOF will also set/throw the failbit (below).
			std::ifstream infile;
			infile.open(file_path);

			if (infile.fail() == 1)
			{
				Log(LogType::Error, true) << "environment variable cache file not found" << std::endl;
				return count;
			}

			if (!_using_env_cache)
			{
				_using_env_cache = true;
			}
			else
			{
				throw amptest_exception("amptest_context_t::load_environment_variable_cache_from_file() cache already exists and does not support multiple loads");
			}

			while(infile.is_open() && !infile.eof())
			{
				std::string in;
				std::getline(infile, in);

				size_t index = in.find_first_of('=');
				if (index != std::string::npos)
				{

					std::string key = in.substr(0, index);
					std::string value;

					// If '=' is not the last character, then do the substr() operation and trim whitespace.
					if (in.size() > (index + 1) )
					{
						value = trim(in.substr(index+1));
					}

					// The last entry always wins...
					auto it = _env_cache.find(key);
					if (it != _env_cache.end())
					{
						if (!value.empty())
						{
							it->second = value;
						}
						else
						{
							// If value is empty, treat this as an "set <var>=", which removes the
							// environment variable.
							--count;
							_env_cache.erase(it);
						}
					}
					else if (!value.empty())
					{
						++count;
						_env_cache.insert(std::make_pair(key, value));
					}
				}
			}

			return count;
		}

		bool amptest_context_t::using_environment_variable_cache() const
		{
			return _using_env_cache;
		}

		int amptest_context_t::dump_environment_variable_cache() const
		{
			if (!_using_env_cache)
			{
				Log(LogType::Info, true) << "Environment Variable Cache is not being used" << std::endl;
				return 0;
			}

			int count = 0;
			Log(LogType::Info, true) << "Environment Variable Cache: dumping entries" << std::endl;
			std::for_each(_env_cache.begin(), _env_cache.end(), [&](std::pair<std::string, std::string> value)
			{
				Log(LogType::Info, true) << "    " << value.first << "=" << value.second <<std::endl;
				++count;
			});

			return count;
		}

		void amptest_context_t::destroy_environment_variable_cache()
		{
			_using_env_cache = false;
			_env_cache.clear();
		}
	}
}

