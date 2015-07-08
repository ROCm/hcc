// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
#pragma once
/**********************************************************************************
* runall.h
**********************************************************************************/

#include <amp.h>
#include <string>
#include <amptest/platform.h>

using namespace concurrency;

// Note: These have been moved out of the Concurrency::Test namespace.

// runall specific test return values
static const int runall_pass   =   0;
static const int runall_fail   =   1;
static const int runall_skip   =   2;
static const int runall_cascade_fail = 3;
static const int runall_no_value     = 4;


/// Returns a friendly name for the runall result.
std::string AMP_TEST_API runall_result_name(int result);
/// Returns a friendly name for the runall result.
std::string AMP_TEST_API runall_result_name(bool passed);

inline bool is_runall_exit_code(int val) restrict(cpu,amp) {
	return val == runall_pass
		|| val == runall_fail
		|| val == runall_skip
		|| val == runall_cascade_fail
		|| val == runall_no_value
		;
}

/// The strong type for a result type for the runall script
struct runall_result {
private:
	static int aggregate_exit_codes(int lhs, int rhs) restrict(cpu,amp) {
		if(lhs == rhs) {
			return lhs;
		}
		
		// The following are done by highest priority first
		if(lhs == runall_cascade_fail || rhs == runall_cascade_fail) {
			return runall_cascade_fail;
		} else if(lhs == runall_fail || rhs == runall_fail) {
			return runall_fail;
		} else if(lhs == runall_skip || rhs == runall_skip) {
			return runall_skip;
		} else if(lhs == runall_no_value || rhs == runall_no_value) {
			return runall_no_value;
		}
		
		// The last remaining value is pass
		return runall_pass;
	}

private:
	int _exit_code;

	void verify_exit_code() restrict(cpu);

	void verify_exit_code() restrict(amp) {
		if(!is_runall_exit_code(_exit_code)) {
			_exit_code = runall_fail;
		}
	}

public:
	/// Creates a new instance with a default result of passed.
	runall_result() restrict(cpu,amp) : _exit_code(runall_pass) {}

	/// Creates a new instance using the specified runall result constant. (i.e. runall_pass, runall_fail, runall_skip)
	/// Unknown result values:
	/// - On the gpu: will be converted to a runall_fail.
	/// - On the cpu: will throw a std::exception.
	runall_result(int result) restrict(cpu,amp) : _exit_code(result) {
		verify_exit_code();
	}

	/// Creates a new instance and initializes it with a boolean value that is treated as a pass/fail (true -> pass, false -> fail)
	runall_result(bool passed) restrict(cpu,amp) : _exit_code(passed ? runall_pass : runall_fail) {}


	/// Aggregates two runall_result instances.
	runall_result operator&(const runall_result& other) const restrict(cpu,amp) {
		return aggregate_exit_codes(_exit_code, other._exit_code);
	}

	/// Aggregates two runall_result instances.
	runall_result& operator&=(const runall_result& other) restrict(cpu,amp) {
		_exit_code = aggregate_exit_codes(_exit_code, other._exit_code);
		return *this;
	}

	bool operator==(const runall_result& other) const restrict(cpu,amp) {
		return _exit_code == other._exit_code;
	}

	bool operator!=(const runall_result& other) const restrict(cpu,amp) {
		return _exit_code != other._exit_code;
	}

	friend bool operator==(int lhs, const runall_result& rhs) restrict(cpu,amp);
	friend bool operator!=(int lhs, const runall_result& rhs) restrict(cpu,amp);
	friend runall_result operator&(int lhs, const runall_result& rhs) restrict(cpu,amp);

	/// Gets runall.pl exit code that can be returned by a main() or passed to exit().
	int get_exit_code() const restrict(cpu,amp) { return _exit_code; }


	bool get_is_pass() const restrict(cpu,amp) { return _exit_code == runall_pass; }

	/// Gets a value indicating whether this result represents a regular failure result.
	/// i.e. runall_fail.
	bool get_is_failure() const restrict(cpu,amp) { return _exit_code == runall_fail; }

	/// Gets a value indicating whether this result specifically represents a cascade failure result.
	bool get_is_cascade_failure() const restrict(cpu,amp) { return _exit_code == runall_cascade_fail; }

	bool get_is_skip() const restrict(cpu,amp) { return _exit_code == runall_skip; }

	/// Gets a value indicating whether this result specifically represents a "no value" result.
	/// See the runall documentation for how this type of return value may be used.
	bool get_is_no_value() const restrict(cpu,amp) { return _exit_code == runall_no_value; }


	/// Returns a friendly name for the runall result.
	std::string get_name() const;

	/// Returns a friendly name for the runall result.
	/// This version returns the unicode name.
	std::wstring get_name_w() const;

	/// Returns a new runall_result instance based on the current instance.
	/// If the current instance is a skip, then pass is returned; otherwise,
	/// a copy of this instance.
	runall_result treat_skip_as_pass() const restrict(cpu,amp) {
		if(get_is_skip()) {
			return runall_pass;
		} else {
			return *this;
		}
	}

};

inline bool operator==(int lhs, const runall_result& rhs) restrict(cpu,amp) {
	return rhs == runall_result(lhs);
}

inline bool operator!=(int lhs, const runall_result& rhs) restrict(cpu,amp) {
	return rhs != runall_result(lhs);
}

inline runall_result operator&(int lhs, const runall_result& rhs) restrict(cpu,amp) {
	return rhs & runall_result(lhs);
}

// define the stream operators
inline std::ostream& operator<<(std::ostream &os, const runall_result& result) {
	os << result.get_name();
    return os;
};
inline std::wostream& operator<<(std::wostream &os, const runall_result& result) {
	os << result.get_name_w();
    return os;
};


namespace Concurrency {
	namespace Test {
		/// Defines the base class for the types of objects thrown as exceptions by AmpTest library
		/// and consumed by its test_main runner.
		class amptest_exception : public std::exception {
		public:
			explicit amptest_exception(const std::string& what) : m_what(what) {}
			virtual const char* what() const AMP_NOEXCEPT override { return m_what.c_str(); }
		private:
			std::string m_what;
		};

		/// Defines the type of objects thrown as exceptions to report test failures.
		class amptest_failure : public amptest_exception {
		public:
			explicit amptest_failure(const std::string& what) : amptest_exception(what) {}
		};

		/// Defines the type of objects thrown as exceptions to report test skips.
		class amptest_skip : public amptest_exception {
		public:
			explicit amptest_skip(const std::string& what) : amptest_exception(what) {}
		};

		/// Defines the type of objects thrown as exceptions to report test cascade failrues.
		class amptest_cascade_failure : public amptest_exception {
		public:
			explicit amptest_cascade_failure(const std::string& what) : amptest_exception(what) {}
		};
	}
}
