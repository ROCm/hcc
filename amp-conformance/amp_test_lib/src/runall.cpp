// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.

#include <amptest/runall.h>

static const std::string runall_pass_name         = "Pass";
static const std::string runall_fail_name         = "Failure";
static const std::string runall_cascade_fail_name = "Cascade Failure";
static const std::string runall_skip_name         = "Skip";
static const std::string runall_no_value_name     = "No Value";
static const std::string runall_unknown_name      = "Unknown";

/// Returns a friendly name for the runall result type.
std::string AMP_TEST_API runall_result_name(int result) {
	if(result == runall_pass) {
		return runall_pass_name;
	} else if(result == runall_fail) {
		return runall_fail_name;
	} else if(result == runall_cascade_fail) {
		return runall_cascade_fail_name;
	} else if(result == runall_skip) {
		return runall_skip_name;
	} else if(result == runall_no_value) {
		return runall_no_value_name;
	} else {
		return runall_unknown_name;
	}
}

/// Returns a friendly name for the runall result type.
std::string AMP_TEST_API runall_result_name(bool passed) {
	if(passed) {
		return runall_pass_name;
	} else {
		return runall_fail_name;
	}
}

std::string runall_result::get_name() const {
	return runall_result_name(_exit_code);
}

void runall_result::verify_exit_code() restrict(cpu) {
	if(!is_runall_exit_code(_exit_code)) {
		throw std::invalid_argument("Invalid exit_code passed to runall_result(int) constructor.");
	}
}
