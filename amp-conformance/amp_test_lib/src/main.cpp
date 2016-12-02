// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
#include <amp.h>
#include <amptest/context.h>
#include <amptest/logging.h>
#include <amptest/runall.h>
#include <typeinfo>
#include <string>
#include <sstream>

// Declare global vars first
namespace Concurrency {
	namespace Test {

		bool amptest_context_enable_debugging = true;	// The actual default is false in amptest_load_context
														// This initial value makes it so we can debug any issues that occur while setting up the test context.
	}
}

/// The signature of the test function that amptest_main will call.
/// This will be defined in the test code.
runall_result AMP_TEST_API test_main();

namespace Concurrency {
	namespace Test {

		// Defined in device.cpp:
		// This is used internally to control whether require_device should throw an exception or just exit
		// when no device is available.
		void AMP_TEST_API set_require_device_behavior(bool exit_on_no_device);


		namespace details {

			// Reads command-line and environment variables to apply context settings
			static void amptest_load_context() {
				try {
					amptest_context_enable_debugging = amptest_context.get_environment_variable("AMPTEST_ENABLE_DEBUGGING", false);
				} catch(std::exception& ex) {
					Log(LogType::Error, true) << "Error loading AMPTest context: " << ex.what() << std::endl;
					exit(runall_fail);
				}
			}

			static runall_result invoke_test_main() {
				runall_result result;
				try {
					// Make sure the DMF doesn't use exit()
					set_require_device_behavior(false /*exit_on_no_device*/);

					result = test_main();
				} catch(amptest_skip e) {
					Log(LogType::Warning, true) << e.what() << std::endl;
					result = runall_skip;
				} catch(amptest_cascade_failure e) {
					Log(LogType::Error, true) << e.what() << std::endl;
					result = runall_cascade_fail;
				} catch(amptest_failure e) {
					Log(LogType::Error, true) << e.what() << std::endl;
					result = runall_fail;
				} catch(const amptest_exception& e) {
					Log(LogType::Error, true) << "test_main() threw unhandled " << get_type_name(e) << " exception: " << e.what() << std::endl;
					result = runall_fail;
				}
				Log(LogType::Info, true) << "test_main(): Returned " << result << std::endl;
				return result;
			}

			static runall_result invoke_test_main_with_exception_handling() {

				try {
					return invoke_test_main();
				} catch(concurrency::accelerator_view_removed& ex) {
					// If we catch a TDR, then lets be sure to print out the reason too
					Log(LogType::Error, true) << "test_main() threw unhandled " << get_type_name(ex) << " exception "
						<< "(error code: 0x" << std::hex << ex.get_error_code() << std::dec
						<< ", reason code: 0x" << std::hex << ex.get_view_removed_reason() << std::dec
						<< "): "
						<< ex.what() << std::endl;
					return runall_fail;
				} catch(concurrency::runtime_exception& ex) {
					Log(LogType::Error, true) << "test_main() threw unhandled " << get_type_name(ex) << " exception "
						<< "(error code: 0x" << std::hex << ex.get_error_code() << std::dec << "): "
						<< ex.what() << std::endl;
					return runall_fail;
				} catch(const std::exception& ex) {
					Log(LogType::Error, true) << "test_main() threw unhandled " << get_type_name(ex) << " exception: " << ex.what() << std::endl;
					return runall_fail;
				} catch(const char* exmsg) {
					Log(LogType::Error, true) << "test_main() threw unhandled exception: " << exmsg << std::endl;
					return runall_fail;
				}
				catch(...)
				{
					Log(LogType::Error, true) << "test_main() threw unhandled unknown exception caught." << std::endl;
					return runall_fail;
				}
			}

		} // namespace details

		int AMP_TEST_API amptest_main(amptest_context_t& context) {
			amptest_context = context;
			details::amptest_load_context();

			runall_result test_result;
			if(amptest_context_enable_debugging) {
				test_result = details::invoke_test_main();
			} else {
				test_result = details::invoke_test_main_with_exception_handling();
			}

			return test_result.get_exit_code();
		}
	} // namespace Test
} // namespace Concurrency


