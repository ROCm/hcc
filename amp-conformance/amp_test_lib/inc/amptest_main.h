// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
#pragma once
#include <amptest/context.h>
#include <amptest/platform.h>
#include <amptest/runall.h>

/// The signature of the test function that amptest_main will call.
runall_result AMP_TEST_API test_main();

namespace Concurrency
{
	namespace Test
	{
		/// The entry point for an AMP test. This function handles setting up the environment context
		/// such as setting up handlers for unhandled exceptions thrown from within the process.
		/// This function invokes the test_main() implementation.
		int AMP_TEST_API amptest_main(amptest_context_t& context);
	}
}

// Note: the function is defined in a header file on purpose!
// Warning: This header should not be included in more than one compilation unit!
int AMP_TEST_API main(int argc, char** argv)
{
	Concurrency::Test::amptest_context_t context(argc, argv);
	return Concurrency::Test::amptest_main(context);
}


