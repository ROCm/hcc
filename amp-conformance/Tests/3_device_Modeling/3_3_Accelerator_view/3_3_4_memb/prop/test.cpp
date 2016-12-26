// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.

/// <summary>Tests properties of accelerator views created on get_all accelerators</summary>

#include <amptest.h>
#include <amptest_main.h>

using namespace Concurrency;
using namespace Concurrency::Test;

bool test_accelerator_view(const accelerator_view& av, const accelerator& acc)
{
	return av.get_accelerator() == acc
		&& av.get_is_debug() == acc.get_is_debug()
		&& av.get_version() == acc.get_version();
}

runall_result test_main()
{
	runall_result result;

	std::vector<accelerator> accs = accelerator::get_all();
	std::for_each(accs.begin(), accs.end(),
		[&](accelerator& acc)
		{
        		Log(LogType::Info, true) << "For device : " << acc.get_description() << std::endl;

			// default accelerator view
			result &= REPORT_RESULT(test_accelerator_view(acc.get_default_view(), acc));
		}
	);

	return result;
}
