// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.

/// <summary>Verify function data() returning NULL on CPU for non-CPU accelerators</summary>

#include <iterator>
#include "./../../member.h"
#include <amptest.h>
#include <amptest_main.h>

typedef int32_t __int32;

template<typename _type, int _rank>
bool test_feature(accelerator_view device_view)
{	
	int edata[_rank];
    for (int i = 0; i < _rank; i++)
        edata[i] = i+1;
    extent<_rank> e1(edata);

    std::vector<_type> data(e1.size());
    for (unsigned int i = 0; i < e1.size(); i++)
        data[i] = static_cast<_type>(i+1);


    {
        array<_type, _rank> src(e1, data.begin(), data.end(), device_view);

        _type* dst_data = src.data();

        if (dst_data != NULL)
            return false;
    }

    {
        array<_type, _rank> src(e1, data.begin(), device_view);

        const _type* dst_data = src.data();

        if (dst_data != NULL)
            return false;
    }

    {
        const array<_type, _rank> src(e1, data.begin(), data.end(), device_view);
        
        const _type* dst_data = src.data();

        if (dst_data != NULL)
            return false;
    }

    return true;
}

runall_result test_main()
{

	// Test is using doubles therefore we have to make sure that it is not executed 
	// on devices that does not support double types.
	// Test is relying on default device, therefore check below is also done on default device.
	accelerator device = require_device_with_double(Test::Device::ALL_DEVICES);
	accelerator_view device_view = device.get_default_view();
	
	if(device.get_supports_cpu_shared_memory())
	{
		WLog() << "Accelerator " << device.get_description() << " supports zero-copy" << std::endl;
		return runall_skip;
	}
	
	runall_result res;
	
	res &= REPORT_RESULT((test_feature<int, 5>(device_view)));
	res &= REPORT_RESULT((test_feature<float, 7>(device_view)));
	res &= REPORT_RESULT((test_feature<double, 7>(device_view)));
	res &= REPORT_RESULT((test_feature<__int32, 5>(device_view)));
	
    return res;
}
                                                                                                                                      
