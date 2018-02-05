// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Create an array of a user defined union with supported type</summary>

#include "./../../../constructor.h"
#include <amptest_main.h>

typedef union
{
    int n;
    unsigned u;
    long l;
    unsigned long ul;
    float f;
    double  d;
} usr_union;

template<typename _type, int _rank>
void kernel_userdefined(index<_rank>& idx, array<_type, _rank>& f) __GPU
{
    f[idx].n = -5;
    f[idx].u = static_cast<unsigned>(-5);
    f[idx].l = -5;
    f[idx].ul = static_cast<unsigned long>(-5);
    f[idx].f = -5.0f;
    f[idx].d = -5.0;  // This is the only one verified
}

template<typename _type, int _rank>
bool verify_kernel_userdefined(array<_type, _rank>& arr)
{
    std::vector<usr_union> vdata = arr;
    for (size_t i = 0; i < vdata.size(); i++)
    {
        // verify the last assignment in union
        if (vdata[i].d != -5.0)
            return false;
    }

    return true;
}

runall_result test_main()
{
    // Test is using doubles therefore we have to make sure that it is not executed
    // on devices that does not support double types.
	accelerator::set_default(require_device_with_double().get_device_path());

	runall_result result;

    int extdata[] = {2, 2};
	result &= REPORT_RESULT((test_array_userdefined<usr_union, 1>(extdata)));
	result &= REPORT_RESULT((test_array_userdefined<usr_union, 2>(extdata)));

	return result;
}

