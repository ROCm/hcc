// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Verify array accelerator_view property - on all devices for all queuemode</summary>

#include "./../../../member.h"

template<typename _type, int _rank>
bool test_feature()
{
	vector<accelerator> devices = accelerator::get_all();

    int edata[_rank];
    for (int i = 0; i < _rank; i++)
        edata[i] = 3;
    const extent<_rank> e1(edata);

	printf("Found %zu devices\n", devices.size());

	for (size_t i = 0; i < devices.size(); i++)
	{
		accelerator device = devices[i];

        { // non-const accelerator_view verification
            accelerator_view av = device.get_default_view();
		    array<_type, _rank> src(e1, device.get_default_view());

            if (src.get_accelerator_view() != av)
                return false;
        }

        { // const accelerator_view verification + immediate mode
            accelerator_view av = device.create_view(queuing_mode_immediate);
		    const array<_type, _rank> src(e1, av);

            if (src.get_accelerator_view() != av)
                return false;
        }

        { // verify defered mode
            accelerator_view av = device.create_view(queuing_mode_automatic);
		    const array<_type, _rank> src(e1, av);

            if (src.get_accelerator_view() != av)
                return false;
        }

		printf("Finished with device %zu\n", i);
	}

	return true;
}

int main()
{
    int passed = test_feature<float, 5>() ? runall_pass : runall_fail;

    printf("%s\n", (passed == runall_pass) ? "Passed!" : "Failed!");

    return passed;
}

