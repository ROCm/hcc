// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Test that parallel_for_each allows marshaling of stateless classes
/// </summary>

#include <amptest.h>

using namespace Concurrency;
using namespace Concurrency::Test;

using std::vector;

template<typename T>
void test(vector<T>& data, const int size, accelerator_view av)
{
	extent<1> ex(size);
	array<T, 1> arr(ex, data.begin(), av);
	
	auto f = [] (T v1, T v2) restrict(amp) { return v1 + v2; };

	parallel_for_each(ex, [&, f](index<1> idx) restrict(amp)
	{
		arr[idx] = f(arr[idx], 20);
	});

    data = arr;
}
int main()
{
    const int size = 10;
	vector<int> data(size);

    for(int i = 0; i < size; i++)
    {
        data[i] = i;
    }
		
    accelerator_view av = require_device(Device::ALL_DEVICES).get_default_view();
	test<int>(data, size, av);

    for(int i = 0; i < size; i++)
    {
        if(data[i] != i+20)
        {
            printf("Fail: Incorrect output value. Expected:[%d] Actual:[%d]\n", i+20, data[i]);
            return runall_fail;
        }
    }
	
    printf("Pass\n");
    return runall_pass;
}


