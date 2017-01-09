// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Test for diffrent shapes of array: 1D, 2D, 3D and 4D</summary>

#include <amptest.h>
#include <amptest_main.h>
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <float.h>

using namespace Concurrency;
using namespace Concurrency::Test;
using std::vector;

void InitializeArray(float *pM, int size)
{
    for(int i=0; i<size; ++i)
    {
        pM[i] = static_cast<float>(rand() % 256);
    }
}

template<int DIM>
void ShapeKernel(index<DIM>& idx, array<float, DIM>& mC, array<float, DIM>& mA, array<float, DIM>& mB) restrict(amp)
{
    mC[idx] = mA[idx] + mB[idx];
}

template<int DIM>
runall_result Shape(accelerator_view av, int extents[], int base)
{
    srand(2010);

    int size = 1;
    for (int i=0; i<DIM; ++i)
    {
        size *= base;
    }

    vector<float> pA(size);
    vector<float> pB(size);
    vector<float> pC(size);
    vector<float> pRef(size);

    InitializeArray(pA.data(), size);
    InitializeArray(pB.data(), size);

    // Compute result on CPU
    for(int i=0; i<size; ++i)
    {
        pA[i] = static_cast<float>(i);
        pB[i] = static_cast<float>(i + 1);

        pRef[i] = pA[i] + pB[i];
    }

    extent<DIM> e(extents);

    array<float, DIM> mA(e, pA.begin(), av), mB(e, pB.begin(), av), mC(e, av);

    parallel_for_each(e, [&](index<DIM> idx) restrict(amp) {
        ShapeKernel<DIM>(idx, mC, mA, mB);
    });

    pC = mC;

    return Verify(pC, pRef);
}

runall_result test_main()
{
	accelerator_view av = require_device(device_flags::NOT_SPECIFIED).get_default_view();

    const int base = 64;

    int e1[] = {base};
    int e2[] = {base, base};
    int e3[] = {base, base, base};
    int e4[] = {base, base, base, base};

	runall_result result;

	result &= REPORT_RESULT(Shape<1>(av, e1, base));
	result &= REPORT_RESULT(Shape<2>(av, e2, base));
	result &= REPORT_RESULT(Shape<3>(av, e3, base));
	result &= REPORT_RESULT(Shape<4>(av, e4, base));

	return result;
}

