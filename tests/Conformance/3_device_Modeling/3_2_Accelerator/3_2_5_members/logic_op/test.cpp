// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.

/// <summary>Tests the logical operator on an assigned accelerator and a copied accelerator. This test assumes a default accelerator is available and is different from cpu accelerator.If assumption is invalid test will skip</summary>
// RUN: %cxxamp %s %link
// RUN: ./a.out
#include "../../../accelerator.common.h"
#include "../../../../device.h"

using namespace Concurrency;
using namespace Concurrency::Test;
int main()
{
    int result=1;

    accelerator acc_def = require_device();
    accelerator acc = accelerator(accelerator::cpu_accelerator);

    // verify the logical operators using a copy constructed accelerator
    result &=((acc == accelerator(accelerator::cpu_accelerator)) == true);
    result &=((acc != accelerator(accelerator::cpu_accelerator)) == false);

    acc = acc_def;

    // verify the logical operators using an assign constructed accelerator
    result &=((acc == acc_def) == true);
    result &=((acc != acc_def) == false);

    return !result;
}
