// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.

/// <summary>Copy from Array View with const type to Iterator</summary>
// RUN: %amp_device -D__GPU__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll && mkdir -p %t
// RUN: %clamp-device %t.ll %t/kernel.cl
#include "../../CopyTestFlow.h"
#include"../../../../device.h"
#include <amp.h>
#include <deque>
#include <tuple>

using namespace Concurrency;
const int RANK = 3;
typedef int DATA_TYPE;

int ArrayViewConstOnCpu()
{
  accelerator_view cpu_av = accelerator(accelerator::cpu_accelerator).get_default_view();
    return (CopyAndVerifyFromArrayViewConstToIterator<DATA_TYPE, RANK, std::vector>(cpu_av, access_type_none));
}

int ArrayViewConstOnGpu()
{
  accelerator gpu_acc = require_device_for<DATA_TYPE>();
  accelerator_view gpu_av = gpu_acc.get_default_view();

  int res = 1;

  if(gpu_acc.get_supports_cpu_shared_memory())
  {
    // Set the default cpu access type for this accelerator
    gpu_acc.set_default_cpu_access_type(access_type_read_write);

    res &= (CopyAndVerifyFromArrayViewConstToIterator<DATA_TYPE, RANK, std::vector>(gpu_av, access_type_none));
    res &= (CopyAndVerifyFromArrayViewConstToIterator<DATA_TYPE, RANK, std::vector>(gpu_av, access_type_read));
    res &= (CopyAndVerifyFromArrayViewConstToIterator<DATA_TYPE, RANK, std::vector>(gpu_av, access_type_write));
    res &= (CopyAndVerifyFromArrayViewConstToIterator<DATA_TYPE, RANK, std::vector>(gpu_av, access_type_read_write));
  }
  else
  {
    res &= (CopyAndVerifyFromArrayViewConstToIterator<DATA_TYPE, RANK, std::vector>(gpu_av, access_type_none));
  }

  return res;
}

int main()
{
  int res = 1;

  res &= ArrayViewConstOnCpu();
  res &= ArrayViewConstOnGpu();

  return !res;
}

