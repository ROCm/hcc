//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// FIXME this file will place C++AMP Runtime implementation (OpenCL version)

#include <amp.h>
#include <CL/opencl.h>
namespace Concurrency {

AMPAllocator& getAllocator()
{
    static AMPAllocator amp;
    return amp;
}

} // namespace Concurrency

