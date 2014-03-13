// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
#pragma once
/**********************************************************************************
* amptest\restrict.h
*
* Defines the minimal APIs from amp.h that are needed for amptest_minimal.h but
* don't require including amp.h.
**********************************************************************************/

#if !defined(__GPU)
// These are already defined in amp.h but we explicitly define them here.
// but note that __CPU_ONLY is defining the restrict(cpu) explicitly where amp.h does it implicitly
// WARNING: As these macros are the same as defined in amp.h and they are not publicly
// spec'd macros, we shouldn't use them in our tests.

// Since we're redefining these macros as they are defined in amp.h, just ignore the redefinition warning
#define __GPU      restrict(amp,cpu)
#define __GPU_ONLY restrict(amp)
#define __CPU_ONLY		// Note: amp.h defines this as an implicit restrict(cpu).

#endif

// For now, we'll define the explicit macro (used in syntax tests)
#define __CPU_ONLY_EXPLICIT restrict(cpu)

