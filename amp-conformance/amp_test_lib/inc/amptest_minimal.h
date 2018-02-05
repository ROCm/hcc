// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
#pragma once
/**********************************************************************************
* amptest_minimal.h
*
* Defines the minimal API for testing C++ AMP. The main difference between this
* header and amptest.h is that this doesn't include the AMP runtime or depend on
* any of its data structures.
*
**********************************************************************************/

// Include the appropriate libraries
//#include <amptest\dpctest_lib.h>

//
#include <amptest/restrict.h> // This will re-define and add explicit macros

//
#include <amptest/context.h>
#include <amptest/device.h>
#include <amptest/runall.h>
#include <amptest/logging.h>
//#include <amptest/data.h>


