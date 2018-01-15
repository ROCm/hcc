// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>C6706 : function pointer, function reference, or pointer to member function is not supported in amp restricted code</summary>
//#Expects: Error: error C3581
//#Expects: Error: error C2530

#include <amptest.h>

void FunctionReferenceNotSupported(int x) __GPU_ONLY
{
	int (&pt2Function)(float);
}

int main()
{
	//Execution should never reach here
	//return 1 to indicate failure
	return 1;
}

