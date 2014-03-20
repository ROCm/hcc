// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P2</tags>
/// <summary>syntax generator test for simple-declaration (namespace scope)</summary>

#include "amptest_minimal.h"

#ifdef UDT_ENABLE
struct UDT
{
    UDT() : x(0) {};
    UDT(int x) : x(x) {};
    bool operator==(const UDT& other)
    {
        return other.x == this->x;
    }
    int x;
};
#endif

#ifdef NAMESPACE_ENABLE
namespace NS {
#endif

typedef TYPE_SPECIFIER LEADING_DECLARATION name(PARAMETERS) __CPU_ONLY_EXPLICIT EXCEPTION_SPECIFICATION TRAILING_DECLARATION;
typedef TYPE_SPECIFIER LEADING_DECLARATION (*pname)(PARAMETERS) __CPU_ONLY_EXPLICIT EXCEPTION_SPECIFICATION TRAILING_DECLARATION;
typedef TYPE_SPECIFIER LEADING_DECLARATION (&rname)(PARAMETERS) __CPU_ONLY_EXPLICIT EXCEPTION_SPECIFICATION TRAILING_DECLARATION;

#ifdef NAMESPACE_ENABLE
}
#endif

int main()
{
    #ifdef NAMESPACE_ENABLE
    NS::name *obj;
    NS::pname pobj;
    #else
    name *obj;
    pname pobj;
    #endif
    return runall_pass;
}

