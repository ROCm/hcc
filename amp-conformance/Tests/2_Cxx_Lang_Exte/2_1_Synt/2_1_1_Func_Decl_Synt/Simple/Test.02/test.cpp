// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P2</tags>
/// <summary>syntax generator test for simple-declaration (namespace scope)</summary>

#include "amptest/restrict.h"

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

ATTRIBUTE_SPECIFIER DECL_SPECIFIERS TYPE_SPECIFIER LEADING_DECLARATION DECL_MODIFIER1 func(PARAMETERS) CV_QUALIFIER REF_QUALIFIER __CPU_ONLY_EXPLICIT EXCEPTION_SPECIFICATION -> TRAILING_RETURN_TYPE TRAILING_DECLARATION;

ATTRIBUTE_SPECIFIER DECL_SPECIFIERS TYPE_SPECIFIER DECL_MODIFIER1 func(PARAMETERS) CV_QUALIFIER REF_QUALIFIER __CPU_ONLY_EXPLICIT EXCEPTION_SPECIFICATION -> TRAILING_RETURN_TYPE
{
    return RETURN_EXPRESSION;
}

#ifdef NAMESPACE_ENABLE
}
#endif

int main()
{
    #ifdef NAMESPACE_ENABLE
    TYPE_SPECIFIER (DECL_MODIFIER1 *pointer)(PARAMETERS) CV_QUALIFIER REF_QUALIFIER __CPU_ONLY_EXPLICIT EXCEPTION_SPECIFICATION -> TRAILING_RETURN_TYPE = &NS::func;
    return pointer(PARAMETER_VALUES) == RETURN_VALUE ? 0 : 1;
    #else
    TYPE_SPECIFIER (DECL_MODIFIER1 *pointer)(PARAMETERS) CV_QUALIFIER REF_QUALIFIER __CPU_ONLY_EXPLICIT EXCEPTION_SPECIFICATION -> TRAILING_RETURN_TYPE = &func;
    return pointer(PARAMETER_VALUES) == RETURN_VALUE ? 0 : 1;
    #endif
}

