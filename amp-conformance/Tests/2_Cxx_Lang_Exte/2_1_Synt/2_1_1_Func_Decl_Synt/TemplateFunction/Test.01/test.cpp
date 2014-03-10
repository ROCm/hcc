// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P2</tags>
/// <summary>syntax generator test for template function declaration (namespace scope)</summary>

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

template <typename T>
DECL_SPECIFIERS TYPE_SPECIFIER func(PARAMETERS) __CPU_ONLY_EXPLICIT EXCEPTION_SPECIFICATION;

template <typename T>
DECL_SPECIFIERS TYPE_SPECIFIER func(PARAMETERS) __CPU_ONLY_EXPLICIT EXCEPTION_SPECIFICATION
{
    return RETURN_EXPRESSION;
}

#ifdef NAMESPACE_ENABLE
}
#endif

int main()
{
    #ifdef NAMESPACE_ENABLE
    return NS::func(PARAMETER_VALUES) == RETURN_VALUE ? 0 : 1;
    #else
    return func(PARAMETER_VALUES) == RETURN_VALUE ? 0 : 1;
    #endif
}

