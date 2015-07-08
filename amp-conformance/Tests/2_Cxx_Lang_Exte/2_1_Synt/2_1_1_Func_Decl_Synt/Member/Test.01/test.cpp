// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P2</tags>
/// <summary>syntax generator test for function declaration (class scope)</summary>

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

struct FooClass
{
    ATTRIBUTE_SPECIFIER DECL_SPECIFIERS TYPE_SPECIFIER DECL_MODIFIER1 func(PARAMETERS) CV_QUALIFIER REF_QUALIFIER __CPU_ONLY_EXPLICIT EXCEPTION_SPECIFICATION;
};

TYPE_SPECIFIER DECL_MODIFIER1 FooClass::func(PARAMETERS) CV_QUALIFIER REF_QUALIFIER __CPU_ONLY_EXPLICIT EXCEPTION_SPECIFICATION
{
    return RETURN_EXPRESSION;
}


int main()
{
    FooClass f;
    TYPE_SPECIFIER (DECL_MODIFIER1 FooClass::* pointer)(PARAMETERS) CV_QUALIFIER REF_QUALIFIER __CPU_ONLY_EXPLICIT EXCEPTION_SPECIFICATION = &FooClass::func;
    return ((f).*(pointer))(PARAMETER_VALUES) == RETURN_VALUE ? 0 : 1;

}

