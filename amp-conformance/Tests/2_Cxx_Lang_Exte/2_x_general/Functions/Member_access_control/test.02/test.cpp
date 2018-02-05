// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Add the modifier to a protected member</summary>

#include <amptest.h>


// Test Cases
///////////////
class testclass {

    int r;

protected:

    int setr(int a) __GPU    {
        r = a;
        return r;
    };

public:

    int callprotected(int a)  {
        return setr(a);
    }

};
////////////////////////////////////////////


#define PRINT_RESULT(x) (x?"Failed!":"Passed")
#define CHECK_RESULT(r,x) (r=x?r:1)

// Main entry point
int main(int argc, char **argv)
{
    int result = 0;
    testclass test = testclass();

    CHECK_RESULT( result, test.callprotected(1) );

    printf("(Member functions 13) 	(P2)  "
        "Add the modifier to a protected member  : %s",
        PRINT_RESULT( result ) );

    return result;
}
