// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>This test checks that a function qualifier can be used as a global variable name</summary>

#include <amptest.h>
#include <iostream>

using namespace Concurrency;
using namespace std;

double restrict ;

int Test1(double d)
{
    restrict = d;

    return ((restrict == d) ? 0 : 1);
}


// Main entry point
int main(int argc, char **argv)
{
    bool passed = true;

    cout << "Test: declare global variable with the same name as modifier" << endl;
    if(Test1(10.0001) == 0)
    {
        cout << "Passed!" << endl;
    }
    else
    {
        cout << "Failed!" << endl;
        passed = false;
    }

    return passed ? 0 : 1;
}
