// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Verify that restriction modifier keyword is contextual keyword with respect to lambda expression.</summary>

#include <amptest.h>
#include <iostream>

using namespace Concurrency;
using namespace std;

bool test1()
{
   typedef int restrict;

   cout << "Test if context of restriction modifier is shielded from type-name in trailing-return-type clause" << endl;

   auto x = []()mutable restrict(cpu) -> restrict { return 1;}();

   return (x == 1);
}

class restrict : std::exception
{
    public:
    explicit restrict(const char* message)
    {
    }
};

auto lambda1 = []() restrict(cpu) throw(restrict) { throw restrict("Test");};
bool test2()
{
   cout << "Define restrict as exception and use it in lambda: []() restrict(cpu) throw(restrict)" << endl;


   try
   {
      lambda1();
      return false;
   }
   catch(restrict ex)
   {
      return true;
   }
}


bool test3()
{
   cout << "int restrict = []() restrict { return 1;}" << endl;

   int restrict = []() restrict(cpu) -> int{ return 1;}();

   return (restrict == 1);
}

bool test4()
{
   cout << "[](restrict a) restrict {}(); - pass type that has restrict name." << endl;

   class restrict
   {
      public:
        int data;
   };

   restrict x, y;

   x.data = 10;
   y.data = 20;

   int  result = [&](restrict a) restrict(cpu) -> int {return x.data + a.data;}(y);

    return (result == 30);
}

bool test5()
{
    cout << "[](int restrict) restrict(cpu) {}(); - use variable name that is d3d11." << endl;

    auto result  = [](int restrict) restrict(cpu) -> int {return 10;}(10);

    return (result == 10);
}

bool test6()
{
    cout << "lambda that captures by value variable with the restrict name." << endl;

    int restrict  = 0;

    auto result = [restrict]() restrict(cpu) -> bool {return true;}();

    return result;
}


bool test7()
{
    cout << "lambda that captures by reference variable with the restrict name." << endl;

    int restrict  = 0;

    [&]() restrict(cpu) {restrict++;}();

    return (restrict == 1);
}

// Main entry point
int main(int argc, char **argv)
{
    bool passed = true;

    if(test1())
    {
        cout << "Passed!" << endl;
    }
    else
    {
        passed = false;
        cout << "Failed!" << endl;
    }

    if(test2())
    {
        cout << "Passed!" << endl;
    }
    else
    {
        passed = false;
        cout << "Failed!" << endl;
    }

    if(test3())
    {
        cout << "Passed!" << endl;
    }
    else
    {
        passed = false;
        cout << "Failed!" << endl;
    }

    if(test4())
    {
        cout << "Passed!" << endl;
    }
    else
    {
        passed = false;
        cout << "Failed!" << endl;
    }

    if(test5())
    {
        cout << "Passed!" << endl;
    }
    else
    {
        passed = false;
        cout << "Failed!" << endl;
    }

    if(test6())
    {
        cout << "Passed!" << endl;
    }
    else
    {
        passed = false;
        cout << "Failed!" << endl;
    }

    if(test7())
    {
        cout << "Passed!" << endl;
    }
    else
    {
        passed = false;
        cout << "Failed!" << endl;
    }

    return passed ? 0 : 1;
}
