// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Overload implicit conversions on restriction.</summary>

// Case 1: ctor and func are restrict(cpu,amp)
struct A { };

struct B1 {
  B1(A) restrict(cpu,amp) { }
};

void func1(B1) restrict(cpu,amp) { }

void entry1() restrict(cpu,amp) {
   A s;
   func1(s);
}


// Case 2: ctor has seperate implementations for restrict(cpu) and restrict(amp), func uses restrict(cpu,amp)
struct B2 {
  B2(A) restrict(cpu) { }
  B2(A) restrict(amp) { }
};

void func2(B2) restrict(cpu,amp) { }

void entry2() restrict(cpu,amp) {
   A s;
   func2(s);
}


// Case 3: func has seperate implementations for restrict(cpu) and restrict(amp), cpu uses restrict(cpu,amp)
struct B3 {
  B3(A) restrict(cpu,amp) { }
};

void func3(B3) restrict(cpu) { }
void func3(B3) restrict(amp) { }

void entry3() restrict(cpu,amp) {
   A s;
   func3(s);
}


// Case 4: Both the func and ctor have seperate implementations for restrict(cpu) and restrict(amp)
struct B4 {
  B4(A) restrict(cpu) { }
  B4(A) restrict(amp) { }
};

void func4(B4) restrict(cpu) { }
void func4(B4) restrict(amp) { }

void entry4() restrict(cpu,amp) {
   A s;
   func4(s);
}


