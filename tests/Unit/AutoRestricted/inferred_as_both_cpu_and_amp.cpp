// RUN: %cxxamp %s -o %t.out && %t.out
#include <iostream>
#include <amp.h>
using namespace std;

int f1() restrict(cpu) {return 1;} 
int f1() restrict(amp) {return 2;}

int f2() restrict(auto) {
  return f1();
}

// If not inferred or wrongly inferred
int CPU_Func() restrict(cpu)
{
  if(f2() != 1) // if referred to be amp only, expected-error{{call from CPU-restricted function to AMP-restricted function}}
    std::cout<<"Fail to verify result of f2() in CPU path!\n";

  return f2();
}


// If not inferred or wrongly inferred
int AMP_Func() restrict(amp)
{
  if(f2() != 2)  // if referred to be cpu only, expected-error{{call from AMP-restricted function to CPU-restricted function}}
  {
    std::cout<<"Fail to verify result of f2() in GPU path!\n"; 
    exit(1);
  }

  return f2();
}

int AMP_AND_CPU_Func() restrict(cpu,amp)
{
  return f2(); // OK
}


int main(void)
{
  return 0;
}

