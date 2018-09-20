// RUN: %cxxamp %s -o %t.out && %t.out
#include <hc.hpp>
using std::vector;
using namespace hc;

//2_Cxx_Lang_Exte/2_4_amp_Rest_Modi/2_4_1_Rest_on_Type/2_4_1_3_Comp_Type/Negative/BoolPointer/test.cpp
void f_boolpointer() restrict(auto) // Not a negative test anymore since pointer to bool is now supported
{
    bool b;
    bool * pb = &b;
    *pb = true;
}

void AMP_AND_CPU_Func() [[cpu, hc]]
{
  f_boolpointer(); // OK
}

int main(void)
{
  return 0;
}

