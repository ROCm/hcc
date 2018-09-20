// RUN: %amp_device -D__KALMAR_ACCELERATOR__ %s -emit-llvm -c -S -O2 -o %t.ll 2>&1 | %FileCheck --strict-whitespace %s

//////////////////////////////////////////////////////////////////////////////////
// Do not delete or add any line; it is referred to by absolute line number in the
// FileCheck lines below
//////////////////////////////////////////////////////////////////////////////////
#include <hc.hpp>
using namespace hc;

// test enum with illegal underlying type
enum Suit : char {
    Diamonds,
    Hearts,
    Clubs,
    Spades
};

bool foo_enum(Suit suit) restrict(auto)
{
    if (suit == Diamonds)
        return true;
    else
        return false;
}

void AMP_AND_CPU_Func() [[cpu, hc]] {
  foo_enum(Hearts);
}
// CHECK: Enum.cpp:[[@LINE-2]]:3: error: call from AMP-restricted function to CPU-restricted function
// CHECK-NEXT: foo_enum(Hearts);
// CHECK-NEXT: ^

int main(void)
{
  return 0;
}

