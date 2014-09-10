// RUN: %amp_device -D__GPU__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll && mkdir -p %t
// RUN: %clamp-device %t.ll %t/kernel.cl
// RUN: pushd %t && %embed_kernel kernel.cl %t/kernel.o && popd
// RUN: %cxxamp %link %t/kernel.o %s -o %t.out && %t.out
#include <amp.h>
using std::vector;
using namespace concurrency;

//2_Cxx_Lang_Exte/2_4_amp_Rest_Modi/2_4_1_Rest_on_Type/2_4_1_3_Comp_Type/Negative/BoolPointer/test.cpp
void f_boolpointer() restrict(auto) // Not a negative test anymore since pointer to bool is now supported
{
    bool b;
    bool * pb = &b;
    *pb = true;
}

void AMP_AND_CPU_Func() restrict(cpu,amp)
{
  f_boolpointer(); // OK
}

int main(void)
{
  return 0;
}

