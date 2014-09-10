// RUN: %amp_device -D__GPU__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll && mkdir -p %t
// RUN: %clamp-device %t.ll %t/kernel.cl
// RUN: pushd %t && %embed_kernel kernel.cl %t/kernel.o && popd
// RUN: %cxxamp %link %t/kernel.o %s -o %t.out && %t.out
#include <amp.h>

int f1() restrict(cpu,amp) {return 1;} 
// DeclRefExpr
int f2() restrict(cpu,auto) {
  return f1();
}

// null
void f_null() restrict(cpu,auto) {
}


// ReturnStmt
int f_return() restrict(cpu,auto) {
  return 1;
}

// CXXTryStmt
// GotoStmt
// LabelStmt


int AMP_CPU_Func() restrict(cpu,amp) 
{
  f2();  // OK, 'auto' is inferred to amp, so f2 is both (cpu,amp) restricted
  f_null(); // OK
  f_return();  // OK 
}

int main(void)
{
  return 0;  // expected: success
}

