// RUN: %amp_device -D__KALMAR_ACCELERATOR__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll 2>&1 | %FileCheck --strict-whitespace %s

//////////////////////////////////////////////////////////////////////////////////
// Do not delete or add any line; it is referred to by absolute line number in the
// FileCheck lines below
//////////////////////////////////////////////////////////////////////////////////
class baz {
 public:
  void cho(void) restrict(amp) {};
  int bar;
  int* n[10];
};
// CHECK: compile_error_for_arraytype.cpp:[[@LINE-2]]:3: error: the field type is not amp-compatible
// CHECK-NEXT: int* n[10];
// CHECK-NEXT: ^


int kerker(void) restrict(amp,cpu) {
  baz bl;
  return 0;
}
// CHECK: compile_error_for_arraytype.cpp:[[@LINE-3]]:3: error: 'class baz': unsupported type in amp restricted code
// CHECK-NEXT: baz bl;
// CHECK-NEXT: ^

