// RUN: %amp_device -c -S -emit-llvm %s -o -|%FileCheck %s
extern "C" {
int foo(void) {
	return 42;
}
int bar(void) restrict(amp) {
	return 43;
}
}

class baz {
 public:
  int bzzt(void) {
    return 44;
  }
  __attribute__((noinline))
  int cho(void) restrict(amp) {
    return 45;
  }
};
int kerker(void) restrict(amp,cpu) {
  baz b1;
  return b1.cho()+b1.bzzt();
}
// CHECK-NOT: foo
// CHECK: bar
// CHECK: cho 
// CHECK-NOT: {{define.*bzzt}} 
