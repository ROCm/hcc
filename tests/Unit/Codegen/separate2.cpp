// RUN: %cxxamp -emit-llvm -S -c %s -o -|%FileCheck %s
extern "C" {
#if 0
int foo(void) [[cpu, hc]] {
	return 42;
}
#endif
int bar(void) [[hc]] {
	return 43;
}
}

class baz {
 public:
  int bzzt(void) [[cpu]] {
    return 44;
  }
  int cho(void) [[hc]] {
    return 45;
  }
};
int kerker(void) [[cpu, hc]] {
  baz b1;
  return b1.cho()+b1.bzzt();
}
// CHECK-NOT: bar
// CHECK-NOT: {{define.*cho}}
// CHECK: {{define.*bzzt}} 
