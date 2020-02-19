// RUN: %cxxamp %s -o %t.out && %t.out
#include <hc.hpp>
using namespace hc;

#define LLVM_OVERRIDE override

class A {
   public:
    virtual void foo() = 0;
};

class B : public A {
   public:
    virtual void foo() noexcept LLVM_OVERRIDE {}
};

int main(void) { return 0; }
