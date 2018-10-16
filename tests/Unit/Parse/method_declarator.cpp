// RUN: %cxxamp -c %s

class AClass {
public:
  AClass();

  AClass(int n) [[cpu, hc]];   // constructor with restrict should be accepted.

  int method_1() const;               // not a problem

  int method_2() [[cpu, hc]];  // should accept

  int method_3() restrict;            // not to be confused with C++AMP restrict.
};

int func() [[hc]] {
  return 0;
}


