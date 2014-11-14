// RUN: %cxxamp -c %s

class AClass {
public:
  AClass();

  AClass(int n) restrict(amp, cpu);   // constructor with restrict should be accepted.

  int method_1() const;               // not a problem

  int method_2() restrict(amp, cpu);  // should accept

  int method_3() restrict;            // not to be confused with C++AMP restrict.
};

int func() restrict(amp) {
  return 0;
}


