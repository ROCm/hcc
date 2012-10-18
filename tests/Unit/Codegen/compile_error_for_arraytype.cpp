//RUN: %clang++ -cc1 -std=c++amp -internal-isystem -fsyntax-only -verify %s 
class baz {
 public:
  void cho(void) restrict(amp) {};
  int bar;
  int n[10]; // expected-error{{the field type is not amp-compatible}}
};

int kerker(void) restrict(amp,cpu) {
  baz bl;
  return 0;
}
