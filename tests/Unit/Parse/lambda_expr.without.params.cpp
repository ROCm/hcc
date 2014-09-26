// RUN: %cxxamp -c %s

int f1() restrict(amp) { return 1;}
int f_amp() restrict(amp) {
  []
  {
    f1(); // OK
  };

  return 0;
}

