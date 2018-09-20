// RUN: %cxxamp -c %s

int f1() [[hc]] { return 1;}
int f_amp() [[hc]] {
  []
  {
    f1(); // OK
  };

  return 0;
}

