// RUN: %cxxamp -c %s

int func() restrict(amp) {
  return 0;
}

