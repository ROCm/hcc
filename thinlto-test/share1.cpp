extern "C" int foo(int grid_size) [[hc]] {
  return grid_size + 1;
}

extern "C" int bar(int grid_size) [[hc]] {
  return grid_size * 2;
}
