
// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>

#include <iostream>

// a test which checks if each thread has its own default accelerator view
bool test() {

  hc::accelerator acc;
  hc::accelerator_view av1 = acc.get_default_view();
  hc::accelerator_view av2 = acc.get_default_view();
  hc::accelerator_view av3 = acc.get_default_view();

  bool ret = true;

  // av1, av2, av3 are created on the same thread so they must be equal
  ret &= ((av1 == av2) && (av2 == av3) && (av1 == av3));

  std::cout << ret << "\n";

  std::thread t2([&]() {
    av2 = acc.get_default_view();
  });

  std::thread t3([&]() {
    av3 = acc.get_default_view();
  });

  t2.join();
  t3.join();

#if !TLS_QUEUE
  // without TLS queue, av1, av2, av3 should still be the same
  ret &= ((av1 == av2) && (av2 == av3) && (av1 == av3));
#else
  // av1, av2, av3 are now created on different threads so they must NOT be equal
  ret &= ((av1 != av2) && (av2 != av3) && (av1 != av3));
#endif

  std::cout << ret << "\n";

  return ret;
}

int main() {
  bool ret = true;

  ret &= test();

  return !(ret == true);
}

