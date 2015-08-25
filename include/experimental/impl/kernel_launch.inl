#pragma once

namespace details {

// hc kernel invocation
template<typename Kernel>
inline void kernel_launch(int N, Kernel k) {
  hc::ts_allocator tsa;
  // FIXME: ** wait on completion_future **
  //hc::completion_future fut =
  hc::parallel_for_each(hc::extent<1>(N), tsa, k);
  //fut.wait();
}

}
