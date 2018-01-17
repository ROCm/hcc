#pragma once

namespace details {

// hc kernel invocation
template<typename Kernel>
inline void kernel_launch(int N, Kernel k, int tile = 0) {
    if (tile != 0) {
        hc::parallel_for_each(hc::extent<1>(N).tile(tile), k).wait();
    } else {
        hc::parallel_for_each(hc::extent<1>(N), k).wait();
    }
}

} // namespace details
