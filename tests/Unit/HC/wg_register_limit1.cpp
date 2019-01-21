// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>
#include <string>
#include <cmath>
#include <hc_math.hpp>

int main() {
  bool pass = false;

  try  {
    constexpr int array_size = 8192;
    hc::array_view<int,1> a(array_size);

    // We expect the following kernel to use a lot of registers and can't support
    // a tile size of 512, which will trigger an exception
    hc::parallel_for_each(a.get_extent().tile(512), [=](hc::tiled_index<1> idx) [[hc]] {

      constexpr int num_elements = 129;
      int temp[num_elements];
      
      const int start = idx.global[0] - num_elements/2;
      const int end = idx.global[0] + num_elements/2;

      #pragma unroll
      for (int i = 0; i < num_elements; ++i) {
        int load_idx = std::min(std::max(start + i,0), array_size - 1);
        temp[i] = a[load_idx];
      }

      int result_index = idx.global[0];
      #pragma unroll
      for (int i = 0; i < 10; i++) {

        int sum = 0;
        #pragma unroll
        for (int j = 0; j < num_elements; j++)
          sum+=temp[j];

        a[std::min(result_index++, array_size-1)] += sum;

        #pragma unroll
        for (int j = 0; j < num_elements-1; j++)
          temp[j] = temp[j+1];
        temp[num_elements-1] = a[std::min(end + i, array_size - 1)];
      }
    });

  } catch (Kalmar::runtime_exception e) {
    std::string err_str = e.what();
    pass = err_str.find("The number of work items") != std::string::npos &&
    err_str.find("per work group exceeds the limit") != std::string::npos;
  }

  return pass==true?0:1;
}
