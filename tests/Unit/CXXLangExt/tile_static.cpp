

// RUN: %hc -DTYPE="char"  %s -o %t.out && %t.out
// RUN: %hc -DTYPE="signed char"  %s -o %t.out && %t.out
// RUN: %hc -DTYPE="unsigned char"  %s -o %t.out && %t.out

// RUN: %hc -DTYPE="short"  %s -o %t.out && %t.out
// RUN: %hc -DTYPE="signed short"  %s -o %t.out && %t.out
// RUN: %hc -DTYPE="unsigned short"  %s -o %t.out && %t.out

#include <hc.hpp>
#include <iostream>

// A test which uses different types not allowed in C++AMP specification
// as tile_static arrays.  In HC mode they are available.
int main() {
  bool ret = true;

  // define grid size and group size
  const int vecSize = 2048;
  const int groupSize = 256;

  int table[vecSize] { 0 };
  hc::array_view<int, 1> av(vecSize, table);

  hc::tiled_extent<1> e(vecSize, groupSize);
  hc::completion_future fut = hc::parallel_for_each(e, [=](hc::tiled_index<1> tidx) __attribute__((hc)) {
    // create a tile_static array
    tile_static TYPE group[groupSize];
    group[tidx.local[0]] = tidx.local[0];

    tidx.barrier.wait();

    av(tidx.global[0]) = (int)group[tidx.local[0]];
  });
  
  fut.wait();
  av.synchronize();

  // verify
  int error = 0;
  for (unsigned i = 0; i < vecSize; ++i) {
    //std::cout << table[i] << " ";
    error += std::abs(table[i] - (TYPE)(i % groupSize));
    //if ((i + 1) % groupSize == 0)
    //  std::cout << "\n";
  }

  return (error != 0);
}

