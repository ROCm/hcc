// RUN: %hc -lhc_am %s -o %t.out && %t.out

#include <iostream>
#include <hc.hpp>
#include <hc_am.hpp>
#include <pinned_vector.hpp>

constexpr size_t small_size = 1024;
constexpr size_t gargantuan_size = static_cast<size_t>(-1);


bool test_data_ptr() {
  // pinned_vector<T> uses am_alloc for memory allocation, which inserts
  // the resulting pointer to pinned memory in an internal tracking structure.
  // Check if the vectors data() pointer is present in the tracking structure
  // with expected values for some of its attributes.
  hc::pinned_vector<char> v(small_size);
  hc::accelerator acc;
  hc::AmPointerInfo ap(nullptr, nullptr, 0, acc);

  if(am_memtracker_getinfo(&ap, v.data()) != AM_SUCCESS){
    std::cout << "pinned_vector memory not tracked by AmPointerTracker\n";
    return false;
  }

  if(ap._hostPointer != ap._devicePointer
     or ap._isInDeviceMem
     or not ap._isAmManaged){
    std::cout << "sanity check on tracked pinned_vector memory failed\n";
    return false;
  }

  std::cout << "sanity check on tracked pinned_vector memory passed\n";
  return true;
}


bool test_bad_alloc() {
  // am_alloc returns nullptr if memory allocation fails. Pinned_vector's allocator
  // is supposed to throw a bad_alloc in that case.
  try {
    hc::pinned_vector<char> v(gargantuan_size);
  }
  catch(std::bad_alloc& e){
    std::cout << "expected bad_alloc caught\n";
    return true;
  }

  std::cout << "expected bad_alloc not caught\n";
  return false;
}


int main() {
  bool ret = true;

  ret &= test_data_ptr();
  ret &= test_bad_alloc();

  return !(ret == true);
}

