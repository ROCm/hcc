
// RUN: %hc %s -o %t.out && %t.out
#include <hc.hpp>                                                               
                                                                                
// a test which deliberately dispatch multiple kernels in a number
// which exceeds the size of builtin kernarg pool
// the runtime is expected to enlarge the pool dynamically so all
// kernels could still be successfully dispatched

bool test(int N) {                                                                
  hc::array_view<int, 1> a(10);
  hc::array_view<int, 1> b(10);                                                                    
  for (int i = 0; i < 10; ++i) a[i] = (i + 1);
                                                                                
  while (N--) {                                                                 
    hc::parallel_for_each(hc::accelerator().get_default_view(),                 
                          hc::extent<1>(10),                                    
                          [=](hc::index<1> idx) __attribute((hc)) {          
      b(idx) = a(idx);                                                    
    });                                                                         
  }                                                                             
  hc::accelerator().get_default_view().wait();                                  
                                                                                
  bool ret = true;                                                              
  for (int i = 0; i < 10; i++) {                                                
    ret &= (b[i] == i+1);                                                       
  }                                                                             
  return ret;                                                                  
}

int main() {
  bool ret = true;

  ret &= test(65);
  ret &= test(129);
  ret &= test(193);
  ret &= test(257);
  ret &= test(1025);

  return !(ret == true);
}

