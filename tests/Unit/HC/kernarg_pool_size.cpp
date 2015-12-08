// XFAIL: Linux,boltzmann
// RUN: %hc %s -o %t.out && %t.out
#include <hc.hpp>                                                               
                                                                                
// a test which deliberately dispatch multiple kernels in a number
// which exceeds the size of builtin kernarg pool
// the runtime is expected to enlarge the pool dynamically so all
// kernels could still be successfully dispatched

bool test(int N) {                                                                
  int a[10] = {1,2,3,4,5,6,7,8,9,10};                                           
  int b[10];                                                                    
  int *s = a;                                                                   
  int *d = b;                                                                   
                                                                                
  while (N--) {                                                                 
    hc::parallel_for_each(hc::accelerator().get_default_view(),                 
                          hc::extent<1>(10),                                    
                          [s, d](hc::index<1> idx) __attribute((hc)) {          
      d[idx[0]] = s[idx[0]];                                                    
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

