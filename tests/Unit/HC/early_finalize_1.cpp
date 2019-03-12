// RUN: %hc -D_OBJ_1 %s -c -o %T/early_finalize_1.obj1.o -fno-gpu-rdc
// RUN: %hc -D_OBJ_2 %s -c -o %T/early_finalize_1.obj2.o -fno-gpu-rdc
// RUN: %hc -D_OBJ_3 %s -c -o %T/early_finalize_1.obj3.o -fno-gpu-rdc
// RUN: c++ %T/early_finalize_1.obj1.o %T/early_finalize_1.obj2.o  %T/early_finalize_1.obj3.o -L %lib_output_path -Wl,-rpath=%lib_output_path -lmcwamp -o %t.out && %t.out

#include <hc.hpp>


extern int sum(hc::array_view<int,1>& input);
extern int neg_sum(hc::array_view<int,1>& input);

#ifdef _OBJ_1
int sum(hc::array_view<int,1>& input) {

  hc::array_view<int,1> s(1);
  s[0]=0;

  hc::parallel_for_each(input.get_extent(), [=](hc::index<1> idx) [[hc]] {
    if (idx[0]==0) {
      int num = input.get_extent()[0];
      for (int i = 0; i < num; i++) {
        s[0]+=input[i];
      }
    }
  }).wait();

  return s[0];
}
#endif


#ifdef _OBJ_2
int neg_sum(hc::array_view<int,1>& input) {

  hc::array_view<int,1> s(1);
  s[0]=0;

  hc::parallel_for_each(input.get_extent(), [=](hc::index<1> idx) [[hc]] {
    if (idx[0]==0) {
      int num = input.get_extent()[0];
      for (int i = 0; i < num; i++) {
        s[0]+=-input[i];
      }
    }
  }).wait();

  return s[0];
}
#endif

#ifdef _OBJ_3
int main() {
  hc::array_view<int,1> av(64);
  for (int i = 0;i < 64; i++)
    av[i] = i;

  int s = sum(av);
  int ns = neg_sum(av);

  int host_s = 0;
  int host_ns = 0;
  for (int i = 0; i < av.get_extent()[0]; ++i) {
    host_s+=av[i];
    host_ns+=-av[i];
  }

#if 0
  printf("sum:  expected=%d, actual=%d\n", host_s, s);
  printf("neg_sum:  expected=%d, actual=%d\n", host_ns, ns);
#endif

  return (s==host_s && ns==host_ns)?0:1;
}
#endif
