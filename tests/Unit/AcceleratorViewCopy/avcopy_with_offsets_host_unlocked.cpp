// RUN: %hc %s -o %t.out -lhc_am && %t.out

// Test hc::acclerator_view::copy()
// with GPU buffers having offsets from the result of am_alloc
// with CPU buffers un-locked, and have offsets

#include <hc.hpp>
#include <hc_am.hpp>

void rocm_device_synchronize()
{
  hc::accelerator_view av = hc::accelerator().get_default_view();
  hc::completion_future fut = av.create_marker();
  fut.wait();
}
void deep_copy(void * dst, const void * src, size_t n) {
  hc::accelerator_view av = hc::accelerator().get_default_view();
  av.copy( src , dst , n*sizeof(int));
}

template<int N, int offset>
bool test() {
  bool ret = true;

  hc::accelerator acc;

  // ap, bp are device memory buffers
  int * ap = hc::am_alloc((offset+N)*sizeof(int),acc,0);
  int * bp = hc::am_alloc((offset+N)*sizeof(int),acc,0);

  // a, b are device memory with offsets
  int * a = &ap[offset];
  int * b = &bp[offset];

  // h_ap, h_ap are un-locked host memory buffers
  int * h_ap = (int *)malloc((offset+N)*sizeof(int));
  int * h_bp = (int *)malloc((offset+N)*sizeof(int));

  // h_a, h_b are un-locked host memory with offsets
  int * h_a = &h_ap[offset];
  int * h_b = &h_bp[offset];

  // initialize h_a, h_b
  for(int i=0; i<N; i++)
    h_a[i] = i+1;
  for(int i=0; i<N; i++)
    h_b[i] = 5555;

  // execute a kernel to populate data on GPU
  hc::extent<1> e(N);
  hc::parallel_for_each(e,[=](hc::index<1> idx)__HC__{
    a[idx[0]] = 5;
  });

  // test 1. D2H copy
  deep_copy(h_b,a,N);

  for(int i=0; i<N; i++) {
    if (h_a[i] != (i+1))
      ret = false;
    if (h_b[i] != 5)
      ret = false;
  }

  // test 2. H2D followed by D2D and then D2H copy
  deep_copy(a,h_a,N);
  deep_copy(b,a,N);
  deep_copy(h_b,b,N);

  for(int i=0; i<N; i++) {
    if (h_a[i] != (i+1))
      ret = false;
    if (h_b[i] != (i+1))
      ret = false;
  }

  // test 3. do another D2H
  deep_copy(h_b,a,N);

  for(int i=0; i<N; i++) {
    if (h_a[i] != (i+1))
      ret = false;
    if (h_b[i] != (i+1))
      ret = false;
  }

  // flush GPU queue
  rocm_device_synchronize();

  // release resources
  free(h_ap);
  free(h_bp);
  hc::am_free(ap);
  hc::am_free(bp);

  return ret;
}

int main() {
  bool ret = true;

  ret &= test<64, 0>();
  ret &= test<64, 64>();
  ret &= test<1024, 0>();
  ret &= test<1024, 17>();
  ret &= test<1024, 32>();
  ret &= test<1024, 129>();
  ret &= test<1024, 256>();
  ret &= test<1024, 513>();
  ret &= test<1024, 1024>();

  return !(ret == true);
}
