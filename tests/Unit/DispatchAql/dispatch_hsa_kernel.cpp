// RUN: %hc %s %S/hsacodelib.CPP -I/opt/rocm/include -L/opt/rocm/lib -lhsa-runtime64 -lhc_am -o %t.out && %t.out %S/vcpy_isa.hsaco

#include <hc.hpp>

#include <iostream>
#include <random>

//#include <hsa/hsa.h>
#include <hsa/hsa.h>

#include "hsacodelib.h"
#include <hc_am.hpp>

int p_db = 1;
int p_wait = 1;

const char *hsaco_filename = NULL;

// An example which shows how to use accelerator_view::create_blocking_marker(completion_future&)
///
/// The test case only works on HSA because it directly uses HSA runtime API
/// It would use completion_future::get_native_handle() to retrieve the
/// underlying hsa_signal_t data structure to query if dependent kernels have
/// really finished execution before the new kernel is executed.
bool test() {
  bool ret = true;


  hc::accelerator acc = hc::accelerator();
  hc::accelerator_view av = acc.get_default_view();

  Kernel k = load_hsaco(&av, hsaco_filename, "hello_world");


  //int bufferElements = 1024*1024;
  int bufferElements = 1024;
  int groupSize      = 1024;
  assert(bufferElements <= groupSize); // limitation of the kernel used in the test
  int bufferSize = bufferElements * sizeof(float);
  float *in_h  = (float*)malloc(bufferSize);
  float *out_h = (float*)malloc(bufferSize);
  float *in_d  = hc::am_alloc(bufferSize, acc, 0);
  float *out_d = hc::am_alloc(bufferSize, acc, 0);

  const float expected = 42.0;
  for (int i=0; i<bufferElements; i++) {
      in_h[i] = expected;
      out_h[i] = 13.0;
  }


  hsa_kernel_dispatch_packet_t dispatch_packet;
  memset(&dispatch_packet, 0, sizeof(dispatch_packet));

  dispatch_packet.setup  = 1 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
  dispatch_packet.grid_size_x = (uint32_t) bufferElements;
  dispatch_packet.grid_size_y = 1;
  dispatch_packet.grid_size_z = 1;
  dispatch_packet.workgroup_size_x = (uint16_t)groupSize;
  dispatch_packet.workgroup_size_y = (uint16_t)1;
  dispatch_packet.workgroup_size_z = (uint16_t)1;
  dispatch_packet.completion_signal.handle = 0; //signal;
  dispatch_packet.kernel_object = k._kernelCodeHandle;
  dispatch_packet.kernarg_address = nullptr;  
  dispatch_packet.private_segment_size = k._privateSegmentSize;
  dispatch_packet.group_segment_size = k._groupSegmentSize;

  uint16_t header;
  header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
  header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;
  header |= HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  //header |= (1 << HSA_PACKET_HEADER_BARRIER);
  dispatch_packet.header = header;

#define USE_HIDDEN
  struct __attribute__ ((aligned(16))) args_t {
#ifdef USE_HIDDEN
      uint32_t hidden[6];
#endif
      float* in;
      float* out;
  } args;

#ifdef USE_HIDDEN
  args.hidden[0]=dispatch_packet.grid_size_x;
  args.hidden[1]=1;
  args.hidden[2]=1;
  args.hidden[3]=dispatch_packet.workgroup_size_x;
  args.hidden[4]=1;
  args.hidden[5]=1;
#endif

  args.in=in_d;
  args.out=out_d;


  av.copy(in_h, in_d, bufferSize);

  if (p_db) {
    printf ("info: calling dispatch\n");
  }
 
  // TODO - test completion-future case: 
  av.dispatch_hsa_kernel(&dispatch_packet, &args, sizeof(args), NULL/*completion-future*/);

  if (p_wait) {
    av.wait();
    printf ("warning: waiting...\n");
  }

  if (p_db) {
    printf ("info: dispatch finished, copy back results\n");
  }

  av.copy(out_d, out_h, bufferSize);

  if (p_db) {
    printf ("info: results copied back, performing check\n");
  }


  for (int i=0; i<bufferElements; i++) {
    if (out_h[i] != expected) {
      printf ("mismatch at element=%d, %f != expected %f\n", i, out_h[i], expected); 
      ret = false;
      break;
    }
  }
  
  return ret;
}


bool test_negative() 
{
  int exceptionsToCatch = 2;

  hc::accelerator acc = hc::accelerator();
  hc::accelerator_view av = acc.get_default_view();

  hsa_kernel_dispatch_packet_t dispatch_packet;
  memset(&dispatch_packet, 0, sizeof(dispatch_packet));

  try {
      av.dispatch_hsa_kernel(&dispatch_packet, nullptr, 0, NULL);
  }
  //catch (Kalmar::runtime_exception &e) {
  catch (std::exception &e) {
      exceptionsToCatch--;
      std::cout << "info: successfully caught exception=" << "\n";
  }


  dispatch_packet.setup =  1 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;

  try {
      av.dispatch_hsa_kernel(&dispatch_packet, nullptr, 0, NULL);
  }
  //catch (Kalmar::runtime_exception &e) {
  catch (std::exception &e) {
      exceptionsToCatch--;
      std::cout << "info: successfully caught exception=" << "\n";
  }

  return (exceptionsToCatch == 0);;

}



int main(int argc, char* argv[]) {
  bool success = true;

  if(argc > 1) {
    hsaco_filename = argv[1];
  } else {
      printf ("error - usage: %s HSACO_FILE\n", argv[0]);
      assert(0);
  }

  success &= test_negative();

  success &= test();

  return !(success == true);
}

