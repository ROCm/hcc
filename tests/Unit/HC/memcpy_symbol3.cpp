
// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>

#include <iostream>

#define GRID_SIZE (16)

// globalVar would be agent-allocated global variable with program linkage
// add one initial value to prevent a bug in HLC
[[hc]] float tableGlobal[GRID_SIZE] = { 0.1 };

using namespace hc;

bool test1() {

  bool ret = true;

  // array which would be copied into the global variable array
  float tableInput[GRID_SIZE] { 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };

  // array to store the outputs from the kernel
  array_view<float, 1> tableOutput1(GRID_SIZE);

  // array to store the result copied from device memory
  float tableOutput2[GRID_SIZE] { 0 };

  // use hc::accelerator::memcpySymbol() to copy testValue to globalVar
  // get the default accelerator
  accelerator acc = accelerator();
  acc.memcpy_symbol("tableGlobal", tableInput, sizeof(float) * GRID_SIZE);

  // dispatch a kernel which reads from globalVar and stores result to table1
  extent<1> ex(GRID_SIZE);
  completion_future fut = parallel_for_each(ex, [=](index<1>& idx) __attribute__((hc)) {
    tableOutput1(idx) = tableGlobal[idx[0]];
  });

  // wait for the kernel to be completed
  fut.wait();

  // copy data from device -> host
  acc.memcpy_symbol("tableGlobal", tableOutput2, sizeof(float) * GRID_SIZE, 0, hcMemcpyDeviceToHost);

  // read out the outputs, it should agree with testValue
  for (int i = 0; i < GRID_SIZE; ++i) {
    ret &= (tableInput[i] == tableOutput1[i]);
    ret &= (tableInput[i] == tableOutput2[i]);
  } 

  return ret;
}

int main() {
  bool ret = true;

  ret &= test1();

  return !(ret == true);
}

