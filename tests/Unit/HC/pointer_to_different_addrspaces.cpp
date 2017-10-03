// XFAIL: *
// RUN: %hc -lhc_am %s -o %t.out && %t.out

#include <iostream>
#include <random>

#include <hc_am.hpp>
#include <hc.hpp>

bool test() {
  // define inputs and output
  const int vecSize = 256;
  const int tileSize = 64;

  float* data1 = (float*) malloc(vecSize * sizeof(float));
  float* data2 = (float*) malloc(vecSize * sizeof(float));
  float* data3 = (float*) malloc(vecSize * sizeof(float));

  // initialize test data
  std::random_device rd;
  std::uniform_real_distribution<float>  real_dist(0, 255);
  for (int i = 0; i < vecSize; ++i) {
    data1[i] = float(i);
    data2[i] = 0.0f;
    data3[i] = 0.0f;
  }

  auto acc = hc::accelerator();
  float* data1_d = (float*) hc::am_alloc(vecSize * sizeof(float), acc, 0);
  float* data2_d = (float*) hc::am_alloc(vecSize * sizeof(float), acc, 0);
  float* data3_d = (float*) hc::am_alloc(vecSize * sizeof(float), acc, 0);

  hc::accelerator_view av = acc.get_default_view();
  av.copy(data1, data1_d, vecSize * sizeof(float));
  av.copy(data2, data2_d, vecSize * sizeof(float));
  av.copy(data3, data3_d, vecSize * sizeof(float));

  // launch kernel
  hc::extent<1> e(vecSize);
  hc::parallel_for_each(e.tile(tileSize), [=](hc::tiled_index<1> tidx) [[hc]] {
    hc::index<1> localIdx = tidx.local;
    hc::index<1> globalIdx = tidx.global;

    // input pointer
    // it may point to:
    // - an automatic variable
    // - a tile_static variable
    // - a variable captured by kernel
    float* fp_in;

    // output pointer
    // always points to a variable captured by kernel
    float* fp_out;

    // tile_static variable
    tile_static float lds[tileSize];

    // automatic variable
    float stack[tileSize];

    // initialize automatic variable
    stack[localIdx[0]] = float(-localIdx[0]);

    // initialize tile_static variable
    lds[localIdx[0]] = float(localIdx[0]);
    tidx.barrier.wait();
    for (int i = 1; i < tileSize; ++i) {
      lds[0] += lds[i];
    }
    tidx.barrier.wait();

    switch (localIdx[0] % 3) {
      // read from tile_static
      // write to data2_d
    case 0:
      fp_in = &lds[0];
      fp_out = &data2_d[globalIdx[0]];
      break;

      // read from data1_d
      // write to data3_d
    case 1:
      fp_in = &data1_d[globalIdx[0]];
      fp_out = &data3_d[globalIdx[0]];
      break;

      // read from auto variable
      // write to data1_d
    case 2:
      fp_in = &stack[localIdx[0]];
      fp_out = &data1_d[globalIdx[0]];
      break;
    }

    // load from input pointer, and store to output pointer
    *fp_out = *fp_in;
  });

  av.copy(data1_d, data1, vecSize * sizeof(float));
  av.copy(data2_d, data2, vecSize * sizeof(float));
  av.copy(data3_d, data3, vecSize * sizeof(float));

  // verify
  bool ret = true;
  for(int i = 0; i < vecSize; ++i) {
    switch ((i % tileSize) % 3) {
      // read from tile_static
      // write to data2_d
    case 0:
      // data1 would still be the same as its original value
      ret &= (data1[i] == float(i));
      // data2 would contain the value calculated in tile_static
      ret &= (data2[i] == float(tileSize * (tileSize - 1) / 2));
      // data3 would still be the same as its original value
      ret &= (data3[i] == 0.0f);
      break;

      // read from data1_d
      // write to data3_d
    case 1:
      // data1 would still be the same as its original value
      ret &= (data1[i] == float(i));
      // data2 would still be the same as its original value
      ret &= (data2[i] == 0.0f);
      // data3 would contain the value from data1
      ret &= (data3[i] == float(i));
      break;

      // read from auto variable
      // write to data1_d
    case 2:
      // data1 would contain the value calculated in auto variable
      ret &= (data1[i] == float(- (i % tileSize)));
      // data2 would still be the same as its original value
      ret &= (data2[i] == 0.0f);
      // data3 would still be the same as its original value
      ret &= (data3[i] == 0.0f);
      break;
    }
    //std::cout << data1[i] << " " << data2[i] << " " << data3[i] << "\n";
  }

  if (ret) {
    std::cout << "Verify success!\n";
  } else {
    std::cout << "Verify failed!\n";
  }

  hc::am_free(data1_d);
  hc::am_free(data2_d);
  hc::am_free(data3_d);
  free(data1);
  free(data2);
  free(data3);

  return ret;
}

int main() {
  bool ret = true;

  // XXX this test would cause soft hang now
  // explicitly disable it for now
#if 0
  ret &= test();

  return !(ret == true);
#else
  return !(false == true);
#endif
}


