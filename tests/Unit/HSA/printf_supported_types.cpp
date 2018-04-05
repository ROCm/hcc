// RUN: %hc %s -DHCC_ENABLE_ACCELERATOR_PRINTF -lhc_am -o %t.out && %t.out | %FileCheck %s

#include <cassert>
#include <hc.hpp>
#include <hc_printf.hpp>

// create 2 tiles of 64 threads
#define TILE (8)
#define GLOBAL (TILE*2)

/*
* Supported Types
*
* Pointer Types
* void*
* const void*
*
* Integer Types
* uint8_t, int8_t - unsigned char, char
* uint16_t, int16_t - unsigned short, short, uchar16_t, char16_t
* uint32_t, int32_t - unsigned int, int, unsigned long, long, uchar32_t, char32_t
* uint64_t, int64_t - 64 bit uint/ints
* unsigned long long, long long - at least 64 bits
*
* Floating Point Types
* half - 16 bit fp
* float - 32 bit fp
* double - 64 bit fp
*/

int main() {

  hc::parallel_for_each(hc::extent<1>(GLOBAL).tile(TILE), [=](hc::tiled_index<1> tidx) [[hc]] {

      // Pointer Types
      const char* str1 = "const char* %s: %03d\n";
      const char* const_char_str1 = "1 thread";
      hc::printf(str1, const_char_str1, tidx.global[0]);
      const char* str2 = "char* %s: %03d\n";
      hc::printf(str2, "2 thread", tidx.global[0]);

      // Integer Types
      const char* str3 = "uint8_t %c: %03d\n";
      char char3 = '3';
      hc::printf(str3, char3, tidx.global[0]);
      const char* str4 = "int8_t %c: %03d\n";
      signed char char4 = '4';
      hc::printf(str4, char4, tidx.global[0]);
      const char* str5 = "uint16_t %hu: %03d\n";
      unsigned short ushort5 = 5;
      hc::printf(str5, ushort5, tidx.global[0]);
      const char* str6 = "int16_t %hd: %03d\n";
      short short6 = 6;
      hc::printf(str6, short6, tidx.global[0]);
      const char* str7 = "uint32_t %lu: %03d\n";
      unsigned long ulong7 = 7;
      hc::printf(str7, ulong7, tidx.global[0]);
      const char* str8 = "int32_t %ld: %03d\n";
      long long8 = 8;
      hc::printf(str8, long8, tidx.global[0]);
      const char* str9 = "uint64_t %llu: %03d\n";
      uint64_t uint64_9 = 9;
      hc::printf(str9, uint64_9, tidx.global[0]);
      const char* str10 = "int64_t %lld: %03d\n";
      int64_t int64_10 = 10;
      hc::printf(str10, int64_10, tidx.global[0]);
      const char* str11 = "ull %llu: %03d\n";
      unsigned long long ull_11 = 11;
      hc::printf(str11, ull_11, tidx.global[0]);
      const char* str12 = "ll %lld: %03d\n";
      long long ll_12  = 12;
      hc::printf(str12, ll_12, tidx.global[0]);

      // Floating Point Types
      const char* str13 = "half %f: %03d\n";
      hc::half half_13 = 13.13;
      //hc::printf(str13, half_13, tidx.global[0]);
      const char* str14 = "float %2.2f: %03d\n";
      float float_14 = 14.14;
      hc::printf(str14, float_14, tidx.global[0]);
      const char* str15 = "double %2.2f: %03d\n";
      double double_15 = 15.15;
      hc::printf(str15, double_15, tidx.global[0]);
  }).wait();

  return 0;
}


// CHECK-DAG: const char* 1 thread: 000
// CHECK-DAG: char* 2 thread: 000
// CHECK-DAG: uint8_t 3: 000
// CHECK-DAG: int8_t 4: 000
// CHECK-DAG: uint16_t 5: 000
// CHECK-DAG: int16_t 6: 000
// CHECK-DAG: uint32_t 7: 000
// CHECK-DAG: int32_t 8: 000
// CHECK-DAG: uint64_t 9: 000
// CHECK-DAG: int64_t 10: 000
// CHECK-DAG: ull 11: 000
// CHECK-DAG: ll 12: 000
// TODO: Add half support
// CHECK-DAG: float 14.14: 000
// CHECK-DAG: double 15.15: 000
