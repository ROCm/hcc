// XFAIL: Linux
// RUN: %cxxamp -Xclang -fhsa-ext %s -o %t.out && %t.out
#include <iostream>
#include <iomanip>
#include <atomic>
#include <thread>
#include <amp.h>
#include <ctime>
#include "point.h"

#define DEBUG 1

void put_ptr_a(void* addr) restrict(amp);
void put_ptr_b(void* addr) restrict(amp);
void put_ptr_c(void* addr) restrict(amp);

// An HSA version of C++AMP program
int main ()
{
  // define inputs and output
  const int vecSize = 16;
  const int tileSize = 4;
  const int tileCount = vecSize / tileSize;
  const int cpuSleepMsec = 25;

  #define MAX_VEC_SIZE 10240
  #define MAX_TILE_SIZE 256
  #define MAX_TILE_COUNT MAX_VEC_SIZE

  // specify syscall number
  std::atomic_int table_a[MAX_TILE_COUNT];
  auto ptr_a = &table_a[0];

  // syscall parameter
  std::atomic_long table_b[MAX_TILE_COUNT];
  auto ptr_b = &table_b[0];

  // test result
  std::atomic_long table_c[MAX_VEC_SIZE];
  auto ptr_c = &table_c[0];

  // CPU syscall service thread control
  std::atomic_bool done(false);
  auto ptr_done = &done;

  // initialize test data
  for (int i = 0; i < MAX_TILE_COUNT; ++i) {
    table_a[i].store(0);
    table_b[i].store(0);
  }

  for (int i = 0; i < MAX_VEC_SIZE; ++i) {
    table_c[i].store(0);
  }

  int syscall_count = 0;

  // fire CPU thread
  std::thread cpu_thread([=, &syscall_count]() {
    std::cout << "Enter CPU syscall service thread..." << std::endl;
    std::chrono::milliseconds dura( cpuSleepMsec );
    int syscall;
    while (!*ptr_done) {
      for (int i = 0; i < MAX_TILE_COUNT; ++i) {
        syscall = (ptr_a + i)->load(std::memory_order_acquire);

        if (syscall) {
          syscall_count++;
          // load parameter
          long param = (ptr_b + i)->load(std::memory_order_acquire);

          // do actual stuff
          long result;
          switch (syscall) {
            case 1: // malloc
              result = (long)malloc(param);
#if DEBUG
              std::cout << std::dec << "malloc(" << param << "), "
                << "ret: " << "0x" << std::setfill('0') << std::setw(2) << std::hex << result << "\n";
#endif
            break;
            case 2: // free
#if DEBUG
              std::cout << std::dec << ", free(" << std::hex << param << ")\n";
#endif
              free((void*)param);
              result = 0;
            break;
          }

          // store result
          (ptr_b + i)->store(result, std::memory_order_release);

          // reset flag
          (ptr_a + i)->store(0, std::memory_order_release);
        }

      }
        std::this_thread::sleep_for( dura );
    }
      std::cout << "Leave CPU syscall service thread." << std::endl;
  });

  // launch kernel
  unsigned long int sumCPP[vecSize];
  Concurrency::array_view<unsigned long int, 1> sum(vecSize, sumCPP);

  clock_t m_start = clock();
  parallel_for_each(
    Concurrency::extent<1>(vecSize),
    [=](Concurrency::index<1> idx) restrict(amp) {

    put_ptr_a(ptr_a);
    put_ptr_b(ptr_b);
    put_ptr_c(ptr_c);

    sum[idx[0]] = (unsigned long int)new Point[2]();
  });
  clock_t m_stop = clock();

  // stop CPU thread
  done.store(true);
  cpu_thread.join();

#if DEBUG
  for (int i = 0; i < vecSize; i++)
  {
    Point *p = (Point *)sum[i];
    printf("Value of addr %p is %d & %d, addr %p is %d & %d\n",
      (void*)p, p->get_x(), p->get_y(),
      (void*)(p + 1), (p + 1)->get_x(), (p + 1)->get_y());
  }
#endif

  // Verify
  int error = 0;
  for(int i = 0; i < vecSize; i++) {
    Point *p = (Point*)sum[i];
    Point pt;
    error += (abs(p->get_x() - pt.get_x()) + abs(p->get_y() - pt.get_y())
               + abs((p + 1)->get_x() - pt.get_x()) + abs((p + 1)->get_y() - pt.get_y()));
  }
  if (error == 0) {
    std::cout << "Verify success!\n";
  } else {
    std::cout << "Verify failed!\n";
  }
  clock_t t1 = clock();
  clock_t t2 = clock();
  clock_t m_overhead = t2 - t1;
  double elapsed = ((double)(m_stop - m_start - m_overhead)) / CLOCKS_PER_SEC;
  std::cout << "Execution time of amp restrict lambda is " << std::dec << elapsed << " s.\n";
  std::cout << "System call is executed " << std::dec << syscall_count << " times\n";
  return (error != 0);
}

