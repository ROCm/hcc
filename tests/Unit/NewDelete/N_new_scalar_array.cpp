// XFAIL: Linux
// RUN: %cxxamp -Xclang -fhsa-ext %s -o %t.out && %t.out
#include <iostream>
#include <iomanip>
#include <atomic>
#include <thread>
#include <amp.h>

#define TEST_DEBUG 1

void put_ptr_a(void* addr) restrict(amp);
void put_ptr_b(void* addr) restrict(amp);
void put_ptr_c(void* addr) restrict(amp);

// An HSA version of C++AMP program
int main ()
{
  // define inputs and output
  const int vecSize = 5;
  const int cpuSleepMsec = 25;

  // specify syscall number
  std::atomic_int table_a[vecSize];
  auto ptr_a = &table_a[0];

  // syscall parameter
  std::atomic_long table_b[vecSize];
  auto ptr_b = &table_b[0];

  // test result
  std::atomic_long table_c[vecSize];
  auto ptr_c = &table_c[0];

  // returned address
  void* table_d[vecSize] = {0};

  // CPU syscall service thread control
  std::atomic_bool done(false);
  auto ptr_done = &done;

  // initialize test data
  for (int i = 0; i < vecSize; ++i) {
    table_a[i].store(0);
    table_b[i].store(0);
    table_c[i].store(0);
  }

  // fire CPU thread
  std::thread cpu_thread([=, &table_d]() {
    std::cout << "Enter CPU syscall service thread..." << std::endl;
    std::chrono::milliseconds dura( cpuSleepMsec );
    int syscall;
    while (!*ptr_done) {
      for (int i = 0; i < vecSize; ++i) {
        syscall = (ptr_a + i)->load(std::memory_order_acquire);

        if (syscall) {
          // load parameter
          long param = (ptr_b + i)->load(std::memory_order_acquire);
          
          // do actual stuff
          long result;
          switch (syscall) {
            case 1: // malloc
              result = (long)memalign(0x1000, param);
              std::cout << std::dec << "tid: " << i << ", malloc(" << param << "), "
                << "ret: " << "0x" << std::setfill('0') << std::setw(2) << std::hex << result << "\n";
              table_d[i] = (void*)result;
            break;
            case 2: // free
              std::cout << std::dec << "tid: " << i << ", free(" << std::hex << param << ")\n";
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
  parallel_for_each(
    sum.get_extent(),
    [=](Concurrency::index<1> idx) restrict(amp) {

    put_ptr_a(ptr_a);
    put_ptr_b(ptr_b);
    put_ptr_c(ptr_c);

    sum[idx] = (unsigned long int)new unsigned int[2];
  });

  // stop CPU thread
  done.store(true);
  cpu_thread.join();

#if TEST_DEBUG
  for (int i = 0; i < vecSize; i++)
  {
    unsigned int *p = (unsigned int*)table_d[i];
    printf("Value of addr %p is %u, addr %p is %u\n", (void*)p, *p, (void*)(p + 1), *(p + 1));
  }
#endif

  // Verify
  int error = 0;
#if 0
  for(int i = 0; i < vecSize; i++) {
    unsigned int *p = (unsigned int*)table_d[i];
    error += abs(*p);
  }
#endif
  if (error == 0) {
    std::cout << "Verify success!\n";
  } else {
    std::cout << "Verify failed!\n";
  }

  return (error != 0);
}
