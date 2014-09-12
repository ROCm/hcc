// XFAIL: Linux
// RUN: %amp_device -D__GPU__ -Xclang -fhsa-ext %s -m32 -emit-llvm -c -S -O2 -o %t.ll && mkdir -p %t
// RUN: %clamp-device %t.ll %t/kernel.cl
// RUN: pushd %t && %embed_kernel kernel.cl %t/kernel.o && popd
// RUN: %cxxamp -Xclang -fhsa-ext %link %t/kernel.o %s -o %t.out && %t.out
#include <iostream>
#include <iomanip>
#include <atomic>
#include <thread>
#include <amp.h>

#define TEST_DEBUG 0

void put_ptr_a(void* addr) restrict(amp);
void put_ptr_b(void* addr) restrict(amp);
void put_ptr_c(void* addr) restrict(amp);

void* get_ptr_a() restrict(amp);

// An HSA version of C++AMP program
int main ()
{
  // define inputs and output
  const int vecSize = 1024;
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
#if TEST_DEBUG
              std::cout << std::dec << "tid: " << i << ", malloc(" << param << "), "
                << "ret: " << "0x" << std::setfill('0') << std::setw(2) << std::hex << result << "\n";
#endif
              table_d[i] = (void*)result;
            break;
            case 2: // free
#if TEST_DEBUG
              std::cout << std::dec << "tid: " << i << ", free(" << std::hex << param << ")\n";
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
  int n = 10;
  parallel_for_each(
    sum.get_extent(),
    [=](Concurrency::index<1> idx) restrict(amp) {

    put_ptr_a(ptr_a);
    put_ptr_b(ptr_b);
    put_ptr_c(ptr_c);

    int *fib = new int[n + 1];

    fib[0] = 0;
    fib[1] = 1;

    for (int i = 2; i <= n; i++)
    {
      fib[i] = fib[i-1] + fib[i-2];
    }

    sum[idx] = fib[n];
    delete[] fib;
  });

  // stop CPU thread
  done.store(true);
  cpu_thread.join();

#if TEST_DEBUG
  for (int i = 0; i < vecSize; i++)
  {
    printf("Fib[n] is %lu\n", sum[i]);
  }
#endif

  // Verify
  int *fibh = new int[n + 1];

  fibh[0] = 0;
  fibh[1] = 1;

  for (int i = 2; i <= n; i++)
  {
    fibh[i] = fibh[i-1] + fibh[i-2];
  }

  int ans = fibh[n];
  delete[] fibh;

  for (int i = 0; i < n; i++)
  {
    if (ans != sum[i]) {
      std::cout << "Verify failed!\n";
      return 1;
    }
  }

  std::cout << "Verify success!\n";
  return 0;
}

