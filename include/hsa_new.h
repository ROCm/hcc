#ifndef _HSA_NEW_H
#define _HSA_NEW_H

#include <iostream>
#include <iomanip>
#include <atomic>
#include <thread>

// Use HSA's API
#include "hsa.h"

#define DEBUG 1
#define USE_SIGNAL 1

#define check(msg, status) \
if (status != HSA_STATUS_SUCCESS) { \
    printf("%s failed with error %x.\n", #msg, status); \
    exit(1); \
} else { \
   printf("%s succeeded.\n", #msg); \
}

/// Helper rountines declaration - These pass the data structure used by 
/// new/delet into HSA side. Function body is implemented in 
/// lib/new_delete.hsail
///
/// Note that these routines needs called when you enter the amp restriced 
/// lambda because of the HSA's alloc qualifier issue

void putXmallocFlag(hsa_signal_t handle) restrict(amp);
void putMallocFlag(hsa_signal_t handle) restrict(amp);
void put_ptr_a(void* addr) restrict(amp);
void put_ptr_b(void* addr) restrict(amp);
void put_ptr_c(void* addr) restrict(amp);
void put_ptr_x(void* addr) restrict(amp);
void put_ptr_y(void* addr) restrict(amp);
void put_ptr_z(void* addr) restrict(amp);

class NewInit {
public:
  /// Constants
  static const int cpuSleepMsec = 10; //25;

  static const int max_vec_size = 409600;
  static const int max_tile_size = 256;
  static const int max_tile_count = max_vec_size;

  static const hsa_signal_value_t Halt = 0;
  static const hsa_signal_value_t Exit = -1;

  // handle of signals
  hsa_signal_t XmallocFlag;
  hsa_signal_t mallocFlag;

  // pointer to syscall numbers
  std::atomic_int *ptr_a;
  std::atomic_int *ptr_x;

  // pointer to syscall parameters
  std::atomic_long *ptr_b;
  std::atomic_long *ptr_y;

  // pointer to test results
  std::atomic_long *ptr_c;
  std::atomic_long *ptr_z;

  NewInit ();
  ~NewInit ();

  int get_Xmalloc_count() { return Xmalloc_count; }
  int get_malloc_count() { return malloc_count; }  
  int get_Xfree_count() { return Xfree_count; }

private:
  // specify syscall number
  std::atomic_int table_a[max_tile_count]; // Xmalloc thread
  std::atomic_int table_x[max_vec_size];  // malloc/free/Xfree thread

  // syscall parameter
  std::atomic_long table_b[max_tile_count]; // Xmalloc thread
  std::atomic_long table_y[max_vec_size]; // malloc/free/Xfree thread

  // test result
  std::atomic_long table_c[max_vec_size]; // Xmalloc thread
  std::atomic_long table_z[max_vec_size]; // malloc/free/Xfree thread

  // Thread entities
  std::thread Xmalloc_thread;
  std::thread malloc_thread;

  int Xmalloc_count;
  int malloc_count;
  int Xfree_count;

  void XmallocThread();
  void mallocThread();
};

NewInit newInit;


/// NewInit's implementation

NewInit::NewInit() {
  hsa_init();

  hsa_status_t err;
  err = hsa_signal_create(0, 0, NULL, &XmallocFlag);
  check(Creating a HSA signal used to control Xmalloc thread, err);
  err = hsa_signal_create(0, 0, NULL, &mallocFlag);
  check(Creating a HSA signal used to control malloc thread, err);

  // initialize
  Xmalloc_count = 0;
  malloc_count = 0;
  Xfree_count = 0;

  for (int i = 0; i < max_tile_count; ++i) {
    table_a[i].store(0);
    table_b[i].store(0);
  }

  for (int i = 0; i < max_vec_size; ++i) {
    table_c[i].store(0);
    table_x[i].store(0);
    table_y[i].store(0);
    table_z[i].store(0);
  }

  ptr_a = &table_a[0];
  ptr_b = &table_b[0];
  ptr_c = &table_c[0];
  ptr_x = &table_x[0];
  ptr_y = &table_y[0];
  ptr_z = &table_z[0];

  // fire CPU thread
  Xmalloc_thread = std::thread(&NewInit::XmallocThread, this);
#if !USE_SIGNAL
  Xmalloc_thread.detach();
#endif
  malloc_thread = std::thread(&NewInit::mallocThread, this);
#if !USE_SIGNAL
  malloc_thread.detach();
#endif
}

NewInit::~NewInit() {
  hsa_signal_store_release(XmallocFlag, Exit);
  hsa_signal_store_release(mallocFlag, Exit);

#if USE_SIGNAL
  Xmalloc_thread.join();
  malloc_thread.join();
#endif

  hsa_signal_destroy(XmallocFlag);
  hsa_signal_destroy(mallocFlag);

  hsa_shut_down();
}

void NewInit::XmallocThread() {
  std::cout << "Enter Xmalloc syscall service thread..." << std::endl;

  std::chrono::milliseconds dura(cpuSleepMsec);
  int syscall;
  int XmallocIterates = 0;
  while (true) {

#if USE_SIGNAL
    hsa_signal_value_t XmallocWaitRet;

    while ((XmallocWaitRet = hsa_signal_wait_acquire(XmallocFlag, HSA_NE,
      Halt, UINT64_MAX, HSA_WAIT_EXPECTANCY_UNKNOWN)) == 0);

    if (XmallocWaitRet == Exit)
      break;
#endif

#if DEBUG
    std::cout << "Xmalloc Thread iterates ... " << std::dec << ++XmallocIterates << std::endl;
#endif

    for (int i = 0; i < max_tile_count; ++i) {
      syscall = (ptr_a + i)->load(std::memory_order_acquire);

      if (syscall) {
        Xmalloc_count++;
        // load parameter
        long param = (ptr_b + i)->load(std::memory_order_acquire);

        // do actual stuff
        long result;
        switch (syscall) {
          case 1: { // malloc
            result = (long)malloc(param);
#if DEBUG
            std::cout << std::dec << "malloc(" << param << "), "
              << "ret: " << "0x" << std::setfill('0') << std::setw(2) << std::hex << result << "\n";
#endif
            break;
          }
        }

        // store result
        (ptr_b + i)->store(result, std::memory_order_release);

        // reset flag
        (ptr_a + i)->store(0, std::memory_order_release);
      }

    }
    std::this_thread::sleep_for(dura);
  }

  std::cout << "Leave Xmalloc syscall service thread." << std::endl;
}

void NewInit::mallocThread() {
  std::cout << "Enter malloc/free/Xfree service thread..." << std::endl;

  std::chrono::milliseconds dura(cpuSleepMsec);
  int syscall;
  int mallocIterates = 0;
  while (true) {

#if USE_SIGNAL
    hsa_signal_value_t mallocWaitRet;

    while ((mallocWaitRet = hsa_signal_wait_acquire(mallocFlag, HSA_NE,
      Halt, UINT64_MAX, HSA_WAIT_EXPECTANCY_UNKNOWN)) == 0);

    if (mallocWaitRet == Exit)
      break;
#endif

#if DEBUG
    std::cout << "malloc Thread iterates ... " << std::dec << ++mallocIterates << std::endl;
#endif

    for (int i = 0; i < max_vec_size; ++i) {
      syscall = (ptr_x + i)->load(std::memory_order_acquire);

      if (syscall) {
        // load parameter
        long param = (ptr_y + i)->load(std::memory_order_acquire);

        // do actual stuff
        long result;
        switch (syscall) {
          case 1: { // malloc
            malloc_count++;
            result = (long)malloc(param);
#if DEBUG
            std::cout << std::dec << "malloc(" << param << "), "
              << "ret: " << "0x" << std::setfill('0') << std::setw(2) << std::hex << result << "\n";
#endif
            break;
          }
          case 2: { // Xfree/free
            char *alloc = (char *)param;
            if (alloc == NULL) break;
            int *p_header = (int *)alloc - 1;
            int header_offset = *p_header;
            int *p_counter = (int *)((char *)p_header - header_offset);

            *p_counter -= 1;
#if DEBUG
            std::cout << "param(alloc): " << std::hex << (void *)param << ", "
              << "value in *alloc: " << std::dec << *((int *)alloc) << ", "
              << "p_header: " << std::hex << p_header << ", "
              << "header_offset: " << std::dec << header_offset << ", "
              << "p_counter: " << std::dec << (void *)p_counter << ", "
              << "counter: " << std::dec << *p_counter << "\n";
#endif
            if (*p_counter == 0) {
              Xfree_count++;
              free ((void *)p_counter);
#if DEBUG
              std::cout << "free: " << std::hex << (void *)p_counter << "\n";
#endif
              break;
            }
            result = 0;
          }
        }

        // store result
        (ptr_y + i)->store(result, std::memory_order_release);

        // reset flag
        (ptr_x + i)->store(0, std::memory_order_release);
      }

    }
    std::this_thread::sleep_for(dura);
  }

  std::cout << "Leave malloc/free/Xfree syscall service thread." << std::endl;
}

#undef DEBUG

#endif
