#ifndef _HSA_NEW_H
#define _HSA_NEW_H

#include <atomic>
#include <thread>

#define DEBUG 1

/// Helper rountines declaration - These pass the data structure used by 
/// new/delet into HSA side. Function body is implemented in 
/// lib/new_delete.hsail
///
/// Note that these routines needs called when you enter the amp restriced 
/// lambda because of the HSA's alloc qualifier issue

void put_ptr_a(void* addr) restrict(amp);
void put_ptr_b(void* addr) restrict(amp);
void put_ptr_c(void* addr) restrict(amp);

class NewInit {
public:
  /// Constants
  static const int cpuSleepMsec = 25;

  static const int max_vec_size = 10240;
  static const int max_tile_size = 256;
  static const int max_tile_count = max_vec_size;

  // pointer to syscall numbers
  std::atomic_int *ptr_a;

  // pointer to syscall parameters
  std::atomic_long *ptr_b;

  // pointer to test results
  std::atomic_long *ptr_c;

  NewInit ();

  int get_syscall_count() { return syscall_count; }
  
private:
  // specify syscall number
  std::atomic_int table_a[max_tile_count];

  // syscall parameter
  std::atomic_long table_b[max_tile_count];

  // test result
  std::atomic_long table_c[max_vec_size];

  // CPU syscall service thread control
  std::atomic_bool done;

  // Thread entities
  std::thread Xmalloc_thread;

  int syscall_count;

  void XmallocThread();
};

NewInit newInit;


/// NewInit's implementation

NewInit::NewInit() {
  // initialize
  for (int i = 0; i < max_tile_count; ++i) {
    table_a[i].store(0);
    table_b[i].store(0);
  }

  for (int i = 0; i < max_vec_size; ++i) {
    table_c[i].store(0);
  }

  ptr_a = &table_a[0];
  ptr_b = &table_b[0];
  ptr_c = &table_c[0];
  done.store(false);

  // fire CPU thread
  Xmalloc_thread = std::thread(&NewInit::XmallocThread, this);
  Xmalloc_thread.detach();
}

void NewInit::XmallocThread() {
  std::cout << "Enter Xmalloc syscall service thread..." << std::endl;
  std::chrono::milliseconds dura(cpuSleepMsec);
  int syscall;
  while (!done) {
    for (int i = 0; i < max_tile_count; ++i) {
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
      std::this_thread::sleep_for(dura);
  }
    std::cout << "Leave Xmalloc syscall service thread." << std::endl;
}

#undef DEBUG

#endif
