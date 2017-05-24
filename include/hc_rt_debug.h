#pragma once

#include <cstdlib>
#include <cstdio>
#define UNW_LOCAL_ONLY
#include <libunwind.h>
#include <cxxabi.h>
#include <string>
#include <sstream>
#include <iostream>


#define DB_MISC      0x0  // 0x001  // misc debug, not yet classified.
#define DB_CMD       0x1  // 0x002  // Kernel and COpy Commands and synchronization
#define DB_WAIT      0x2  // 0x004  // Synchronization and waiting for commands to finish.
#define DB_AQL       0x3  // 0x008  // Decode and display AQL packets 
#define DB_QUEUE     0x4  // 0x010  // Queue creation and desruction commands
#define DB_SIG       0x5  // 0x020  // Signal creation, allocation, pool
#define DB_LOCK      0x6  // 0x040  // Locks and HCC thread-safety code
#define DB_KERNARG   0x7  // 0x080  // Decode and display AQL packets 
#define DB_COPY2     0x8  // 0x100  // Detailed copy debug
// If adding new define here update the table below:

extern unsigned HCC_DB;


// Keep close to debug defs above since these have to be kept in-sync
static std::vector<std::string> g_DbStr = {"misc", "cmd", "wait", "aql", "queue", "sig", "lock", "kernarg", "copy2" };


// Macro for prettier debug messages, use like:
// DBOUT(" Something happened" << myId() << " i= " << i << "\n");
#define COMPILE_HCC_DB 1

#define DBFLAG(db_flag) (HCC_DB & (1<<db_flag))

// Use str::stream so output is atomic wrt other threads:
#define DBOUT(db_flag, msg) \
if (COMPILE_HCC_DB && (HCC_DB & (1<<(db_flag)))) { \
    std::stringstream sstream;\
    sstream << "   hcc-" << g_DbStr[db_flag] << " tid:" << hcc_tlsShortTid._shortTid << " " << msg ; \
    std::cerr << sstream.str();\
};

// Like DBOUT, but add newline:
#define DBOUTL(db_flag, msg) \
if (COMPILE_HCC_DB && (HCC_DB & (1<<(db_flag)))) { \
    std::stringstream sstream;\
    sstream << "   hcc-" << g_DbStr[db_flag] << " tid:" << hcc_tlsShortTid._shortTid << " " << msg << "\n"; \
    std::cerr << sstream.str();\
};


// Class with a constructor that gets called when new thread is created:
struct ShortTid {
    ShortTid() ;
    int _shortTid;
};

extern thread_local ShortTid hcc_tlsShortTid;

namespace hc {


  static std::string get_backtrace() {
    constexpr int buffer_size = 512;

    std::string bt("");

    unw_cursor_t cursor;
    unw_context_t context;
    unw_getcontext(&context);
    unw_init_local(&cursor, &context);

    bt += std::string("Backtrace:\n");

    while(unw_step(&cursor) > 0) {
      // get the program counter
      unw_word_t pc;
      unw_get_reg(&cursor, UNW_REG_IP, &pc);
      if (pc == 0x00)
        break;

      // get the function name
      char func[buffer_size];
      char* demangled = nullptr;
      const char* print_func_name;
      unw_word_t offp;
      if (unw_get_proc_name(&cursor, func, sizeof(func), &offp) == 0) {
        int status;
        demangled = abi::__cxa_demangle(func, nullptr, nullptr, &status);
        print_func_name = demangled ? demangled : func;
      }
      else {
        print_func_name = "<unknown function>";
      }

      char loc[buffer_size];
      std::snprintf(loc, buffer_size, "0x%016lx:\t%s + 0x%lx\n", pc, print_func_name, offp);
      bt += std::string(loc);

      if (demangled)
        free(demangled);
    }
    return bt;
  }

  static void print_backtrace() {
    std::string bt = get_backtrace();
    std::printf("\n%s\n", bt.c_str());
  }

} // namespace hc
