#pragma once

#include <cstdlib>
#include <cstdio>
#define UNW_LOCAL_ONLY
#include <libunwind.h>
#include <cxxabi.h>
#include <string>
#include <sstream>
#include <iostream>


#define DB_API        0  /* 0x0001  HCC runtime API calls */
#define DB_CMD        1  /* 0x0002  Kernel and Copy Commands and Barriers */
#define DB_WAIT       2  /* 0x0004  Synchronization and waiting for commands to finish. */
#define DB_AQL        3  /* 0x0008  Decode and display AQL packets  */
#define DB_QUEUE      4  /* 0x0010  Queue creation and desruction commands, and queue contents after each command push. */
#define DB_SIG        5  /* 0x0020  Signal creation, allocation, pool */
#define DB_LOCK       6  /* 0x0040  Locks and HCC thread-safety code */
#define DB_KERNARG    7  /* 0x0080  Show first 128 bytes of kernarg blocks passed to kernels */
#define DB_COPY       8  /* 0x0100  Copy debug */
#define DB_COPY2      9  /* 0x0200  Detailed copy debug */
#define DB_RESOURCE  10  /* 0x0400  Resource (signal/kernarg/queue) allocation and growth, and other unusual potentially performance-impacting events. */
#define DB_INIT      11  /* 0x0800  HCC initialization and shutdown. */
#define DB_MISC      12  /* 0x1000  misc debug, not yet classified. */
#define DB_AQL2      13  /* 0x2000  Show raw bytes of AQL packet */
#define DB_CODE      14  /* 0x4000  Show CreateKernel and code creation debug */
// If adding new define here update the table below:

extern unsigned HCC_DB;


// Keep close to debug defs above since these have to be kept in-sync
static std::vector<std::string> g_DbStr = {"api", "cmd", "wait", "aql", "queue", "sig", "lock", "kernarg", "copy", "copy2", "resource", "init", "misc", "aql2", "code"};


// Macro for prettier debug messages, use like:
// DBOUT(" Something happened" << myId() << " i= " << i << "\n");
#define COMPILE_HCC_DB 1

#define DBFLAG(db_flag) (HCC_DB & (1<<(db_flag)))

#define DBSTREAM std::cerr

// Use str::stream so output is atomic wrt other threads:
#define DBOUT(db_flag, msg) \
if (COMPILE_HCC_DB && (HCC_DB & (1<<(db_flag)))) { \
    std::stringstream sstream;\
    sstream << "   hcc-" << g_DbStr[db_flag] << " tid:" << hcc_tlsShortTid._shortTid << " " << msg ; \
    DBSTREAM << sstream.str();\
};

// Like DBOUT, but add newline:
#define DBOUTL(db_flag, msg) \
if (COMPILE_HCC_DB && (HCC_DB & (1<<(db_flag)))) { \
    std::stringstream sstream;\
    sstream << "   hcc-" << g_DbStr[db_flag] << " tid:" << hcc_tlsShortTid._shortTid << " " << msg << "\n"; \
    DBSTREAM << sstream.str();\
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
