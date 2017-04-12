#pragma once

#include <cstdlib>
#include <cstdio>
#define UNW_LOCAL_ONLY
#include <libunwind.h>
#include <cxxabi.h>

namespace hc {
  static void print_backtrace() {
    unw_cursor_t cursor;
    unw_context_t context;
    unw_getcontext(&context);
    unw_init_local(&cursor, &context);

    std::fprintf(stderr, "\nBacktrace:\n");

    while(unw_step(&cursor) > 0) {
      // get the program counter
      unw_word_t pc;
      unw_get_reg(&cursor, UNW_REG_IP, &pc);
      if (pc == 0x00)
        break;

      // get the function name
      char func[512];
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
      std::fprintf(stderr, "0x%016lx:\t%s + 0x%lx\n", pc, print_func_name, offp);
      if (demangled)
        free(demangled);
    }
    std::fprintf(stderr, "\n");
  }
} // namespace hc
