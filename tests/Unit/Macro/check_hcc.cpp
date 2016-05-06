// RUN: %cxxamp %s -o %t.out && %t.out

#include <hc_defines.h>
#include <iostream>

#ifndef __HCC__
#error __HCC__ is not defined!
#endif

#ifndef __hcc_major__
#error __hcc_major__ is not defined!
#endif

#ifndef __hcc_minor__
#error __hcc_minor__ is not defined!
#endif

#ifndef __hcc_patchlevel__
#error __hcc_patchlevel__ is not defined!
#endif

#ifndef __hcc_version__
#error __hcc_version__ is not defined!
#endif

#ifndef __hcc_workweek__
#error __hcc_workweek__ is not defined!
#endif

#ifndef __hcc_backend__
#error __hcc_backend__ is not defined!
#endif

int main() {
  std::cout << __hcc_major__ << "\n"
            << __hcc_minor__ << "\n"
            << __hcc_patchlevel__ << "\n"
            << __hcc_version__ << "\n"
            << __hcc_workweek__ << "\n"
            << __hcc_backend__ << "\n";

  return 0;
}

