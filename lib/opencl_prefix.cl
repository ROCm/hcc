#ifdef cl_khr_fp64
  #pragma OPENCL EXTENSION cl_khr_fp64: enable
#elif defined(cl_amd_fp64)
  #pragma OPENCL EXTENSION cl_amd_fp64: enable
#else
  #error Double type is NOT supported by the OpenCL stack
#endif

