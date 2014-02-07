#include <amp.h>
namespace Concurrency {
accelerator_view *accelerator::default_view_ = NULL;
const wchar_t accelerator::gpu_accelerator[] = L"gpu";
wchar_t accelerator::default_accelerator[] = L"default";
const wchar_t accelerator::cpu_accelerator[] = L"cpu";
}

namespace {
bool __mcw_cxxamp_compiled = false;
}
extern "C" char * kernel_source_[] asm ("_binary_kernel_cl_start") __attribute__((weak));
extern "C" char * kernel_size_[] asm ("_binary_kernel_cl_size") __attribute__((weak));

namespace Concurrency {
namespace CLAMP {
void CompileKernels(void)
{
  ecl_error error_code;
  if ( !__mcw_cxxamp_compiled ) {
    size_t kernel_size = (size_t)((void *)kernel_size_);
    unsigned char *kernel_source = (unsigned char*)malloc(kernel_size+1);
    memcpy(kernel_source, kernel_source_, kernel_size);
    kernel_source[kernel_size] = '\0';
    if (kernel_source[0] == 'B' && kernel_source[1] == 'C') {
      // Bitcode magic number. Assuming it's in SPIR
      error_code = eclCompileBinary(kernel_source, kernel_size);
      CHECK_ERROR_GMAC(error_code, "Compiling kernel in SPIR binary");
    } else {
      // in OpenCL-C
      const char *ks = (const char *)kernel_source;
      error_code = eclCompileSource(ks, "-D__ATTRIBUTE_WEAK__=");
      CHECK_ERROR_GMAC(error_code, "Compiling kernel in OpenCL-C");
    }
    __mcw_cxxamp_compiled = true;
    free(kernel_source);
  }
}

} // namespace CLAMP
} // namespace Concurrency
