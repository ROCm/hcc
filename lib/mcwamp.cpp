#include <amp.h>
#include <map>
namespace Concurrency {
accelerator_view *accelerator::default_view_ = NULL;
const wchar_t accelerator::gpu_accelerator[] = L"gpu";
wchar_t accelerator::default_accelerator[] = L"default";
const wchar_t accelerator::cpu_accelerator[] = L"cpu";
}

namespace {
bool __mcw_cxxamp_compiled = false;
}

#ifdef __APPLE__
#include <mach-o/getsect.h>
extern "C" intptr_t _dyld_get_image_vmaddr_slide(uint32_t image_index);
#else
extern "C" char * kernel_source_[] asm ("_binary_kernel_cl_start") __attribute__((weak));
extern "C" char * kernel_size_[] asm ("_binary_kernel_cl_size") __attribute__((weak));
#endif

namespace Concurrency {
namespace CLAMP {
void CompileKernels(void)
{
#ifdef CXXAMP_ENABLE_HSA_OKRA
  assert(0 && "Unsupported function");
#else
  ecl_error error_code;
  if ( !__mcw_cxxamp_compiled ) {
#ifdef __APPLE__
    const struct section_64 *sect = getsectbyname("binary", "kernel_cl");
    unsigned char *kernel_source = (unsigned char*)calloc(1, sect->size+1);
    size_t kernel_size = sect->size;
    assert(sect->addr != 0);
    memcpy(kernel_source, (void*)(sect->addr + _dyld_get_image_vmaddr_slide(0)), kernel_size); // whatever
#else
    size_t kernel_size = (size_t)((void *)kernel_size_);
    unsigned char *kernel_source = (unsigned char*)malloc(kernel_size+1);
    memcpy(kernel_source, kernel_source_, kernel_size);
#endif
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
#endif
}

#ifdef CXXAMP_ENABLE_HSA_OKRA
} // namespce CLAMP
} // namespace Concurrency
#include <okraContext.h>
namespace Concurrency {
namespace CLAMP {
/* Used only in HSA Okra runtime */
OkraContext *GetOrInitOkraContext(void)
{
  static OkraContext *context = NULL;
  if (!context) {
    //std::cerr << "Okra: create context\n";
    context = OkraContext::Create();
  }
  if (!context) {
    std::cerr << "Okra: Unable to create context\n";
    abort();
  }
  return context;
}

static std::map<std::string, OkraContext::Kernel *> __mcw_okra_kernels;
void *CreateOkraKernel(std::string s)
{
  OkraContext::Kernel *kernel = __mcw_okra_kernels[s];
  if (!kernel) {
      size_t kernel_size = (size_t)((void *)kernel_size_);
      char *kernel_source = (char*)malloc(kernel_size+1);
      memcpy(kernel_source, kernel_source_, kernel_size);
      kernel_source[kernel_size] = '\0';
      std::string kname = std::string("&__OpenCL_")+s+
          std::string("_kernel");
      kernel = GetOrInitOkraContext()->
          createKernel(kernel_source, kname.c_str());
      //std::cerr << "CLAMP::Okra::Creating kernel: "<< kname<<"\n";
      //std::cerr << "CLAMP::Okra::Creating kernel: "<< kernel <<"\n";
      if (!kernel) {
          std::cerr << "Okra: Unable to create kernel\n";
          abort();
      }
      __mcw_okra_kernels[s] = kernel;
  }
  kernel->clearArgs();
  // HSA kernels generated from OpenCL takes 3 additional arguments at the beginning
  kernel->pushLongArg(0);
  kernel->pushLongArg(0);
  kernel->pushLongArg(0);
  return kernel;
}
namespace Okra {
void RegisterMemory(void *p, size_t sz)
{
    //std::cerr << "registering: ptr " << p << " of size " << sz << "\n";
    GetOrInitOkraContext()->registerArrayMemory(p, sz);
}
}

void OkraLaunchKernel(void *ker, size_t nr_dim, size_t *global, size_t *local)
{
  OkraContext::Kernel *kernel =
      reinterpret_cast<OkraContext::Kernel*>(ker);
  size_t tmp_local[] = {0, 0, 0};
  if (!local)
      local = tmp_local;
  //std::cerr<<"Launching: nr dim = " << nr_dim << "\n";

  kernel->setLaunchAttributes(nr_dim, global, local);
  //std::cerr << "No real launch\n";
  kernel->dispatchKernelWaitComplete();
}

void OkraPushArg(void *ker, size_t sz, const void *v)
{
  //std::cerr << "pushing:" << ker << " of size " << sz << "\n";
  OkraContext::Kernel *kernel =
      reinterpret_cast<OkraContext::Kernel*>(ker);
  void *val = const_cast<void*>(v);
  switch (sz) {
    case sizeof(int):
      kernel->pushIntArg(*reinterpret_cast<int*>(val));
      //std::cerr << "(int) value = " << *reinterpret_cast<int*>(val) <<"\n";
      break;
    default:
      assert(0 && "Unsupported kernel argument size");
  }
}
void OkraPushPointer(void *ker, void *val)
{
    //std::cerr << "pushing:" << ker << " of ptr " << val << "\n";
    OkraContext::Kernel *kernel =
        reinterpret_cast<OkraContext::Kernel*>(ker);
    kernel->pushPointerArg(val);
}
#endif
} // namespace CLAMP
} // namespace Concurrency
