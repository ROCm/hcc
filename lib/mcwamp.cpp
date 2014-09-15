//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <amp.h>
#include <map>
namespace Concurrency {

// initialize static class members
const wchar_t accelerator::gpu_accelerator[] = L"gpu";
const wchar_t accelerator::cpu_accelerator[] = L"cpu";
const wchar_t accelerator::default_accelerator[] = L"default";

std::shared_ptr<accelerator> accelerator::_gpu_accelerator = std::make_shared<accelerator>(accelerator::gpu_accelerator);
std::shared_ptr<accelerator> accelerator::_cpu_accelerator = std::make_shared<accelerator>(accelerator::cpu_accelerator);
std::shared_ptr<accelerator> accelerator::_default_accelerator = nullptr;

} // namespace Concurrency

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

std::vector<std::string> __mcw_kernel_names;
namespace Concurrency {
namespace CLAMP {
// Levenshtein Distance to measure the difference of two sequences
// The shortest distance it returns the more likely the two sequences are equal
static inline int ldistance(const std::string source, const std::string target)
{
  int n = source.length();
  int m = target.length();
  if (m == 0)
    return n;
  if (n == 0)
    return m;
  
  //Construct a matrix
  typedef std::vector < std::vector < int >>Tmatrix;
  Tmatrix matrix(n + 1);
  
  for (int i = 0; i <= n; i++)
    matrix[i].resize(m + 1);
  for (int i = 1; i <= n; i++)
    matrix[i][0] = i;
  for (int i = 1; i <= m; i++)
    matrix[0][i] = i;

  for (int i = 1; i <= n; i++) {
    const char si = source[i - 1];
    for (int j = 1; j <= m; j++) {
      const char dj = target[j - 1];
      int cost;
      if (si == dj)
        cost = 0;
      else
        cost = 1;
      const int above = matrix[i - 1][j] + 1;
      const int left = matrix[i][j - 1] + 1;
      const int diag = matrix[i - 1][j - 1] + cost;
      matrix[i][j] = std::min(above, std::min(left, diag));
    }
  }
  return matrix[n][m];
}
// transformed_kernel_name (mangled) might differ if usages of 'm32' flag in CPU/GPU
// paths are mutually exclusive. We can scan all kernel names and replace
// transformed_kernel_name with the one that has the shortest distance from it by using 
// Levenshtein Distance measurement
void MatchKernelNames(std::string& fixed_name) {
  if (__mcw_kernel_names.size()) {
    // Must start from a big value > 10
    int distance = 1024;
    int hit = -1;
    std::string shortest;
    for (std::vector < std::string >::iterator it = __mcw_kernel_names.begin();
         it != __mcw_kernel_names.end(); ++it) {
      if ((*it) == fixed_name) {
        // Perfect match. Mark no need to replace and skip the loop
        hit = -1;
        break;
      }
      int n = ldistance(fixed_name, (*it));
      if (n <= distance) {
        distance = n;
        hit = 1;
        shortest = (*it);
      }
    }
    /* Replacement. Skip if not hit or the distance is too far (>5)*/
    if (hit >= 0 && distance < 5)
      fixed_name = shortest;
  }
  return;
}
}
}
namespace Concurrency {
namespace CLAMP {
void CompileKernels(void)
{
#if defined(CXXAMP_ENABLE_HSA)
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
    // Extract kernel names
    char** kernel_names = NULL;
    unsigned kernel_num = 0;
    ecl_error error_code;
    error_code =  eclGetKernelNames(&kernel_names, &kernel_num);
    if(error_code == eclSuccess && kernel_names) {
       int i = 0;
       while(kernel_names && i<kernel_num) {
          __mcw_kernel_names.push_back(std::string(kernel_names[i]));
          delete [] kernel_names[i];
          ++i;
        }
       delete [] kernel_names;
       if(__mcw_kernel_names.size()) {
         std::sort(std::begin(__mcw_kernel_names), std::end(__mcw_kernel_names));
         __mcw_kernel_names.erase (std::unique (__mcw_kernel_names.begin (),
                                                       __mcw_kernel_names.end ()), __mcw_kernel_names.end ());
       }
    }
  }
#endif
}

#if defined(CXXAMP_ENABLE_HSA)
} // namespce CLAMP
} // namespace Concurrency
#include "HSAContext.h"
namespace Concurrency {
namespace CLAMP {

static HSAContext *context = NULL;

void FinalizeHSAContext() {
  if (context != NULL) {
    context->dispose();
    context = NULL;
  }

  // TBD dispose all Kernel objects

  // TBD dispose all Dispatch objects
}

/* Used only in HSA runtime */
HSAContext *GetOrInitHSAContext(void)
{
  if (!context) {
    //std::cerr << "CLAMP::HSA: create context\n";
    context = HSAContext::Create();
    atexit(FinalizeHSAContext); // register finalizer
  }
  if (!context) {
    std::cerr << "CLAMP::HSA: Unable to create context\n";
    abort();
  }
  return context;
}

static std::map<std::string, HSAContext::Kernel *> __mcw_hsa_kernels;
void *CreateHSAKernel(std::string s)
{
  HSAContext::Kernel *kernel = __mcw_hsa_kernels[s];
  if (!kernel) {
      size_t kernel_size = (size_t)((void *)kernel_size_);
      char *kernel_source = (char*)malloc(kernel_size+1);
      memcpy(kernel_source, kernel_source_, kernel_size);
      kernel_source[kernel_size] = '\0';
      std::string kname = std::string("&")+s;
      //std::cerr << "CLAMP::HSA::Creating kernel: " << kname << "\n";
      kernel = GetOrInitHSAContext()->
          createKernel(kernel_source, kernel_size, kname.c_str());
      if (!kernel) {
          std::cerr << "CLAMP::HSA: Unable to create kernel\n";
          abort();
      } else {
          //std::cerr << "CLAMP::HSA: Created kernel\n";
      }
      __mcw_hsa_kernels[s] = kernel;
  }

  HSAContext::Dispatch *dispatch = GetOrInitHSAContext()->createDispatch(kernel);
  dispatch->clearArgs();
//#define CXXAMP_ENABLE_HSAIL_HLC_DEVELOPMENT_COMPILER 1
#ifndef CXXAMP_ENABLE_HSAIL_HLC_DEVELOPMENT_COMPILER
  dispatch->pushLongArg(0);
  dispatch->pushLongArg(0);
  dispatch->pushLongArg(0);
  dispatch->pushLongArg(0);
  dispatch->pushLongArg(0);
  dispatch->pushLongArg(0);
#endif
  return dispatch;
}

namespace HSA {
void RegisterMemory(void *p, size_t sz)
{
    //std::cerr << "registering: ptr " << p << " of size " << sz << "\n";
    GetOrInitHSAContext()->registerArrayMemory(p, sz);
}
}

std::future<void> HSALaunchKernelAsync(void *ker, size_t nr_dim, size_t *global, size_t *local)
{
  HSAContext::Dispatch *dispatch =
      reinterpret_cast<HSAContext::Dispatch*>(ker);
  size_t tmp_local[] = {0, 0, 0};
  if (!local)
      local = tmp_local;
  //std::cerr<<"Launching: nr dim = " << nr_dim << "\n";
  //for (size_t i = 0; i < nr_dim; ++i) {
  //  std::cerr << "g: " << global[i] << " l: " << local[i] << "\n";
  //}
  dispatch->setLaunchAttributes(nr_dim, global, local);
  //std::cerr << "Now real launch\n";
  //kernel->dispatchKernelWaitComplete();

  return dispatch->dispatchKernelAndGetFuture();
}

void HSALaunchKernel(void *ker, size_t nr_dim, size_t *global, size_t *local)
{
  HSAContext::Dispatch *dispatch =
      reinterpret_cast<HSAContext::Dispatch*>(ker);
  size_t tmp_local[] = {0, 0, 0};
  if (!local)
      local = tmp_local;
  //std::cerr<<"Launching: nr dim = " << nr_dim << "\n";
  //for (size_t i = 0; i < nr_dim; ++i) {
  //  std::cerr << "g: " << global[i] << " l: " << local[i] << "\n";
  //}
  dispatch->setLaunchAttributes(nr_dim, global, local);
  //std::cerr << "Now real launch\n";
  dispatch->dispatchKernelWaitComplete();
}

void HSAPushArg(void *ker, size_t sz, const void *v)
{
  //std::cerr << "pushing:" << ker << " of size " << sz << "\n";
  HSAContext::Dispatch *dispatch =
      reinterpret_cast<HSAContext::Dispatch*>(ker);
  void *val = const_cast<void*>(v);
  switch (sz) {
    case sizeof(double):
      dispatch->pushDoubleArg(*reinterpret_cast<double*>(val));
      break;
    case sizeof(int):
      dispatch->pushIntArg(*reinterpret_cast<int*>(val));
      //std::cerr << "(int) value = " << *reinterpret_cast<int*>(val) <<"\n";
      break;
    case sizeof(unsigned char):
      dispatch->pushBooleanArg(*reinterpret_cast<unsigned char*>(val));
      break;
    default:
      assert(0 && "Unsupported kernel argument size");
  }
}
void HSAPushPointer(void *ker, void *val)
{
    //std::cerr << "pushing:" << ker << " of ptr " << val << "\n";
    HSAContext::Dispatch *dispatch =
        reinterpret_cast<HSAContext::Dispatch*>(ker);
    dispatch->pushPointerArg(val);
}
#elif defined(CXXAMP_ENABLE_HSA)
} // namespce CLAMP
} // namespace Concurrency

#endif
} // namespace CLAMP
} // namespace Concurrency
