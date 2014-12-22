//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// FIXME this file will place C++AMP Runtime implementation (OpenCL version)

#include <iostream>
#include <vector>
#include <map>
#include <future>
#include <cassert>
#include <stdexcept>

#include <CL/opencl.h>

#include <amp_allocator.h>
#include <amp_runtime.h>

///
/// global values 
///
namespace {
bool __mcw_cxxamp_compiled = false;
std::vector<std::string> __mcw_kernel_names;
const wchar_t gpu_accelerator[] = L"gpu";
const wchar_t cpu_accelerator[] = L"cpu";
const wchar_t default_accelerator[] = L"default";
}

extern "C" void PushArgImpl(void *k_, int idx, size_t sz, const void *s);

///
/// memory allocator
///
namespace Concurrency {

// forward declaration
namespace CLAMP {
    void CLCompileKernels(cl_program&, cl_context&, cl_device_id&, void*, void*);
}

struct rw_info
{
    int count;
    bool used;
};
class OpenCLAMPAllocator : public AMPAllocator
{
public:
    OpenCLAMPAllocator() {
        cl_uint          num_platforms;
        cl_int           err;
        cl_platform_id   platform_id[10];
        int i;
        err = clGetPlatformIDs(10, platform_id, &num_platforms);
        for (i = 0; i < num_platforms; i++) {
            err = clGetDeviceIDs(platform_id[i], CL_DEVICE_TYPE_GPU, 1, &device, NULL);
            if (err == CL_SUCCESS)
                break;
        }
        if (err != CL_SUCCESS) {
            for (i = 0; i < num_platforms; i++) {
                err = clGetDeviceIDs(platform_id[i], CL_DEVICE_TYPE_CPU, 1, &device, NULL);
                if (err == CL_SUCCESS)
                    break;
            }
        }
        assert(err == CL_SUCCESS);
        context = clCreateContext(0, 1, &device, NULL, NULL, &err);
        assert(err == CL_SUCCESS);
        queue = clCreateCommandQueue(context, device, 0, &err);
        assert(err == CL_SUCCESS);
    }
    void init(void *data, int count) {
        if (count > 0) {
            cl_int err;
            cl_mem dm = clCreateBuffer(context, CL_MEM_READ_WRITE, count, NULL, &err);
            rwq[data] = {count, false};
            assert(err == CL_SUCCESS);
            mem_info[data] = dm;
        }
    }
    void append(void *kernel, int idx, void *data) {
        PushArgImpl(kernel, idx, sizeof(cl_mem), &mem_info[data]);
        rwq[data].used = true;
    }
    void write() {
        cl_int err;
        for (auto& it : rwq) {
            rw_info& rw = it.second;
            if (rw.used) {
                err = clEnqueueWriteBuffer(queue, mem_info[it.first], CL_TRUE, 0,
                                           rw.count, it.first, 0, NULL, NULL);
                assert(err == CL_SUCCESS);
            }
        }
    }
    void read() {
        cl_int err;
        for (auto& it : rwq) {
            rw_info& rw = it.second;
            if (rw.used) {
                err = clEnqueueReadBuffer(queue, mem_info[it.first], CL_TRUE, 0,
                                          rw.count, it.first, 0, NULL, NULL);
                assert(err == CL_SUCCESS);
                rw.used = false;
            }
        }
    }
    void free(void *data) {
        auto iter = mem_info.find(data);
        clReleaseMemObject(iter->second);
        mem_info.erase(iter);
    }
    ~OpenCLAMPAllocator() {
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
    }

    std::map<void *, cl_mem> mem_info;
    cl_context       context;
    cl_device_id     device;
    cl_kernel        kernel;
    cl_command_queue queue;
    cl_program       program;
    std::map<void *, rw_info> rwq;
};

static OpenCLAMPAllocator amp;

OpenCLAMPAllocator& getOpenCLAMPAllocator() {
    return amp;
}


} // namespace Concurrency

///
/// kernel compilation / kernel launching
///

namespace Concurrency {
namespace CLAMP {

static inline void getKernelNames(cl_program& prog) {
    std::vector<std::string> n;
    cl_uint kernel_num = 0;
    cl_uint ret = CL_SUCCESS;
    char **names;
    int count = 0;
    ret = clCreateKernelsInProgram(prog, 1024, NULL, &kernel_num);
    if (ret == CL_SUCCESS && kernel_num > 0) {
        cl_kernel *kl = new cl_kernel[kernel_num];
        ret = clCreateKernelsInProgram(prog, kernel_num + 1, kl, &kernel_num);
        if (ret == CL_SUCCESS) {
            std::map<std::string, std::string> aMap;
            for (unsigned i = 0; i < unsigned(kernel_num); ++i) {
                char s[1024] = { 0x0 };
                size_t size;
                ret = clGetKernelInfo(kl[i], CL_KERNEL_FUNCTION_NAME, 1024, s, &size);
                n.push_back(std::string (s));
                clReleaseKernel(kl[i]);
            }
        }
        delete [] kl;
    }
    if (n.size()) {
        std::sort(n.begin(), n.end());
        n.erase(std::unique(n.begin(), n.end()), n.end());
    }
    if (n.size()) {
        names = new char *[n.size()];
        int i = 0;
        std::vector<std::string>::iterator it;
        for (it = n.begin(); it != n.end(); ++it, ++i) {
            size_t len = (*it).length();
            char *name = new char[len + 1];
            memcpy(name, (*it).c_str(), len);
            name[len] = '\0';
            names[i] = name;
        }
        count = unsigned(n.size());
    }
    if (count) {
        int i = 0;
        while (names && i < count) {
            __mcw_kernel_names.push_back(std::string(names[i]));
            delete [] names[i];
            ++i;
        }
        delete [] names;
        if (__mcw_kernel_names.size()) {
            std::sort(std::begin(__mcw_kernel_names), std::end(__mcw_kernel_names));
            __mcw_kernel_names.erase (std::unique (__mcw_kernel_names.begin (),
                                                   __mcw_kernel_names.end ()),
                                      __mcw_kernel_names.end ());
        }
    }
}

void CLCompileKernels(cl_program& program, cl_context& context, cl_device_id& device, void* kernel_size_, void* kernel_source_)
{
    cl_int err;
    if (!__mcw_cxxamp_compiled) {
        size_t kernel_size = (size_t)((void *)kernel_size_);
        unsigned char *kernel_source = (unsigned char*)malloc(kernel_size+1);
        memcpy(kernel_source, kernel_source_, kernel_size);
        kernel_source[kernel_size] = '\0';
        if (kernel_source[0] == 'B' && kernel_source[1] == 'C') {
            // Bitcode magic number. Assuming it's in SPIR
            const unsigned char *ks = (const unsigned char *)kernel_source;
            program = clCreateProgramWithBinary(context, 1, &device, &kernel_size, &ks, NULL, &err);
            if (err == CL_SUCCESS)
                err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
        } else {
            // in OpenCL-C
            const char *ks = (const char *)kernel_source;
            program = clCreateProgramWithSource(context, 1, &ks, &kernel_size, &err);
            if (err == CL_SUCCESS)
                err = clBuildProgram(program, 1, &device, "-D__ATTRIBUTE_WEAK__=", NULL, NULL);
        }
        if (err != CL_SUCCESS) {
            size_t len;
            err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
            assert(err == CL_SUCCESS);
            char *msg = new char[len + 1];
            err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, len, msg, NULL);
            assert(err == CL_SUCCESS);
            msg[len] = '\0';
            std::cerr << msg;
            delete [] msg;
            exit(1);
        }
        __mcw_cxxamp_compiled = true;
        free(kernel_source);
        getKernelNames(program);
    }
}

} // namespce CLAMP
} // namespace Concurrency


extern "C" void *GetAllocatorImpl() {
    return &Concurrency::amp;
}

extern "C" void EnumerateDevicesImpl(int* devices, int* device_number) {
    int deviceTotalCount = 0;
    int idx = 0;
    cl_int err;
    cl_uint platformCount;
    cl_uint deviceCount;
    std::unique_ptr<cl_platform_id[]> platforms;

    err = clGetPlatformIDs(0, nullptr, &platformCount);
    platforms.reset(new cl_platform_id[platformCount]);
    clGetPlatformIDs(platformCount, platforms.get(), nullptr);
    for (int i = 0; i < platformCount; i++) {
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, 0, nullptr, &deviceCount);
        for (int j = 0; j < deviceCount; j++) {
            if (devices != nullptr) {
              devices[idx++] = AMP_DEVICE_TYPE_CPU;
            }
        }
        deviceTotalCount += deviceCount;

        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, nullptr, &deviceCount);
        for (int j = 0; j < deviceCount; j++) {
            if (devices != nullptr) {
              devices[idx++] = AMP_DEVICE_TYPE_GPU;
            }
        }
        deviceTotalCount += deviceCount;
    }
    assert(idx == deviceTotalCount);

    if (device_number != nullptr) {
      *device_number = deviceTotalCount;
    }
}

extern "C" void QueryDeviceInfoImpl(const wchar_t* device_path,
    bool* supports_cpu_shared_memory,
    size_t* dedicated_memory,
    bool* supports_limited_double_precision,
    wchar_t* description) {

    const wchar_t des[] = L"OpenCL";
    wmemcpy(description, des, sizeof(des));

    cl_int err;
    cl_uint platformCount;
    cl_device_id device;
    cl_ulong memAllocSize;
    cl_device_fp_config singleFPConfig;
    std::unique_ptr<cl_platform_id[]> platforms;

    err = clGetPlatformIDs(0, NULL, &platformCount);
    assert(err == CL_SUCCESS);
    platforms.reset(new cl_platform_id[platformCount]);
    clGetPlatformIDs(platformCount, platforms.get(), NULL);
    assert(err == CL_SUCCESS);
    int i;
    for (i = 0; i < platformCount; i++) {
        if (std::wstring(gpu_accelerator) == device_path) {
            
            err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 1, &device, NULL);
            if (err != CL_SUCCESS)
                continue;
            *supports_cpu_shared_memory = false;
            break;
        } else if (std::wstring(cpu_accelerator) == device_path) {
            err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, 1, &device, NULL);
            if (err != CL_SUCCESS)
                continue;
            *supports_cpu_shared_memory = true;
            break;
        }
    }
    if (i == platformCount)
        return;

    err = clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &memAllocSize, NULL);
    assert(err == CL_SUCCESS);
    *dedicated_memory = memAllocSize / (size_t) 1024;

    err = clGetDeviceInfo(device, CL_DEVICE_SINGLE_FP_CONFIG, sizeof(cl_device_fp_config), &singleFPConfig, NULL);
    assert(err == CL_SUCCESS);
    if (singleFPConfig & CL_FP_FMA & CL_FP_DENORM & CL_FP_INF_NAN &
        CL_FP_ROUND_TO_NEAREST & CL_FP_ROUND_TO_ZERO)
         *supports_limited_double_precision = true;
}

extern "C" void *CreateKernelImpl(const char* s, void* kernel_size, void* kernel_source) {
  cl_int err;
  Concurrency::OpenCLAMPAllocator& aloc = Concurrency::getOpenCLAMPAllocator();
  Concurrency::CLAMP::CLCompileKernels(aloc.program, aloc.context, aloc.device, kernel_size, kernel_source);
  aloc.kernel = clCreateKernel(aloc.program, s, &err);
  assert(err == CL_SUCCESS);
  return aloc.kernel;
}

extern "C" void LaunchKernelImpl(void *kernel, size_t dim_ext, size_t *ext, size_t *local_size) {
  cl_int err;
  Concurrency::OpenCLAMPAllocator& aloc = Concurrency::getOpenCLAMPAllocator();
  {
      // C++ AMP specifications
      // The maximum number of tiles per dimension will be no less than 65535.
      // The maximum number of threads in a tile will be no less than 1024.
      // In 3D tiling, the maximal value of D0 will be no less than 64.
      cl_uint dimensions;
      err = clGetDeviceInfo(aloc.device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &dimensions, NULL);
      size_t *maxSizes = new size_t[dimensions];
      err = clGetDeviceInfo(aloc.device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * dimensions, maxSizes, NULL);
      bool is = true;
      int threads_per_tile = 1;
      for(int i = 0; local_size && i < dim_ext; i++) {
          threads_per_tile *= local_size[i];
          // For the following cases, set local_size=NULL and let OpenCL driver arranges it instead
          //(1) tils number exceeds CL_DEVICE_MAX_WORK_ITEM_SIZES per dimension
          //(2) threads in a tile exceeds CL_DEVICE_MAX_WORK_ITEM_SIZES
          //Note that the driver can still handle unregular tile_dim, e.g. tile_dim is undivisble by 2
          //So skip this condition ((local_size[i]!=1) && (local_size[i] & 1))
          if(local_size[i] > maxSizes[i] || threads_per_tile > maxSizes[i]) {
              is = false;
              break;
          }
      }
      if(!is)
          local_size = NULL;
  }

  aloc.write();
  err = clEnqueueNDRangeKernel(aloc.queue, aloc.kernel, dim_ext, NULL, ext, local_size, 0, NULL, NULL);
  assert(err == CL_SUCCESS);
  aloc.read();
  clFinish(aloc.queue);
}

extern "C" void *LaunchKernelAsyncImpl(void *ker, size_t nr_dim, size_t *global, size_t *local) {
  throw std::runtime_error("async_parallel_for_each is unsupported on this platform");
}

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
extern "C" void MatchKernelNamesImpl(char *fixed_name) {
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
      memcpy(fixed_name, shortest.c_str(), shortest.length());
  }
}

extern "C" void PushArgImpl(void *k_, int idx, size_t sz, const void *s) {
  cl_int err;
  err = clSetKernelArg(static_cast<cl_kernel>(k_), idx, sz, s);
  assert(err == CL_SUCCESS);
}

