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

#include <md5.h>
#include <sstream>
#include <fstream>
#include <iomanip>

#include <amp_runtime.h>

///
/// global values 
///
namespace {
bool __mcw_cxxamp_compiled = false;
std::vector<std::string> __mcw_kernel_names;
const wchar_t cpu_accelerator[] = L"cpu";
const wchar_t default_accelerator[] = L"default";
}

extern "C" void PushArgImpl(void *k_, int idx, size_t sz, const void *s);
extern "C" void PushArgPtrImpl(void *k_, int idx, size_t sz, const void *s);

///
/// memory allocator
///
namespace Concurrency {

// forward declaration
namespace CLAMP {
    void CLCompileKernels(cl_program&, cl_context&, cl_device_id&, void*, void*);
}

struct DimMaxSize {
  cl_uint dimensions;
  size_t* maxSizes;
};
typedef std::map<std::string, cl_kernel> KernelObject;
std::map<cl_program, KernelObject> Pro2KernelObject;
void ReleaseKernelObject(cl_program& program) {
  for(const auto& itt : Pro2KernelObject[program])
    if(itt.second)
      clReleaseKernel(itt.second);
}

class OpenCLAllocator;

class OpenCLManager : public AMPManager
{
public:
    OpenCLManager(const cl_device_id device, const std::wstring& path)
        : AMPManager(path), device(device), program(nullptr) {
        cl_int           err;
        context = clCreateContext(0, 1, &device, NULL, NULL, &err);
        assert(err == CL_SUCCESS);

        cl_ulong memAllocSize;
        err = clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &memAllocSize, NULL);
        assert(err == CL_SUCCESS);
        mem = memAllocSize >> 10;

        char vendor[256];
        err = clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(vendor), vendor, NULL);
        assert(err == CL_SUCCESS);
        std::string ven(vendor);
        if (ven.find("Advanced Micro Devices") != std::string::npos)
            des = L"AMD";
        else if (ven.find("NVIDIA") != std::string::npos)
            des = L"NVIDIA";
        else if (ven.find("Intel") != std::string::npos) {
            des = L"Intel";
            // Hack: Don't let conformance to choose this
            mem = 0;
        }


        cl_bool unified;
        err = clGetDeviceInfo(device, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(cl_bool), &unified, NULL);
        assert(err == CL_SUCCESS);
        cpu_shared_memory = unified;

        is_limited_double_ = false;
        is_double_ = false;
        cl_device_fp_config fpconf;
        err = clGetDeviceInfo(device, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(cl_device_fp_config), &fpconf, NULL);
        if (fpconf & (CL_FP_FMA | CL_FP_INF_NAN | CL_FP_DENORM)) {
            is_limited_double_ = true;
            if(fpconf & (CL_FP_ROUND_TO_ZERO | CL_FP_ROUND_TO_NEAREST | CL_FP_ROUND_TO_INF))
                is_double_ = true;
        }

        cl_uint dimensions = 0;
        err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &dimensions, NULL);
        assert(err == CL_SUCCESS);
        size_t *maxSizes = new size_t[dimensions];
        err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * dimensions, maxSizes, NULL);
        assert(err == CL_SUCCESS);
        d.dimensions = dimensions;
        d.maxSizes = maxSizes;
        emulated = false;
    }
    void* CreateKernel(const char* fun, void* size, void* source) override {
        cl_int err;
        Concurrency::CLAMP::CLCompileKernels(program, context, device, size, source);
        Concurrency::KernelObject& KO = Concurrency::Pro2KernelObject[program];
        std::string name(fun);
        if (KO[name] == 0) {
            cl_int err;
            KO[name] = clCreateKernel(program, name.c_str(), &err);
            assert(err == CL_SUCCESS);
        }
        return KO[name];
    }

    bool check(size_t* local_size, size_t dim_ext) override {
        // C++ AMP specifications
        // The maximum number of tiles per dimension will be no less than 65535.
        // The maximum number of threads in a tile will be no less than 1024.
        // In 3D tiling, the maximal value of D0 will be no less than 64.
        size_t *maxSizes = d.maxSizes;
        bool is = true;
        int threads_per_tile = 1;
        for(int i = 0; local_size && i < dim_ext; i++) {
            threads_per_tile *= local_size[i];
            // For the following cases, set local_size=NULL and let OpenCL driver arranges it instead
            //(1) tils number exceeds CL_DEVICE_MAX_WORK_ITEM_SIZES per dimension
            //(2) threads in a tile exceeds CL_DEVICE_MAX_WORK_ITEM_SIZES
            //Note that the driver can still handle unregular tile_dim, e.g. tile_dim is undivisble by 2
            //So skip this condition ((local_size[i]!=1) && (local_size[i] & 1))
            if(local_size[i] > maxSizes[i] || threads_per_tile > maxSizes[i])
                return false;
        }
        return true;
    }

    ~OpenCLManager() {
        if (program) {
            ReleaseKernelObject(program);
            clReleaseProgram(program);
        }
        delete[] d.maxSizes;
        clReleaseContext(context);
    }

    cl_device_id getDevice() const { return device; }
    cl_context getContext() const { return context; }
private:

    void* create(size_t count, void* data, bool hasSource /* unused */) override {
        cl_int err;
        cl_mem dm = clCreateBuffer(context, CL_MEM_READ_WRITE, count, nullptr, &err);
        assert(err == CL_SUCCESS);
        return dm;
    }
    void release(void *device) override { clReleaseMemObject(static_cast<cl_mem>(device)); }
    std::shared_ptr<AMPAllocator> createAloc() override { return newAloc(); }


    std::shared_ptr<AMPAllocator> newAloc();
    struct DimMaxSize d;
    cl_context       context;
    cl_device_id     device;
    cl_program       program;
};

class OpenCLAllocator : public AMPAllocator
{
public:
    OpenCLAllocator(std::shared_ptr<AMPManager> pMan) : AMPAllocator(pMan) {
        auto Man = std::dynamic_pointer_cast<OpenCLManager, AMPManager>(pMan);
        cl_int err;
        queue = clCreateCommandQueue(Man->getContext(), Man->getDevice(), 0, &err);
        assert(err == CL_SUCCESS);

        // Propel underlying OpenCL driver to enque kernels faster (pthread-based)
        // FIMXE: workable on AMD platforms only
        pthread_t self = pthread_self();
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        int result = -1;
        // Get max priority
        int policy = 0;
        result = pthread_attr_getschedpolicy(&attr, &policy);
        if (result != 0)
          perror("getsched error!\n");
        int max_prio = sched_get_priority_max(policy);

        struct sched_param param;
        // Get self priority
        result = pthread_getschedparam(self, &policy, &param);
        if (result != 0)
          perror("getsched self error!\n");
        int self_prio = param.sched_priority;
#define CL_QUEUE_THREAD_HANDLE_AMD 0x403E
#define PRIORITY_OFFSET 2
        void* handle=NULL;
        cl_int status = clGetCommandQueueInfo (queue, CL_QUEUE_THREAD_HANDLE_AMD, sizeof(handle), &handle, NULL );
        // Ensure it is valid
        if (status == CL_SUCCESS && handle) {
            pthread_t thId = (pthread_t)handle;
            result = pthread_getschedparam(thId, &policy, &param);
            if (result != 0)
                perror("getsched q error!\n");
            int que_prio = param.sched_priority;
            // Strategy to renew the que thread's priority, the smaller the highest
            if (max_prio == que_prio) {
            } else if (max_prio < que_prio && que_prio <= self_prio) {
                // self    que    max
                que_prio = (que_prio-PRIORITY_OFFSET)>0?(que_prio-PRIORITY_OFFSET):que_prio;
            } else if (que_prio > self_prio) {
                // que   self    max
                que_prio = (self_prio-PRIORITY_OFFSET)>0?(self_prio-PRIORITY_OFFSET):self_prio;
            } 
            int result = pthread_setschedprio(thId, que_prio);
            if (result != 0)
                perror("Renew p error!\n");
        } 
        pthread_attr_destroy(&attr);
    }

    void flush() override { clFlush(queue); }
    void wait() override { clFinish(queue); }
    ~OpenCLAllocator() { clReleaseCommandQueue(queue); }

private:
    void Push(void *kernel, int idx, void*& data, obj_info& obj) override {
        PushArgImpl(kernel, idx, sizeof(cl_mem), &obj.device);
    }
    void amp_write(obj_info& obj, void *src) override {
        cl_mem dm = static_cast<cl_mem>(obj.device);
        cl_int err = clEnqueueWriteBuffer(queue, dm, CL_FALSE, 0, obj.count, src, 0, NULL, NULL);
        assert(err == CL_SUCCESS);
    }
    void amp_read(obj_info& obj, void* dst) override {
        cl_mem dm = static_cast<cl_mem>(obj.device);
        cl_int err = clEnqueueReadBuffer(queue, dm, CL_TRUE, 0, obj.count, dst, 0, NULL, NULL);
        assert(err == CL_SUCCESS);
    }
    void amp_copy(obj_info& obj, void* dst, void* src, size_t count) override {
        cl_mem dm = static_cast<cl_mem>(obj.device);
        cl_int err = clEnqueueReadBuffer(queue, dm, CL_TRUE, 0, count, dst, 0, NULL, NULL);
        assert(err == CL_SUCCESS);
    }
    void* amp_map(obj_info& obj, bool Write) override {
        cl_mem dm = static_cast<cl_mem>(obj.device);
        cl_int err;
        cl_map_flags flags;
        if (Write)
            flags = CL_MAP_WRITE_INVALIDATE_REGION;
        else
            flags = CL_MAP_READ;
        void* addr = clEnqueueMapBuffer(queue, dm, CL_TRUE, flags, 0, obj.count, 0, NULL, NULL, &err);
        assert(err == CL_SUCCESS);
        return addr;
    }
    void amp_unmap(obj_info& obj, void* addr) override {
        cl_mem dm = static_cast<cl_mem>(obj.device);
        cl_int err = clEnqueueUnmapMemObject(queue, dm, addr, 0, NULL, NULL);
        assert(err == CL_SUCCESS);
    }
    void LaunchKernel(void *kernel, size_t dim_ext, size_t *ext, size_t *local_size) override {
        cl_int err;
        if(!Man->check(local_size, dim_ext))
            local_size = NULL;
        err = clEnqueueNDRangeKernel(queue, (cl_kernel)kernel, dim_ext, NULL, ext, local_size, 0, NULL, NULL);
        assert(err == CL_SUCCESS);
    }
    cl_command_queue queue;
};

std::shared_ptr<AMPAllocator> OpenCLManager::newAloc() {
    return std::shared_ptr<AMPAllocator>(new OpenCLAllocator(shared_from_this()));
}

struct CLFlag
{
    const cl_device_type type;
    const std::wstring base;
    mutable int id;
    CLFlag(const cl_device_type& type, const std::wstring& base)
        : id(0), type(type), base(base) {}
    const std::wstring getPath() const { return base + std::to_wstring(id++); }
};

class OpenCLContext : public AMPContext
{
public:
    OpenCLContext() : AMPContext() {
        cl_uint num_platform;
        cl_int err;
        err = clGetPlatformIDs(0, nullptr, &num_platform);
        assert(err == CL_SUCCESS);
        std::vector<cl_platform_id> platform_id(num_platform);
        err = clGetPlatformIDs(num_platform, platform_id.data(), nullptr);
        std::vector<CLFlag> Flags({CLFlag(CL_DEVICE_TYPE_GPU, L"gpu"),
                                  CLFlag(CL_DEVICE_TYPE_ACCELERATOR, L"acc"),
                                  CLFlag(CL_DEVICE_TYPE_CPU, L"cpu")
                                  });

        for (const auto& Conf : Flags) {
            for (const auto pId : platform_id) {
                cl_uint num_device;
                err = clGetDeviceIDs(pId, Conf.type, 0, nullptr, &num_device);
                assert(err == CL_SUCCESS);
                std::vector<cl_device_id> devs(num_device);
                err = clGetDeviceIDs(pId, Conf.type, num_device, devs.data(), nullptr);
                assert(err == CL_SUCCESS);
                for (const auto dev : devs) {
                    auto Man = std::shared_ptr<AMPManager>(new OpenCLManager(dev, Conf.getPath()));
                    default_map[Man] = Man->createAloc();
                    Devices.push_back(Man);
                }
            }
        }

    }
};

static OpenCLContext ctx;

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
            Concurrency::KernelObject& KO = Concurrency::Pro2KernelObject[prog];
            std::map<std::string, std::string> aMap;
            for (unsigned i = 0; i < unsigned(kernel_num); ++i) {
                char s[1024] = { 0x0 };
                size_t size;
                ret = clGetKernelInfo(kl[i], CL_KERNEL_FUNCTION_NAME, 1024, s, &size);
                n.push_back(std::string (s));
                KO[std::string (s)] = kl[i];
                // Some analysis tool will post warnings about not releasing kernel object in time, 
                // for example, 
                //   Warning: Memory leak detected [Ref = 1, Handle = 0x12f1420]: Object created by clCreateKernelsInProgram
                //   Warning: Memory leak detected [Ref = 1, Handle = 0x12f17a0]: Object created by clCreateProgramWithBinary
                // However these won't be taken as memory leaks in here since all created kenrel objects 
                // will be released in ReleaseKernelObjects and the Ref count will be reset to 0
                #if 0
                clReleaseKernel(kl[i]);
                #endif
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

void CLCompileKernels(cl_program& program, cl_context& context, cl_device_id& device,
                      void* kernel_size_, void* kernel_source_)
{
    cl_int err;
    if (!program) {
        std::string kernel((const char*)kernel_source_, (size_t)((void *)kernel_size_));

        std::string hash_str = kernel;
        char name[256];
        err = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, NULL);
        assert(err == CL_SUCCESS);
        hash_str += "const const _device_name[] = \"";
        hash_str += name;
        hash_str += "\";";

        // calculate MD5 checksum
        unsigned char md5_hash[16];
        memset(md5_hash, 0, sizeof(unsigned char) * 16);
        MD5_CTX md5ctx;
        MD5_Init(&md5ctx);
        MD5_Update(&md5ctx, hash_str.data(), hash_str.length());
        MD5_Final(md5_hash, &md5ctx);

        // compute compiled kernel file name
        std::stringstream compiled_kernel_name;
        compiled_kernel_name << "/tmp/";
        compiled_kernel_name << std::setbase(16);
        for (int i = 0; i < 16; ++i) {
            compiled_kernel_name << static_cast<unsigned int>(md5_hash[i]);
        }
        compiled_kernel_name << ".bin";

        //std::cout << "Try load precompiled kernel: " << compiled_kernel_name.str() << std::endl;

        // check if pre-compiled kernel binary exist
        std::ifstream precompiled_kernel(compiled_kernel_name.str(), std::ifstream::binary);
        if (precompiled_kernel) {
            // use pre-compiled kernel binary
            precompiled_kernel.seekg(0, std::ios_base::end);
            size_t len = precompiled_kernel.tellg();
            precompiled_kernel.seekg(0, std::ios_base::beg);
            //std::cout << "Length of precompiled kernel: " << len << std::endl;
            unsigned char* compiled_kernel = new unsigned char[len];
            precompiled_kernel.read(reinterpret_cast<char*>(compiled_kernel), len);
            precompiled_kernel.close();

            const unsigned char *ks = (const unsigned char *)compiled_kernel;
            program = clCreateProgramWithBinary(context, 1, &device, &len, &ks, NULL, &err);
            if (err == CL_SUCCESS)
                err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
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
            delete [] compiled_kernel;
        } else {
            // pre-compiled kernel binary doesn't exist
            // call CL compiler

            if (kernel[0] == 'B' && kernel[1] == 'C') {
                // Bitcode magic number. Assuming it's in SPIR
                auto size = kernel.length();
                auto str = (const unsigned char*)kernel.data();
                program = clCreateProgramWithBinary(context, 1, &device, &size, &str, NULL, &err);
                if (err == CL_SUCCESS)
                    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
            } else {
                // in OpenCL-C
                auto size = kernel.length();
                auto str = kernel.data();
                program = clCreateProgramWithSource(context, 1, &str, &size, &err);
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

            //Get the number of devices attached with program object
            cl_uint nDevices = 0;
            err = clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES, sizeof(cl_uint),&nDevices, NULL);
            assert(nDevices == 1);
            assert(err == CL_SUCCESS);

            //Get the Id of all the attached devices
            cl_device_id *devices = new cl_device_id[nDevices];
            err = clGetProgramInfo(program, CL_PROGRAM_DEVICES, sizeof(cl_device_id) * nDevices, devices, NULL);
            assert(err == CL_SUCCESS);

            // Get the sizes of all the binary objects
            size_t *pgBinarySizes = new size_t[nDevices];
            err = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t) * nDevices, pgBinarySizes, NULL);
            assert(err == CL_SUCCESS);

            // Allocate storage for each binary objects
            unsigned char **pgBinaries = new unsigned char*[nDevices];
            for (cl_uint i = 0; i < nDevices; i++)
            {
                pgBinaries[i] = new unsigned char[pgBinarySizes[i]];
            }

            // Get all the binary objects
            err = clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(unsigned char*) * nDevices, pgBinaries, NULL);
            assert(err == CL_SUCCESS);

            // save compiled kernel binary
            std::ofstream compiled_kernel(compiled_kernel_name.str(), std::ostream::binary);
            compiled_kernel.write(reinterpret_cast<const char*>(pgBinaries[0]), pgBinarySizes[0]);
            compiled_kernel.close();

            //std::cout << "Kernel written to: " << compiled_kernel_name.str() << std::endl;

            // release memory
            for (cl_uint i = 0; i < nDevices; ++i) {
                delete [] pgBinaries[i];
            }
            delete [] pgBinaries;
            delete [] pgBinarySizes;
            delete [] devices;

        } // if (precompiled_kernel) 

        __mcw_cxxamp_compiled = true;
        getKernelNames(program);
    }
}

} // namespce CLAMP
} // namespace Concurrency

extern "C" void *GetContextImpl() {
    return &Concurrency::ctx;
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
extern "C" void PushArgPtrImpl(void *k_, int idx, size_t sz, const void *s) {
  cl_int err;
  err = clSetKernelArg(static_cast<cl_kernel>(k_), idx, sz, s);
  assert(err == CL_SUCCESS);
}
