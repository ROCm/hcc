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
    void CLCompileKernels(cl_program&, cl_device_id&, void*, void*);
}

struct DimMaxSize {
  cl_uint dimensions;
  size_t* maxSizes;
};

class OpenCLAllocator;
static cl_context context;
static std::map<cl_mem, cl_event> events;
static std::map<cl_event, std::vector<cl_mem>> etomem;

static inline void callback_release_kernel(cl_event event, cl_int event_command_exec_status, void *user_data)
{
    if (user_data)
        clReleaseKernel(static_cast<cl_kernel>(user_data));
}

class OpenCLManager : public AMPManager
{
public:
    OpenCLManager(const cl_device_id device, const std::wstring& path)
        : AMPManager(), programs(), device(device), path(path) {
        cl_int err;

        cl_ulong memAllocSize;
        err = clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &memAllocSize, NULL);
        assert(err == CL_SUCCESS);
        mem = memAllocSize >> 10;

        char vendor[256];
        err = clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(vendor), vendor, NULL);
        assert(err == CL_SUCCESS);
        std::string ven(vendor);
        if (ven.find("Advanced Micro Devices") != std::string::npos)
            description = L"AMD";
        else if (ven.find("NVIDIA") != std::string::npos)
            description = L"NVIDIA";
        else if (ven.find("Intel") != std::string::npos)
            description = L"Intel";

        cl_uint dimensions = 0;
        err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &dimensions, NULL);
        assert(err == CL_SUCCESS);
        size_t *maxSizes = new size_t[dimensions];
        err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * dimensions, maxSizes, NULL);
        assert(err == CL_SUCCESS);
        d.dimensions = dimensions;
        d.maxSizes = maxSizes;
        cpu_type = access_type_none;
    }

    std::wstring get_path() override { return path; }
    std::wstring get_description() override { return description; }
    size_t get_mem() override { return mem; }
    bool is_double() override { return true; }
    bool is_lim_double() override { return true; }
    bool is_unified() override { return false; }
    bool is_emulated() override { return false; }

    void* CreateKernel(const char* fun, void* size, void* source) override {
        cl_int err;
        if (programs.find(source) == std::end(programs)) {
            cl_program program = nullptr;
            Concurrency::CLAMP::CLCompileKernels(program, device, size, source);
            programs[source] = program;
        }
        cl_program program = programs[source];
        std::string name(fun);
        cl_kernel kernel = clCreateKernel(program, fun, &err);
        assert(err == CL_SUCCESS);
        return kernel;
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
        for (auto& it : programs)
            clReleaseProgram(it.second);
        delete[] d.maxSizes;
    }

    cl_device_id getDevice() const { return device; }
    void* create(size_t count) override {
        cl_int err;
        cl_mem dm = clCreateBuffer(context, CL_MEM_READ_WRITE, count, nullptr, &err);
        assert(err == CL_SUCCESS);
        return dm;
    }
    void release(void *device) override {
        cl_mem dm = static_cast<cl_mem>(device);
        events.erase(dm);
        clReleaseMemObject(dm);
    }
    void discard(void *device) override {
        cl_mem dm = static_cast<cl_mem>(device);
        events.erase(dm);
    }
    std::shared_ptr<AMPAllocator> createAloc() override { return newAloc(); }


private:
    std::shared_ptr<AMPAllocator> newAloc();
    std::map<void*, cl_program> programs;
    struct DimMaxSize d;
    cl_device_id     device;
    std::wstring path;
    std::wstring description;
    size_t mem;
};

struct cl_info
{
    cl_mem dm;
    bool isConst;
};

class OpenCLAllocator : public AMPAllocator
{
    enum { queue_size = 4 };
    cl_command_queue queues[queue_size];
    int idx;
    std::vector<cl_info> mems;
    cl_command_queue getQueue() { return queues[(idx++) % queue_size]; }
public:
    OpenCLAllocator(std::shared_ptr<AMPManager> pMan) : AMPAllocator(pMan), mems() {
        auto Man = std::dynamic_pointer_cast<OpenCLManager, AMPManager>(pMan);
        cl_int err;
        idx = 0;
        for (int i = 0; i < queue_size; ++i) {
            queues[i] = clCreateCommandQueue(context, Man->getDevice(), CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
            assert(err == CL_SUCCESS);
        }

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
        for (auto queue : queues) {
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
        }
        pthread_attr_destroy(&attr);
    }

    void flush() override {
        for (auto queue : queues)
            clFlush(queue);
    }
    void wait() override {
        for (auto queue : queues)
            clFinish(queue);
    }
    ~OpenCLAllocator() {
        for (auto queue : queues)
            clReleaseCommandQueue(queue);
    }

    void Push(void *kernel, int idx, void*& data, void* device, bool isConst) override {
        cl_mem dm = static_cast<cl_mem>(device);
        PushArgImpl(kernel, idx, sizeof(cl_mem), &dm);
        mems.push_back({dm, isConst});
    }

    void write(void* device, const void *src, size_t count, size_t offset, bool blocking) override {
        cl_mem dm = static_cast<cl_mem>(device);
        cl_int err = CL_SUCCESS;
        if (blocking)
            err = clEnqueueWriteBuffer(getQueue(), dm, CL_TRUE, offset, count, src, 0, NULL, NULL);
        else {
            cl_event ent;
            err = clEnqueueWriteBuffer(getQueue(), dm, CL_FALSE, offset, count, src, 0, NULL, &ent);
            events[dm] = ent;
        }
        assert(err == CL_SUCCESS);
    }
    void read(void* device, void* dst, size_t count, size_t offset) override {
        cl_mem dm = static_cast<cl_mem>(device);
        cl_int err;
        if (events.find(dm) != std::end(events))
            err = clEnqueueReadBuffer(getQueue(), dm, CL_TRUE, offset, count, dst, 1, &events[dm], NULL);
        else
            err = clEnqueueReadBuffer(getQueue(), dm, CL_TRUE, offset, count, dst, 0, NULL, NULL);
        assert(err == CL_SUCCESS);
    }
    void copy(void* src, void* dst, size_t count, size_t src_offset, size_t dst_offset) override {
        cl_mem sdm = static_cast<cl_mem>(src);
        cl_mem ddm = static_cast<cl_mem>(dst);
        cl_int err;
        cl_event evt;
        if (events.find(sdm) == std::end(events))
            err = clEnqueueCopyBuffer(getQueue(), sdm, ddm, src_offset, dst_offset, count, 0, NULL, &evt);
        else
            err = clEnqueueCopyBuffer(getQueue(), sdm, ddm, src_offset, dst_offset, count, 1, &events[sdm], &evt);
        assert(err == CL_SUCCESS);
        events[ddm] = evt;
    }
    void* map(void* device, size_t count, size_t offset, bool Write) override {
        cl_mem dm = static_cast<cl_mem>(device);
        cl_int err;
        cl_map_flags flags;
        if (Write)
            flags = CL_MAP_WRITE_INVALIDATE_REGION;
        else
            flags = CL_MAP_READ;
        void* addr = nullptr;
        if (events.find(dm) == std::end(events))
            addr = clEnqueueMapBuffer(getQueue(), dm, CL_TRUE, flags, offset, count, 0, NULL, NULL, &err);
        else
            addr = clEnqueueMapBuffer(getQueue(), dm, CL_TRUE, flags, offset, count, 1, &events[dm], NULL, &err);
        assert(err == CL_SUCCESS);
        return addr;
    }
    void unmap(void* device, void* addr) override {
        cl_mem dm = static_cast<cl_mem>(device);
        cl_int err = clEnqueueUnmapMemObject(getQueue(), dm, addr, 0, NULL, NULL);
        assert(err == CL_SUCCESS);
    }
    void LaunchKernel(void *kernel, size_t dim_ext, size_t *ext, size_t *local_size) override {
        cl_int err;
        auto Man = std::dynamic_pointer_cast<OpenCLManager, AMPManager>(getMan());
        if(!Man->check(local_size, dim_ext))
            local_size = NULL;
        std::vector<cl_event> eve;
        std::for_each(std::begin(mems), std::end(mems),
                      [&] (const cl_info& mm) {
                        if (events.find(mm.dm) != std::end(events))
                            eve.push_back(events[mm.dm]);
                      });
        std::sort(std::begin(eve), std::end(eve));
        eve.erase(std::unique(std::begin(eve), std::end(eve)), std::end(eve));
        cl_event evt;
        err = clEnqueueNDRangeKernel(getQueue(), (cl_kernel)kernel, dim_ext, NULL, ext, local_size,
                                     eve.size(), eve.data(), &evt);
        assert(err == CL_SUCCESS);
        err = clSetEventCallback(evt, CL_COMPLETE, &callback_release_kernel, (cl_kernel)kernel);
        assert(err == CL_SUCCESS);
        std::for_each(std::begin(mems), std::end(mems),
                      [&](const cl_info& mm) {
                      if (!mm.isConst)
                        events[mm.dm] = evt;
                      });
        mems.clear();
    }
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
                                  CLFlag(CL_DEVICE_TYPE_CPU, L"cpu")
                                  });

        std::vector<cl_device_id> devs;
        std::vector<std::wstring> path;
        for (const auto& Conf : Flags) {
            for (const auto pId : platform_id) {
                cl_uint num_device;
                err = clGetDeviceIDs(pId, Conf.type, 0, nullptr, &num_device);
                assert(err == CL_SUCCESS);
                if (num_device == 0)
                    continue;
                std::vector<cl_device_id> dev(num_device);
                err = clGetDeviceIDs(pId, Conf.type, num_device, dev.data(), nullptr);
                assert(err == CL_SUCCESS);
                for (int i = 0; i < num_device; ++i) {
                    path.push_back(Conf.getPath());
                    devs.push_back(dev[i]);
                }
            }
        }
        context = clCreateContext(0, devs.size(), devs.data(), NULL, NULL, &err);
        assert(err == CL_SUCCESS);
        for (int i = 0; i < devs.size(); ++i) {
            auto Man = std::shared_ptr<AMPManager>(new OpenCLManager(devs[i], path[i]));
            default_map[Man] = Man->createAloc();
            if (i == 0)
                def = Man;
            Devices.push_back(Man);
        }
    }
    ~OpenCLContext() { clReleaseContext(context); }
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

void CLCompileKernels(cl_program& program, cl_device_id& device,
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
            program = clCreateProgramWithBinary(Concurrency::context, 1, &device, &len, &ks, NULL, &err);
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
                program = clCreateProgramWithBinary(Concurrency::context, 1, &device, &size, &str, NULL, &err);
                if (err == CL_SUCCESS)
                    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
            } else {
                // in OpenCL-C
                auto size = kernel.length();
                auto str = kernel.data();
                program = clCreateProgramWithSource(Concurrency::context, 1, &str, &size, &err);
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
static inline int ldistance(const std::string& source, const std::string& target)
{
  int n = source.length();
  int m = target.length();
  if (m == 0)
    return n;
  if (n == 0)
    return m;

  //Construct a matrix
  typedef std::vector < std::vector < int >>Tmatrix;
  Tmatrix matrix(n + 1, std::vector<int>(m + 1));

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
extern "C" void MatchKernelNamesImpl(std::string& fixed_name) {
    return;
    if (__mcw_kernel_names.size()) {
    // Must start from a big value > 10
    int distance = 1024;
    int hit = -1;
    std::vector<std::string>::iterator shortest;
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
        shortest = it;
      }
    }
    /* Replacement. Skip if not hit or the distance is too far (>5)*/
    if (hit >= 0 && distance < 5)
        fixed_name = *shortest;
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
