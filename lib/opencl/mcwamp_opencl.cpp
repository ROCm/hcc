//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// FIXME this file will place C++AMP Runtime implementation (OpenCL version)

#include <amp.h>
#include <CL/opencl.h>
namespace Concurrency {

AMPAllocator& getAllocator()
{
    static AMPAllocator amp;
    return amp;
}

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

extern std::vector<std::string> __mcw_kernel_names;

namespace Concurrency {
namespace CLAMP {
    static inline void getKernelNames(cl_program& prog) {
        std::vector<std::string> n;
        cl_uint kernel_num = 0;
        cl_uint ret = CL_SUCCESS;
        char **names;
        int count;
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

    void CompileKernels(cl_program& program, cl_context& context, cl_device_id& device)
    {
        cl_int err;
        if (!__mcw_cxxamp_compiled) {
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
