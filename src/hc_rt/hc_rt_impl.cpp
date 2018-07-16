//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <hc.hpp>
#include <hc_rt_debug.hpp>

#include "hc_rt_impl.hpp"

#include <dlfcn.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <string>
#include <tuple>

namespace hc
{
    const wchar_t accelerator::cpu_accelerator[];
    const wchar_t accelerator::default_accelerator[];

    // array_base
    const std::size_t array_base::max_array_cnt_;

    // array_view_base
    const std::size_t array_view_base::max_array_view_cnt_;
} // namespace hc

namespace detail
{
    RuntimeImpl* GetOrInitRuntime()
    {
        static RuntimeImpl* runtimeImpl = nullptr;
        if (runtimeImpl == nullptr) {
            HSAPlatformDetect hsa_rt;

            char* verbose_env = getenv("HCC_VERBOSE");
            if (verbose_env != nullptr) {
            if (std::string("ON") == verbose_env) {
                mcwamp_verbose = true;
            }
            }

            // force use certain C++AMP runtime from HCC_RUNTIME environment variable
            char* runtime_env = getenv("HCC_RUNTIME");
            if (runtime_env != nullptr) {
            if (std::string("HSA") == runtime_env) {
                if (hsa_rt.detect()) {
                runtimeImpl = LoadHSARuntime();
                } else {
                std::cerr << "Ignore unsupported HCC_RUNTIME environment variable: " << runtime_env << std::endl;
                }
            } else if(std::string("CPU") == runtime_env) {
                // CPU runtime should be available
                runtimeImpl = LoadCPURuntime();
                runtimeImpl->set_cpu();
            } else {
                std::cerr << "Ignore unknown HCC_RUNTIME environment variable:" << runtime_env << std::endl;
            }
            }

            // If can't determined by environment variable, try detect what can be used
            if (runtimeImpl == nullptr) {
            if (hsa_rt.detect()) {
                runtimeImpl = LoadHSARuntime();
            } else {
                runtimeImpl = LoadCPURuntime();
                runtimeImpl->set_cpu();
                std::cerr << "No suitable runtime detected. Fall back to CPU!" << std::endl;
            }
            }
        }
        return runtimeImpl;
    }

    static bool in_kernel = false;
    bool in_cpu_kernel() { return in_kernel; }
    void enter_kernel() { in_kernel = true; }
    void leave_kernel() { in_kernel = false; }

    /// Handler for binary files. The bundled file will have the following format
    /// (all integers are stored in little-endian format):
    ///
    /// "OFFLOAD_BUNDLER_MAGIC_STR" (ASCII encoding of the string)
    ///
    /// NumberOfOffloadBundles (8-byte integer)
    ///
    /// OffsetOfBundle1 (8-byte integer)
    /// SizeOfBundle1 (8-byte integer)
    /// NumberOfBytesInTripleOfBundle1 (8-byte integer)
    /// TripleOfBundle1 (byte length defined before)
    ///
    /// ...
    ///
    /// OffsetOfBundleN (8-byte integer)
    /// SizeOfBundleN (8-byte integer)
    /// NumberOfBytesInTripleOfBundleN (8-byte integer)
    /// TripleOfBundleN (byte length defined before)
    ///
    /// Bundle1
    /// ...
    /// BundleN

    static
    inline
    std::uint64_t Read8byteIntegerFromBuffer(const char *data, std::size_t pos)
    {
        std::uint64_t Res = 0;
        for (unsigned i = 0; i < 8; ++i) {
            Res <<= 8;
            std::uint64_t Char = (std::uint64_t)data[pos + 7 - i];
            Res |= 0xffu & Char;
        }
        return Res;
    }

    #define RUNTIME_ERROR(val, error_string, line) { \
    hc::print_backtrace(); \
    printf("### HCC RUNTIME ERROR: %s at file:%s line:%d\n", error_string, __FILENAME__, line); \
    exit(val); \
    }

    #define OFFLOAD_BUNDLER_MAGIC_STR "__CLANG_OFFLOAD_BUNDLE__"
    #define OFFLOAD_BUNDLER_MAGIC_STR_LENGTH (24)
    #define HCC_TRIPLE_PREFIX "hcc-amdgcn-amd-amdhsa--"
    #define HCC_TRIPLE_PREFIX_LENGTH (23)

    // Try determine a compatible code object within kernel bundle for a queue
    // Returns true if a compatible code object is found, and returns its size and
    // pointer to the code object. Returns false in case no compatible code object
    // is found.
    bool DetermineAndGetProgram(
        HCCQueue* pQueue, std::size_t* kernel_size, void** kernel_source)
    {
        bool FoundCompatibleKernel = false;

        // walk through bundle header
        // get bundle file size
        std::size_t bundle_size =
            (std::ptrdiff_t)((void *)kernel_bundle_end) -
            (std::ptrdiff_t)((void *)kernel_bundle_source);

        // point to bundle file data
        const char *data = (const char *)kernel_bundle_source;

        // skip OFFLOAD_BUNDLER_MAGIC_STR
        std::size_t pos = 0;
        if (pos + OFFLOAD_BUNDLER_MAGIC_STR_LENGTH > bundle_size) {
            RUNTIME_ERROR(1, "Bundle size too small", __LINE__)
        }
        std::string MagicStr(data + pos, OFFLOAD_BUNDLER_MAGIC_STR_LENGTH);
        if (MagicStr.compare(OFFLOAD_BUNDLER_MAGIC_STR) != 0) {
            RUNTIME_ERROR(1, "Incorrect magic string", __LINE__)
        }
        pos += OFFLOAD_BUNDLER_MAGIC_STR_LENGTH;

        // Read number of bundles.
        if (pos + 8 > bundle_size) {
            RUNTIME_ERROR(1, "Fail to parse number of bundles", __LINE__)
        }
        std::uint64_t NumberOfBundles = Read8byteIntegerFromBuffer(data, pos);
        pos += 8;

        for (std::uint64_t i = 0; i < NumberOfBundles; ++i) {
            // Read offset.
            if (pos + 8 > bundle_size) {
            RUNTIME_ERROR(1, "Fail to parse bundle offset", __LINE__)
            }
            std::uint64_t Offset = Read8byteIntegerFromBuffer(data, pos);
            pos += 8;

            // Read size.
            if (pos + 8 > bundle_size) {
            RUNTIME_ERROR(1, "Fail to parse bundle size", __LINE__)
            }
            uint64_t Size = Read8byteIntegerFromBuffer(data, pos);
            pos += 8;

            // Read triple size.
            if (pos + 8 > bundle_size) {
            RUNTIME_ERROR(1, "Fail to parse triple size", __LINE__)
            }
            uint64_t TripleSize = Read8byteIntegerFromBuffer(data, pos);
            pos += 8;

            // Read triple.
            if (pos + TripleSize > bundle_size) {
            RUNTIME_ERROR(1, "Fail to parse triple", __LINE__)
            }
            std::string Triple(data + pos, TripleSize);
            pos += TripleSize;

            // only check bundles with HCC triple prefix string
            if (Triple.compare(0, HCC_TRIPLE_PREFIX_LENGTH, HCC_TRIPLE_PREFIX) == 0) {
                // use HCCDevice::IsCompatibleKernel to check
                std::size_t SizeST = (std::size_t)Size;
                void *Content = (unsigned char *)data + Offset;
                if (pQueue->getDev()->IsCompatibleKernel((void*)SizeST, Content)) {
                    *kernel_size = SizeST;
                    *kernel_source = Content;
                    FoundCompatibleKernel = true;
                    break;
                }
            }
        }

        return FoundCompatibleKernel;
    }

    void LoadInMemoryProgram(HCCQueue* pQueue)
    {
        std::size_t kernel_size = 0;
        void* kernel_source = nullptr;

        // Only call BuildProgram in case a compatible code object is found
        if (DetermineAndGetProgram(pQueue, &kernel_size, &kernel_source)) {
            pQueue->getDev()->BuildProgram((void*)kernel_size, kernel_source);
        }
    }

    // used in parallel_for_each.h
    void* CreateKernel(
        const char* name,
        HCCQueue* pQueue,
        std::unique_ptr<void, void (*)(void*)> callable,
        std::size_t callable_size)
    {
    // TODO - should create a HSAQueue:: CreateKernel member function that
    //        creates and returns a dispatch.
        return pQueue->getDev()->CreateKernel(
            name, pQueue, std::move(callable), callable_size);
    }

    HCCContext* getContext()
    {
        return static_cast<HCCContext*>(GetOrInitRuntime()->m_GetContextImpl());
    }
} // namespace detail