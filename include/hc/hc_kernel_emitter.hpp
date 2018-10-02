//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "hc_agent_pool.hpp"
#include "hc_callable_attributes.hpp"
#include "hc_defines.hpp"
#include "hc_index.hpp"
#include "implementation/hc_program_state.hpp"

#include <elfio/elfio.hpp>

#include <link.h>

#include <cstdint>
#include <exception>
#include <mutex>
#include <type_traits>
#include <unordered_map>

namespace hc
{
    template<int> class tiled_index;

    namespace detail
    {
       struct Indexer {
            template<int n>
            operator index<n>() const noexcept [[hc]]
            {
                index<n> tmp;
                for (auto i = 0; i != n; ++i) {
                    tmp[n - i - 1] = hc_get_workitem_absolute_id(i);
                }

                return tmp;
            }

            template<int n>
            operator hc::tiled_index<n>() const noexcept [[hc]]
            {
                return {};
            }
        };

        template<typename Kernel>
        inline
        const char* linker_name_for()
        {
            static std::once_flag f{};
            static std::string r{};

            std::call_once(f, [&]() {
                dl_iterate_phdr([](dl_phdr_info* info, std::size_t, void* pr) {
                    const auto base = info->dlpi_addr;
                    ELFIO::elfio elf;

                    if (!elf.load(base ? info->dlpi_name : "/proc/self/exe")) {
                        return 0;
                    }

                    struct Symbol {
                        std::string name;
                        ELFIO::Elf64_Addr value;
                        ELFIO::Elf_Xword size;
                        unsigned char bind;
                        unsigned char type;
                        ELFIO::Elf_Half section_index;
                        unsigned char other;
                    } tmp{};
                    for (auto&& section : elf.sections) {
                        if (section->get_type() != SHT_SYMTAB) continue;

                        ELFIO::symbol_section_accessor fn{elf, section};

                        static const auto k_addr = reinterpret_cast<
                            std::uintptr_t>(&Kernel::entry_point);
                        auto n = fn.get_symbols_num();
                        while (n--) {
                            fn.get_symbol(
                            n,
                            tmp.name,
                            tmp.value,
                            tmp.size,
                            tmp.bind,
                            tmp.type,
                            tmp.section_index,
                            tmp.other);

                            if (tmp.type != STT_FUNC) continue;

                            if (tmp.value + base == k_addr) {
                                *static_cast<std::string*>(pr) = tmp.name;

                                return 1;
                            }
                        }
                    }

                    return 0;
                }, &r);
            });

            if (!r.empty()) return r.c_str();

            throw std::runtime_error{
                std::string{"Kernel: "} +
                typeid(&Kernel::entry_point).name() +
                " is not available."};
        }

        template<typename Kernel>
        class HSA_kernel {
            template<typename, typename, typename>
            friend
            class Kernel_emitter_base;

            // IMPLEMENTATION - DATA - STATICS
            inline static std::string name_{linker_name_for<Kernel>()};

            // IMPLEMENTATION - DATA
            hsa_executable_symbol_t kernel_{};

            // IMPLEMENTATION - STATICS
            static
            std::string symbol_name_(hsa_executable_symbol_t x)
            {   // TODO: this uses deprecated HSA APIs because ROCr did not
                //       implement the updated ones.
                std::size_t sz{};
                throwing_hsa_result_check(
                    hsa_executable_symbol_get_info(
                        x, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &sz),
                    __FILE__, __func__, __LINE__);

                std::string r(sz, '\0');
                throwing_hsa_result_check(
                    hsa_executable_symbol_get_info(
                        x, HSA_EXECUTABLE_SYMBOL_INFO_NAME, &r[0]),
                    __FILE__, __func__, __LINE__);

                return r;
            }

            static
            std::uint32_t group_size_(hsa_executable_symbol_t x)
            {
                std::uint32_t r{};
                throwing_hsa_result_check(
                    hsa_executable_symbol_get_info(
                        x,
                        HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
                        &r),
                    __FILE__, __func__, __LINE__);

                return r;
            }

            static
            std::uint64_t kernel_object_(hsa_executable_symbol_t x)
            {
                std::uint64_t r{};
                throwing_hsa_result_check(
                    hsa_executable_symbol_get_info(
                        x, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &r),
                    __FILE__, __func__, __LINE__);

                return r;
            }

            static
            hsa_executable_symbol_t kernel_symbol_(hsa_agent_t x)
            {
                for (auto&& kernel : Program_state::kernels()[x]) {
                    if (name_ == symbol_name_(kernel)) return kernel;
                }

                throw std::runtime_error{
                    "Code for kernel " + name_ + " is unavailable."};
            }

            static
            std::uint32_t private_size_(hsa_executable_symbol_t x)
            {
                std::uint32_t r{};
                throwing_hsa_result_check(
                    hsa_executable_symbol_get_info(
                        x,
                        HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
                        &r),
                    __FILE__, __func__, __LINE__);

                return r;
            }

            // IMPLEMENTATION - CREATORS
            explicit
            HSA_kernel(hsa_agent_t x)
            try :
                kernel_{kernel_symbol_(x)},
                group_size{group_size_(kernel_)},
                kernel_object{kernel_object_(kernel_)},
                private_size{private_size_(kernel_)}
            {}
            catch (const std::exception& ex) {
                std::cerr << ex.what() << std::endl;

                throw;
            }
        public:
            // DATA
            std::uint32_t group_size{};
            std::uint64_t kernel_object{};
            std::uint32_t private_size{};

            // CREATORS
            HSA_kernel() = default;
        };

        template<typename Index, typename Kernel, typename Emitter>
        class Kernel_emitter_base {
            // TODO: this validation should be done further above, in pfe
            //       itself, for more clarity. It is also a placeholder.
            static
            std::false_type is_callable_(...) noexcept [[cpu, hc]];
            template<typename I, typename K>
            static
            auto is_callable_(I* idx, const K* f) noexcept [[cpu, hc]]
                -> decltype((*f)(*idx), std::true_type{});

            static_assert(
                decltype(is_callable_(
                    std::declval<Index*>(), std::declval<const Kernel*>())){},
                "Invalid Callable passed to parallel_for_each.");
        public:
            static
            std::unordered_map<hsa_agent_t, HSA_kernel<Emitter>>& kernel()
            {
                static std::unordered_map<hsa_agent_t, HSA_kernel<Emitter>> r;
                static std::once_flag f;

                std::call_once(f, []() {
                    for (auto&& agent : Agent_pool::pool()) {
                        if (agent.second.is_cpu) continue;

                        r.emplace(
                            agent.first, HSA_kernel<Emitter>{agent.first});
                    }
                });

                return r;
            }
        };

        template<typename T>
        inline
        void ignore_arg(T&&)
        {}

        template<typename Index, typename Kernel>
        struct Kernel_emitter :
            public Kernel_emitter_base<
                Index, Kernel, Kernel_emitter<Index, Kernel>> {
            static
            __attribute__((used, annotate("__HCC_KERNEL__")))
            void entry_point(Kernel f) noexcept [[cpu, hc]]
            {
                #if __HCC_ACCELERATOR__ != 0
                    Index tmp = Indexer{};
                    f(tmp);
                #else
                    ignore_arg(f);
                #endif
            }
        };

        template<typename Kernel, typename... Attrs>
        using Kernel_with_attributes =
            hc::attr_impl::Callable_with_AMDGPU_attributes<Kernel, Attrs...>;

        template<typename Index, typename Kernel, typename... Attrs>
        struct Kernel_emitter<Index, Kernel_with_attributes<Kernel, Attrs...>> :
            public Kernel_emitter_base<
                Index,
                Kernel_with_attributes<Kernel, Attrs...>,
                Kernel_emitter<
                    Index, Kernel_with_attributes<Kernel, Attrs...>>> {
            using K = Kernel_with_attributes<Kernel, Attrs...>;

            static
            __attribute__((
                used,
                annotate("__HCC_KERNEL__"),
                amdgpu_flat_work_group_size(
                    K::Flat_wg_size_::minimum(), K::Flat_wg_size_::maximum()),
                amdgpu_waves_per_eu(
                    K::Waves_per_EU_::minimum(), K::Waves_per_EU_::maximum())))
            void entry_point(K f) noexcept [[cpu, hc]]
            {
                #if __HCC_ACCELERATOR__ != 0
                    Index tmp = Indexer{};
                    f(tmp);
                #else
                    ignore_arg(f);
                #endif
            }
        };
    } // Namespace hc::detail.
} // Namespace hc.