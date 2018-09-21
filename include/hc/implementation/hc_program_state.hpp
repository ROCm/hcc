//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#pragma once

// TODO: this must be completely redone, it is representative of a stale
//       iteration of the approach to code object retrieval.

#include "hc_raii_handle.hpp"
#include "hc_code_object_bundle.hpp"
#include "../hc_agent_pool.hpp"
#include "../hc_runtime.hpp"

#include <hsa/hsa.h>

#include <elfio/elfio.hpp>

#include <link.h>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <mutex>
#include <ostream>
#include <string>
#include <unordered_map>

inline
bool operator==(hsa_isa_t x, hsa_isa_t y) noexcept
{
    return x.handle == y.handle;
}

namespace std
{
    template<>
    struct hash<hsa_isa_t> {
        std::size_t operator()(hsa_isa_t x) const noexcept
        {
            return std::hash<decltype(x.handle)>{}(x.handle);
        }
    };
}

namespace hc
{
    namespace detail
    {
        class Program_state {
            using RAIICodeObjectReader_ =
                RAII_move_only_handle<
                    hsa_code_object_reader_t,
                    decltype(hsa_code_object_reader_destroy)*>;
            using RAIIExecutable_ = RAII_move_only_handle<
                hsa_executable_t, decltype(hsa_executable_destroy)*>;

            using CodeObjectTable_ = std::unordered_map<
                hsa_isa_t, std::vector<RAIICodeObjectReader_>>;
            using ExecutableTable_ = std::unordered_map<
                hsa_agent_t, std::vector<RAIIExecutable_>>;
            using KernelTable_ = std::unordered_map<
                hsa_agent_t, std::vector<hsa_executable_symbol_t>>;

            // IMPLEMENTATION - STATICS
            template<typename T = std::vector<std::vector<char>>>
            static
            int copy_kernel_sections_(dl_phdr_info* info, size_t, void* kernels)
            {
                static constexpr const char self[]{"/proc/self/exe"};

                ELFIO::elfio reader;

                const auto f{info->dlpi_addr ? info->dlpi_name : self};

                if (!reader.load(f)) return 0;

                static constexpr const char kernel[]{".kernel"};
                const auto it{std::find_if(
                    std::cbegin(reader.sections),
                    std::cend(reader.sections),
                    [](auto&& x) { return x->get_name() == kernel; })};

                if (it == std::cend(reader.sections)) return 0;

                static_cast<T*>(kernels)->emplace_back(
                    (*it)->get_data(), (*it)->get_data() + (*it)->get_size());

                return 0;
            }

            static
            const std::vector<Bundled_code_header>& kernel_sections_()
            {
                static std::vector<Bundled_code_header> r;
                static std::once_flag f;

                std::call_once(f, []() {
                    std::vector<std::vector<char>> ks;
                    dl_iterate_phdr(copy_kernel_sections_<>, &ks);

                    for (auto&& x : ks) {
                        Bundled_code_header tmp{x};

                        if (valid(tmp)) r.push_back(std::move(tmp));
                    }
                });

                return r;
            }

            static
            RAIICodeObjectReader_ make_code_object_reader_(
                const std::vector<char>& x)
            {
                if (x.empty()) return {};

                RAIICodeObjectReader_ r{{}, hsa_code_object_reader_destroy};
                throwing_hsa_result_check(
                    hsa_code_object_reader_create_from_memory(
                        x.data(), x.size(), &handle(r)),
                    __FILE__, __func__, __LINE__);

                return r;
            }

            static
            void make_code_object_table_(
                const Bundled_code_header& x, CodeObjectTable_& y)
            {
                for (auto&& z : bundles(x)) {
                    y[triple_to_hsa_isa(z.triple)].push_back(
                        make_code_object_reader_(z.blob));
                }
                y.erase(hsa_isa_t{0});
            }

            static
            const CodeObjectTable_& code_objects_()
            {
                static CodeObjectTable_ r;
                static std::once_flag f;

                std::call_once(f, []() {
                    for (auto&& x : kernel_sections_()) {
                        make_code_object_table_(x, r);
                    }
                });

                return r;
            }

            static
            hsa_isa_t agent_isa_(hsa_agent_t x)
            {
                hsa_isa_t r{};
                throwing_hsa_result_check(
                    hsa_agent_iterate_isas(x, [](hsa_isa_t isa, void* p) {
                        *static_cast<hsa_isa_t*>(p) = isa;

                        return HSA_STATUS_SUCCESS;
                    }, &r),
                    __FILE__, __func__, __LINE__);

                return r;
            }

            static
            RAIIExecutable_ make_executable_(
                const RAIICodeObjectReader_& x, hsa_agent_t a)
            {
                RAIIExecutable_ r{{}, hsa_executable_destroy};

                throwing_hsa_result_check(
                    hsa_executable_create_alt(
                        HSA_PROFILE_FULL,// TODO: this is a bug.
                        HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                        nullptr,
                        &handle(r)),
                    __FILE__, __func__, __LINE__);

                throwing_hsa_result_check(
                    hsa_executable_load_agent_code_object(
                        handle(r), a, handle(x), nullptr, nullptr),
                    __FILE__, __func__, __LINE__);

                throwing_hsa_result_check(
                    hsa_executable_freeze(handle(r), nullptr),
                    __FILE__, __func__, __LINE__);

                return r;
            }

            static
            void make_executable_table_(
                const CodeObjectTable_& x, ExecutableTable_& y)
            {
                for (auto&& agent : Agent_pool::pool()) {
                    const auto it = x.find(agent_isa_(agent.first));

                    if (it == x.cend()) continue;

                    for (auto&& z : it->second) {
                        y[agent.first].push_back(
                            make_executable_(z, agent.first));
                    }
                }
            }

            static
            const ExecutableTable_& executables_()
            {
                static ExecutableTable_ r;
                static std::once_flag f;

                std::call_once(f, []() {
                    make_executable_table_(code_objects_(), r);
                });

                return r;
            }

            static
            bool is_kernel_(hsa_executable_symbol_t x)
            {
                hsa_symbol_kind_t r{};
                throwing_hsa_result_check(
                    hsa_executable_symbol_get_info(
                        x, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &r),
                    __FILE__, __func__, __LINE__);

                return r == HSA_SYMBOL_KIND_KERNEL;
            }

            static
            hsa_status_t copy_kernel_symbols(
                hsa_executable_t,
                hsa_agent_t,
                hsa_executable_symbol_t y,
                void* z)
            {
                auto p = static_cast<typename KernelTable_::mapped_type*>(z);

                if (is_kernel_(y)) p->push_back(y);

                return HSA_STATUS_SUCCESS;
            }

            static
            void make_kernel_table_(const ExecutableTable_& x, KernelTable_& y)
            {
                for (auto&& e : x) {
                    for (auto&& ex : e.second) {
                        throwing_hsa_result_check(
                            hsa_executable_iterate_agent_symbols(
                                handle(ex),
                                e.first,
                                copy_kernel_symbols,
                                &y[e.first]),
                            __FILE__, __func__, __LINE__);
                    }
                }
            }
        public:
            static
            KernelTable_& kernels()
            {
                static KernelTable_ r;
                static std::once_flag f;

                std::call_once(f, []() {
                    make_kernel_table_(executables_(), r);
                });

                return r;
            }
        };
    }// Namespace hc::detail.
} // Namespace hc.