//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "raii_handle.hpp"
#include "../functions/integer_computational_basis.hpp"
#include "../functions/hsa_interfaces.hpp"
#include "../types/code_object_bundle.hpp"

#include <hc.hpp>

#include <hsa/hsa.h>

#include "../../external/elfio/elfio.hpp"

#include <link.h>

#include <cstddef>
#include <iterator>
#include <mutex>
#include <ostream>
#include <string>
#include <unordered_map>

namespace hc2
{
    namespace
    {
        using RAII_code_object_reader =
            RAII_move_only_handle<
                hsa_code_object_reader_t,
                decltype(hsa_code_object_reader_destroy)*>;
        using RAII_executable = RAII_move_only_handle<
            hsa_executable_t, decltype(hsa_executable_destroy)*>;

        inline
        RAII_code_object_reader make_code_object_reader(
            const char* f, std::size_t n)
        {
            RAII_code_object_reader r{{}, hsa_code_object_reader_destroy};

            if (positive(n)) {
                throwing_hsa_result_check(
                    hsa_code_object_reader_create_from_memory(f, n, &handle(r)),
                    __FILE__,
                    __func__,
                    __LINE__);
            }

            return r;
        }

        inline
        RAII_code_object_reader make_code_object_reader(
            const std::vector<char>& x)
        {
            return make_code_object_reader(x.data(), x.size());
        }

        inline
        RAII_executable executable(
            const RAII_code_object_reader& x, hsa_agent_t a)
        {
            RAII_executable r{{}, hsa_executable_destroy};

            throwing_hsa_result_check(
                hsa_executable_create_alt(
                    HSA_PROFILE_FULL,//hsa_agent_profile(av) - TODO: this is a bug.
                    hsa_agent_float_rounding_mode(a),
                    nullptr,
                    &handle(r)),
                __FILE__,
                __func__,
                __LINE__);

            throwing_hsa_result_check(
                hsa_executable_load_agent_code_object(
                    handle(r), a, handle(x), nullptr, nullptr),
                __FILE__,
                __func__,
                __LINE__);

            // TODO: temporary.
            std::uint32_t v = UINT32_MAX;
            throwing_hsa_result_check(
                hsa_executable_validate_alt(handle(r), nullptr, &v),
                __FILE__,
                __func__,
                __LINE__);

            assert(zero(v));

            throwing_hsa_result_check(
                hsa_executable_freeze(handle(r), nullptr),
                __FILE__,
                __func__,
                __LINE__);

            return r;
        }
    }

    class Program_state {
        using Code_object_table = std::unordered_map<
            hsa_isa_t, std::vector<RAII_code_object_reader>>;
        using Executable_table = std::unordered_map<
            hsa_agent_t, std::vector<RAII_executable>>;
        using Kernel_table = std::unordered_map<
            hsa_agent_t, std::vector<hsa_executable_symbol_t>>;

        friend const Program_state& program_state();

        friend
        inline
        const Kernel_table& shared_object_kernels(const Program_state& x)
        {
            return x.shared_object_kernel_table_();
        }

        std::vector<hc::accelerator> acc_;

        template<typename T = std::vector<std::vector<char>>>
        static
        int copy_kernel_sections_(dl_phdr_info* x, size_t, void* kernels)
        {
            static constexpr const char kernel[] = ".kernel";

            auto out = static_cast<T*>(kernels);

            ELFIO::elfio tmp;
            if (tmp.load(x->dlpi_name)) {
                for (auto&& y : tmp.sections) {
                    if (y->get_name() == kernel) {
                        out->emplace_back(
                            y->get_data(), y->get_data() + y->get_size());
                    }
                }
            }

            return 0;
        }

        static
        const std::vector<Bundled_code_header>& shared_object_kernel_sections_()
        {
            static std::vector<Bundled_code_header> r;

            static std::once_flag f;
            std::call_once(f, []() {
                std::vector<std::vector<char>> ks;
                dl_iterate_phdr(copy_kernel_sections_<>, &ks);
                for (auto&& x : ks) {
                    size_t offset = 0;
                    while(offset < x.size()) {
                        size_t read_bundle_size = 0;
                        Bundled_code_header tmp{x.cbegin()+offset,
                                                x.cend(), 
                                                &read_bundle_size};
                        if (valid(tmp)) {
                            r.push_back(std::move(tmp));
                            offset+=read_bundle_size;
                        }
                        else {
                            break;
                        }
                    }
                }
            });

            return r;
        }

        static
        void make_code_object_table_(
            const Bundled_code_header& x, Code_object_table& y)
        {
            for (auto&& z : bundles(x)) {
                y[triple_to_hsa_isa(z.triple)].push_back(
                    make_code_object_reader(z.blob));
            }
            y.erase(hsa_isa_t{0});
        }

        static
        const Code_object_table& shared_object_code_object_table_()
        {
            static Code_object_table r;

            static std::once_flag f;
            std::call_once(f, []() {
                for (auto&& x : shared_object_kernel_sections_()) {
                    make_code_object_table_(x, r);
                }
            });

            return r;
        }

        void make_executable_table_(
            const Code_object_table& x, Executable_table& y) const
        {
            for (auto&& a : acc_) {
                const auto it = x.find(hsa_agent_isa(a));
                if (it != x.cend()) {
                    for (auto&& z : it->second) {
                        y[hsa_agent(a)].push_back(executable(z, hsa_agent(a)));
                    }
                }
            }
        }

        const Executable_table& shared_object_executable_table_() const
        {
            static Executable_table r;

            static std::once_flag f;
            std::call_once(f, [this]() {
               make_executable_table_(shared_object_code_object_table_(), r);
            });
            return r;
        }

        static
        decltype(HSA_STATUS_SUCCESS) copy_kernel_symbols(
            hsa_executable_t, hsa_agent_t x, hsa_executable_symbol_t y, void* z)
        {
            auto p = static_cast<typename Kernel_table::mapped_type*>(z);

            if (is_kernel(y)) p->push_back(y);

            return HSA_STATUS_SUCCESS;
        }

        void make_kernel_table_(
            const Executable_table& x, Kernel_table& y) const
        {
            for (auto&& e : x) {
                for (auto&& ex : e.second) {
                    hsa_executable_iterate_agent_symbols(
                        handle(ex), e.first, copy_kernel_symbols, &y[e.first]);
                }
            }
        }

        const Kernel_table& shared_object_kernel_table_() const
        {
            static Kernel_table r;

            static std::once_flag f;
            std::call_once(f, [this]() {
                make_kernel_table_(shared_object_executable_table_(), r);
            });

            return r;
        }

        Program_state() : acc_{hc::accelerator::get_all()}
        {
            acc_.erase(
                std::remove_if(
                    acc_.begin(),
                    acc_.end(),
                    [](const hc::accelerator& x) { return !x.is_hsa_accelerator(); }),
                acc_.end());
        }
    public:
        Program_state(const Program_state&) = default;
        Program_state(Program_state&&) = default;

        Program_state& operator=(const Program_state&) = default;
        Program_state& operator=(Program_state&&) = default;

        ~Program_state() = default;
    };

    inline
    const Program_state& program_state()
    {
        static const Program_state r;

        return r;
    }
}
