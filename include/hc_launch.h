//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "hc_callable_attributes.hpp"
#include "hc_runtime.h"
#include "hc_serialize.h"

#include "../hc2/external/elfio/elfio.hpp"

#include <link.h>

#include <array>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <type_traits>
#include <utility>

namespace hc
{
    template<int> class tiled_extent;
    template<int> class tiled_index;
}

/** \cond HIDDEN_SYMBOLS */
namespace detail {

struct Indexer {
    template<int n>
    operator index<n>() const [[hc]]
    {
        int tmp[n]{};
        for (auto i = 0; i != n; ++i) tmp[n - i - 1] = amp_get_global_id(i);

        return index<n>{tmp};
    }

    template<int n>
    operator hc::tiled_index<n>() const [[hc]]
    {
        return {};
    }
};

template<typename Index, typename Kernel>
struct Kernel_emitter_base {
    // TODO: this validation should be done further above, in pfe itself, for
    //       more clarity. It is also a placeholder.
    static
    std::false_type is_callable(...) [[cpu, hc]];
    template<typename I, typename K>
    static
    auto is_callable(I* idx, const K* f) [[cpu, hc]]
        -> decltype((*f)(*idx), std::true_type{});

    static_assert(
        decltype(is_callable(
            std::declval<Index*>(), std::declval<const Kernel*>())){},
        "Invalid Callable passed to parallel_for_each.");
};

template<typename Index, typename Kernel>
struct Kernel_emitter : public Kernel_emitter_base<Index, Kernel> {
    static
    __attribute__((used, annotate("__HCC_KERNEL__")))
    void entry_point(Kernel f) [[cpu, hc]]
    {
        #if __HCC_ACCELERATOR__ != 0
            Index tmp = Indexer{};
            f(tmp);
        #else
            struct { void operator()(const Kernel&) {} } tmp{};
            tmp(f);
        #endif
    }
};

template<typename Kernel, typename... Attrs>
using Kernel_with_attributes =
    hc::attr_impl::Callable_with_AMDGPU_attributes<Kernel, Attrs...>;

template<typename Index, typename Kernel, typename... Attrs>
struct Kernel_emitter<Index, Kernel_with_attributes<Kernel, Attrs...>> :
    public Kernel_emitter_base<
        Index, Kernel_with_attributes<Kernel, Attrs...>> {
    using K = Kernel_with_attributes<Kernel, Attrs...>;

    static
    __attribute__((
        used,
        annotate("__HCC_KERNEL__"),
        amdgpu_flat_work_group_size(
            K::Flat_wg_size_::minimum(), K::Flat_wg_size_::maximum()),
        amdgpu_waves_per_eu(
            K::Waves_per_EU_::minimum(), K::Waves_per_EU_::maximum())))
    void entry_point(K f) [[cpu, hc]]
    {
        #if __HCC_ACCELERATOR__ != 0
            Index tmp = Indexer{};
            f(tmp);
        #else
            struct { void operator()(const K&) {} } tmp{};
            tmp(f);
        #endif
    }
};

template<typename Kernel>
inline
const char* linker_name_for()
{
    static std::once_flag f{};
    static std::string r{};

    // TODO: this should be fused with the one used in mcwamp_hsa.cpp as a
    //       for_each_elf(...) function.
    std::call_once(f, [&]() {
        dl_iterate_phdr([](dl_phdr_info* info, std::size_t, void* pr) {
            const auto base = info->dlpi_addr;
            ELFIO::elfio elf;

            if (!elf.load(base ? info->dlpi_name : "/proc/self/exe")) return 0;

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

                    static const auto k_addr =
                        reinterpret_cast<std::uintptr_t>(&Kernel::entry_point);
                    if (tmp.value + base == k_addr) {
                        *static_cast<std::string*>(pr) = tmp.name;

                        return 1;
                    }
                }
            }

            return 0;
        }, &r);
    });

    if (r.empty()) {
        throw std::runtime_error{
            std::string{"Kernel: "} +
            typeid(&Kernel::entry_point).name() +
            " is not available."};
    }

    return r.c_str();
}

template<typename T>
struct Index_type;

template<int n>
struct Index_type<hc::extent<n>> {
    using index_type = index<n>;
};

template<int n>
struct Index_type<hc::tiled_extent<n>> {
    using index_type = hc::tiled_index<n>;
};

template<typename T>
using IndexType = typename Index_type<T>::index_type;

template<typename Domain, typename Kernel>
inline
void* make_registered_kernel(
    const std::shared_ptr<HCCQueue>& q, const Kernel& f)
{
    struct Deleter {
        void operator()(void* p) const { delete static_cast<Kernel*>(p); }
    };

    using K = detail::Kernel_emitter<IndexType<Domain>, Kernel>;

    std::unique_ptr<void, void (*)(void*)> tmp{
        new Kernel{f}, [](void* p) { delete static_cast<Kernel*>(p); }};
    void* kernel{CLAMP::CreateKernel(
        linker_name_for<K>(), q.get(), std::move(tmp), sizeof(Kernel))};

    return kernel;
}

template<typename T>
constexpr
inline
std::array<std::size_t, T::rank> local_dimensions(const T&)
{
    return std::array<std::size_t, T::rank>{};
}

template<int n>
inline
std::array<std::size_t, n> local_dimensions(const hc::tiled_extent<n>& domain)
{
    std::array<std::size_t, n> r{};
    for (auto i = 0; i != n; ++i) r[i] = domain.tile_dim[i];

    return r;
}

template<typename Domain>
inline
std::pair<
    std::array<std::size_t, Domain::rank>,
    std::array<std::size_t, Domain::rank>> dimensions(const Domain& domain)
{   // TODO: optimise.
    using R = std::pair<
        std::array<std::size_t, Domain::rank>,
        std::array<std::size_t, Domain::rank>>;

    R r{};
    auto tmp = local_dimensions(domain);
    for (auto i = 0; i != Domain::rank; ++i) {
        r.first[i] = domain[i];
        r.second[i] = tmp[i];
    }

    return r;
}

template<typename Domain, typename Kernel>
inline
std::shared_ptr<HCCAsyncOp> launch_kernel_async(
    const std::shared_ptr<HCCQueue>& q,
    const Domain& domain,
    const Kernel& f)
{
  const auto dims{dimensions(domain)};

  return q->LaunchKernelAsync(
        make_registered_kernel<Domain>(q, f),
        Domain::rank,
        dims.first.data(),
        dims.second.data());
}

template<typename Domain, typename Kernel>
inline
void launch_kernel(
    const std::shared_ptr<HCCQueue>& q,
    const Domain& domain,
    const Kernel& f)
{
    const auto dims{dimensions(domain)};

    q->LaunchKernel(
        make_registered_kernel<Domain>(q, f),
        Domain::rank,
        dims.first.data(),
        dims.second.data());
}

template<typename Domain, typename Kernel>
inline
void launch_kernel_with_dynamic_group_memory(
    const std::shared_ptr<HCCQueue>& q,
    const Domain& domain,
    const Kernel& f)
{
    const auto dims{dimensions(domain)};

    q->LaunchKernelWithDynamicGroupMemory(
        make_registered_kernel<Domain>(q, f),
        Domain::rank,
        dims.first.data(),
        dims.second.data(),
        domain.dynamic_group_segment_size());
}

template<typename Domain, typename Kernel>
inline
std::shared_ptr<HCCAsyncOp> launch_kernel_with_dynamic_group_memory_async(
  const std::shared_ptr<HCCQueue>& q,
  const Domain& domain,
  const Kernel& f)
{
    const auto dims{dimensions(domain)};

    return q->LaunchKernelWithDynamicGroupMemoryAsync(
        make_registered_kernel<Domain>(q, f),
        Domain::rank,
        dims.first.data(),
        dims.second.data(),
        domain.get_dynamic_group_segment_size());
}
} // namespace detail
/** \endcond */
