//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "hc_callable_attributes.hpp"
#include "hc_index.hpp"
#include "hc_kernel_emitter.hpp"
#include "hc_queue_pool.hpp"
#include "hc_runtime.hpp"
#include "hc_signal_pool.hpp"

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <link.h>

#include <array>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <type_traits>
#include <utility>

namespace hc
{
    class accelerator_view;
    template<int> class tiled_extent;
    template<int> class tiled_index;
}

/** \cond HIDDEN_SYMBOLS */
namespace hc
{
    namespace detail
    {
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

        template<typename Kernel>
        inline
        std::unique_ptr<void, void (*)(void*)>  make_kernel_state(
            const Kernel& f)
        {
            static const auto deleter = [](void* p) {
                if (hsa_amd_memory_unlock(p) != HSA_STATUS_SUCCESS) {
                    std::cerr << "Failed to unlock locked kernel memory; "
                        << "HC Runtime may be in an inconsistent state."
                        << std::endl;
                }

                delete static_cast<Kernel*>(p);
            };

            return std::unique_ptr<void, decltype(deleter)>{new Kernel{f}, deleter};
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
        std::array<std::size_t, n> local_dimensions(
            const hc::tiled_extent<n>& domain)
        {
            std::array<std::size_t, n> r{};
            for (auto i = 0; i != n; ++i) r[i] = domain.tile_dim[i];

            return r;
        }

        template<typename T>
        constexpr
        inline
        std::uint32_t dynamic_lds(const T&) noexcept
        {
            return 0;
        }

        template<int n>
        inline
        std::uint32_t dynamic_lds(const hc::tiled_extent<n>& domain) noexcept
        {
            return domain.get_dynamic_group_segment_size();
        }

        template<typename Domain>
        inline
        std::pair<
            std::array<std::size_t, Domain::rank>,
            std::array<std::size_t, Domain::rank>> dimensions(
                const Domain& domain)
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

        enum Packet_type{ barrier, kernel, n };

        template<Packet_type packet>
        constexpr
        inline
        std::uint16_t make_packet_header() noexcept
        {
            constexpr std::array<std::uint16_t, Packet_type::n> type{{
                HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE,
                HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE
            }};
            constexpr std::uint16_t fence_scope{
                (HSA_FENCE_SCOPE_SYSTEM <<
                    HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
                (HSA_FENCE_SCOPE_SYSTEM <<
                    HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE)};
            constexpr std::uint16_t barrier{
                (packet == Packet_type::barrier) << HSA_PACKET_HEADER_BARRIER};

            return type[packet] | fence_scope | barrier;
        }

        template<typename Kernel, typename Domain>
        inline
        hsa_signal_t make_kernel_dispatch(
            const Domain& domain,
            hsa_kernel_dispatch_packet_t* slot,
            hsa_agent_t agent,
            void* locked_kernel) noexcept
        {
            if (!locked_kernel || !slot) return {};

            *slot = {};

            slot->header = HSA_PACKET_TYPE_INVALID;
            slot->setup =
                Domain::rank << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;

            const auto dims = dimensions(domain);

            slot->workgroup_size_x = dims.second[Domain::rank - 1];
            slot->workgroup_size_y =
                (Domain::rank > 1) ? dims.second[Domain::rank - 2] : 1;
            slot->workgroup_size_z =
                (Domain::rank > 2) ? dims.second[Domain::rank - 3] : 1;
            slot->grid_size_x = dims.first[Domain::rank - 1];
            slot->grid_size_y =
                (Domain::rank > 1) ? dims.first[Domain::rank - 1] : 1;
            slot->grid_size_z =
                (Domain::rank > 2) ? dims.first[Domain::rank - 2] : 1;

            using K = Kernel_emitter<IndexType<Domain>, Kernel>;

            slot->private_segment_size = K::kernel()[agent].private_size;
            slot->group_segment_size =
                K::kernel()[agent].group_size + dynamic_lds(domain);
            slot->kernel_object = K::kernel()[agent].kernel_object;
            slot->kernarg_address = locked_kernel;

            slot->reserved2 = make_packet_header<Packet_type::kernel>();
            slot->completion_signal = Signal_pool::allocate();

            return slot->completion_signal;
        }

        template<typename AcceleratorView, typename Domain, typename Kernel>
        inline
        void launch_kernel(
            const AcceleratorView& av,
            const Domain& domain,
            const Kernel& f)
        {
            launch_kernel_async(av, domain, f).wait();
        }

        template<typename AcceleratorView, typename Domain, typename Kernel>
        inline
        std::shared_future<void> launch_kernel_async(
            const AcceleratorView& av,
            const Domain& domain,
            const Kernel& f)
        {
            auto ks = make_kernel_state(f);

            auto slot = Queue_pool::queue_slot(
                static_cast<hsa_queue_t*>(av.get_hsa_queue()));
            auto signal = make_kernel_dispatch<Kernel>(
                domain,
                static_cast<hsa_kernel_dispatch_packet_t*>(slot.first),
                *static_cast<hsa_agent_t*>(
                    av.get_accelerator().get_hsa_agent()),
                ks.get());
            Queue_pool::enable(slot);

            return std::async([=, ks = std::move(ks)]() mutable {
                Signal_pool::wait(signal);
                ks.reset();
                Signal_pool::deallocate(signal);
            }).share();
        }

        inline
        hsa_signal_t make_barrier(hsa_barrier_and_packet_t* slot) noexcept
        {
            if (!slot) return {};

            *slot = {};

            slot->header = HSA_PACKET_TYPE_INVALID;
            slot->reserved2 = make_packet_header<Packet_type::barrier>();
            slot->completion_signal = Signal_pool::allocate();

            return slot->completion_signal;
        }

        template<typename AcceleratorView>
        inline
        std::shared_future<void> insert_barrier(const AcceleratorView& av)
        {
            auto slot = Queue_pool::queue_slot(
                static_cast<hsa_queue_t*>(av.get_hsa_queue()));
            auto signal = make_barrier(
                static_cast<hsa_barrier_and_packet_t*>(slot.first));
            Queue_pool::enable(slot);

            return std::async([=]() {
                Signal_pool::wait(signal);
                Signal_pool::deallocate(signal);
            }).share();
        }
    } // Namespace hc::detail.
} // Namespace hc.
/** \endcond */
