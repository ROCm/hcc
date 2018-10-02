//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "hc_agent_pool.hpp"

#include <hsa/hsa.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace hc
{
    namespace detail
    {
        class Queue_pool {
            struct Deleter {
                void operator()(hsa_queue_t* queue) const noexcept
                {
                    if (hsa_queue_destroy(queue) == HSA_STATUS_SUCCESS) return;

                    std::cerr << "Failed to destroy queue; HC Runtime may be in"
                        << " an inconsistent state." << std::endl;
                }
            };

            using RAIIQueue_ = std::unique_ptr<hsa_queue_t, Deleter>;
            using OnceRAIIQueue_ = std::pair<std::once_flag, RAIIQueue_>;

            // IMPLEMENTATION - DATA - STATICS
            static constexpr std::size_t default_queue_{0u};
            static constexpr std::size_t first_queue_idx_{default_queue_ + 1};

            // IMPLEMENTATION - STATICS
            template<typename T>
            static
            const T& clamp_(const T& lo, const T& x, const T& hi) noexcept
            {
                if (x < lo) return lo;
                if (hi < x) return hi;
                return x;
            }

            static
            RAIIQueue_ make_queue_(hsa_agent_t x)
            {
                static constexpr std::uint32_t default_sz{256u};

                const auto sz = clamp_(
                    Agent_pool::pool()[x].min_queue_size,
                    default_sz,
                    Agent_pool::pool()[x].max_queue_size);

                hsa_queue_t* r{};
                throwing_hsa_result_check(
                    hsa_queue_create(
                        x,
                        sz,
                        HSA_QUEUE_TYPE_MULTI,
                        [](hsa_status_t status, hsa_queue_t*, void*) {
                            try {
                                throwing_hsa_result_check(
                                    status, __FILE__, __func__, __LINE__);
                            }
                            catch (const std::exception& ex) {
                                std::cerr << ex.what() << std::endl;

                                throw;
                            }
                        },
                        nullptr,
                        UINT32_MAX,
                        Agent_pool::pool()[x].max_tile_static_size,
                        &r),
                    __FILE__, __func__, __LINE__);

                return RAIIQueue_{r, Deleter{}};
            }

            static
            std::unordered_map<hsa_agent_t, std::vector<OnceRAIIQueue_>>&
                pool_()
            {
                static std::unordered_map<
                    hsa_agent_t, std::vector<OnceRAIIQueue_>> r;
                static std::once_flag f;

                std::call_once(f, []() {
                    for (auto&& agent : Agent_pool::pool()) {
                        r.emplace(
                            std::piecewise_construct,
                            std::make_tuple(agent.first),
                            std::make_tuple(agent.second.max_queue_count));
                    }
                });

                return r;
            }

            static
            std::uint64_t read_index_(hsa_queue_t* x) noexcept
            {
                return hsa_queue_load_read_index_scacquire(x);
            }

            static
            std::uint64_t write_index_(hsa_queue_t* x) noexcept
            {
                std::uint64_t r;
                do {
                    r = hsa_queue_load_write_index_scacquire(x);
                }
                while (hsa_queue_cas_write_index_scacq_screl(x, r, r + 1) != r);

                return r;
            }
        public:
            static
            hsa_queue_t* default_queue(hsa_agent_t agent)
            {
                if (pool_()[agent].empty()) return nullptr;

                std::call_once(pool_()[agent][default_queue_].first, [=]() {
                    pool_()[agent][default_queue_].second = make_queue_(agent);
                });

                return pool_()[agent][default_queue_].second.get();
            }

            static
            hsa_queue_t* defined_queue(hsa_agent_t agent)
            {
                if (pool_()[agent].empty()) return nullptr;

                static std::unordered_map<
                    hsa_agent_t, std::atomic<std::uint16_t>> cnt;

                const auto defined_queue_cnt = pool_()[agent].size() - 1;
                const auto idx =
                    first_queue_idx_ + (cnt[agent]++ % defined_queue_cnt);

                std::call_once(pool_()[agent][idx].first, [=]() {
                    pool_()[agent][idx].second = make_queue_(agent);
                });

                return pool_()[agent][idx].second.get();
            }

            static
            void enable(
                std::pair<void*, std::uint64_t>& slot,
                hsa_queue_t* queue) noexcept
            {   // Precondition: reserved2 = fully formed packet header.
                auto p = static_cast<hsa_barrier_and_packet_t*>(slot.first);
                std::uint16_t h = p->reserved2;
                p->reserved2 = 0;

                __atomic_store(&p->header, &h, __ATOMIC_SEQ_CST);

                hsa_signal_store_screlease(queue->doorbell_signal, slot.second);
            }

            static
            std::pair<void*, std::uint64_t> queue_slot(hsa_queue_t* queue)
            {   // TODO: add per-queue backoff.
                if (!queue) {
                    throw std::logic_error{
                        "Tried to get slot in non-existing queue."};
                }

                auto p = static_cast<hsa_kernel_dispatch_packet_t*>(
                    queue->base_address);
                do {
                    const auto f = read_index_(queue);
                    const auto l = write_index_(queue);

                    if (queue->size <= l - f) continue;

                    return {p + (l % queue->size), l};
                } while (true);
            }
        };
    }
} // Namespace hc.