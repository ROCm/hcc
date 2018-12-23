//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <hsa/hsa.h>

#include <array>
#include <atomic>
#include <climits>
#include <cstddef>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace hc
{
    namespace detail
    {
        class Signal_pool {
            struct RAII_signal {
                hsa_signal_t signal;

                // CREATORS
                ~RAII_signal()
                {
                    try {
                        throwing_hsa_result_check(
                            hsa_signal_destroy(signal),
                            __FILE__, __func__, __LINE__);
                    }
                    catch (const std::exception& ex) {
                        std::cerr << ex.what() << std::endl;
                    }
                }

                // ACCESSORS
                constexpr
                operator hsa_signal_t() const noexcept { return signal; }
            };

            // IMPLEMENTATION - DATA - STATICS
            static constexpr hsa_signal_value_t init_value_{1};
            static constexpr std::size_t pool_size_{256u};

            using PoolType =
                std::vector<std::pair<std::atomic_flag, RAII_signal>>;

            // IMPLEMENTATION - STATICS
            static
            PoolType& pool_()
            {
                static PoolType r{pool_size_};
                static std::once_flag f;

                std::call_once(f, []() {
                    for (auto&& s : r) {
                        hsa_signal_create(
                            init_value_, 0u, nullptr, &s.second.signal);
                    }
                });

                return r;
            }
        public:
            // DATA - STATICS
            static constexpr hsa_signal_value_t init_value{init_value_};

            // STATICS
            static
            hsa_signal_t allocate() noexcept
            {   // TODO: add backoff and termination.
                do {
                    for (auto&& s : pool_()) {
                        if (s.first.test_and_set()) continue;

                        hsa_signal_store_release(s.second.signal, init_value_);

                        return s.second;
                    }
                } while (true);
            }

            static
            void deallocate(hsa_signal_t x)
            {
                for (auto&& s : pool_()) {
                    if (s.second.signal.handle != x.handle) continue;

                    s.first.clear();

                    return;
                }

                throw std::logic_error{
                    "Tried to deallocate unallocated signal."};
            }

            static
            void wait(hsa_signal_t x) noexcept
            {
                while (hsa_signal_wait_scacquire(
                    x,
                    HSA_SIGNAL_CONDITION_LT,
                    init_value,
                    UINT64_MAX,
                    HSA_WAIT_STATE_BLOCKED) > init_value);
            }
        };
    } // Namespace hc::detail.
} // Namespace hc