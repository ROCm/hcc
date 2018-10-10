//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <atomic>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace hc
{
    namespace detail
    {
        template<typename T, std::size_t n = 128, std::size_t size = 65536u>
        class N_way_set_associative_cache {
            static_assert(
                n <= size,
                "Number of sets must not be greater than cache size.");

            using GuardedLockedPtr_ = std::pair<
                std::atomic_flag, std::pair<const void*, void*>>;

            // IMPLEMENTATION - DATA - STATICS
            static constexpr std::uint8_t bit_cnt_{
                sizeof(std::uintptr_t) * CHAR_BIT};
            static constexpr std::uint8_t byte_offset_bits_{2u};
            static constexpr std::uint8_t set_bits_{
                bit_cnt_ - __builtin_clzll(n) - 1u};
            static constexpr auto set_size_ = size / n;
            static constexpr std::uint8_t tag_bits_{
                bit_cnt_ - set_bits_ - byte_offset_bits_};

            // IMPLEMENTATION - DATA
            std::array<GuardedLockedPtr_, size> cache_{};

            // IMPLEMENTATION - STATICS
            static
            constexpr
            std::uintptr_t make_bitmask_(
                std::uint8_t first, std::uint8_t last) noexcept [[cpu, hc]]
            {
                return (first == last) ?
                    0u : ((UINTPTR_MAX >> (bit_cnt_ - (first - last))) << last);
            }

            static
            std::uintptr_t byte_offset_(const void* p) noexcept [[cpu, hc]]
            {
                constexpr auto mask = make_bitmask_(byte_offset_bits_, 0u);

                return reinterpret_cast<std::uintptr_t>(p) & mask;
            }

            static
            std::uintptr_t set_(const void* p) noexcept [[cpu, hc]]
            {
                constexpr auto mask = make_bitmask_(
                    set_bits_ + byte_offset_bits_, byte_offset_bits_);

                return (reinterpret_cast<std::uintptr_t>(p) & mask) >>
                    byte_offset_bits_;
            }

            static
            std::uintptr_t tag_(const void* p) noexcept [[cpu, hc]]
            {
                constexpr auto mask = make_bitmask_(
                    tag_bits_ + set_bits_ + byte_offset_bits_,
                    set_bits_ + byte_offset_bits_);

                return (reinterpret_cast<std::uintptr_t>(p) & mask) >>
                    (set_bits_ + byte_offset_bits_);
            }

            static
            constexpr
            std::uint32_t flat_set_idx_(const void* ptr) noexcept [[cpu, hc]]
            {
                return set_(ptr) * size / n;
            }

            // IMPLEMENTATION - ACCESSORS
            typename decltype(cache_)::size_type find_cache_entry_(
                const void* ptr) const noexcept [[cpu, hc]]
            {
                const auto idx = flat_set_idx_(ptr);

                for (auto i = 0u; i != set_size_; ++i) {
                    if (cache_[idx + i].second.first == ptr) return idx + i;
                }

                return cache_.size();
            }
        public:
            // TODO: these are not yet truly iterators, and proper iteration is
            //       to be added in the future.
            using const_iterator = const T*;
            using iterator = T*;
            using size_type = std::size_t;

            // CREATORS
            N_way_set_associative_cache() [[cpu, hc]] = default;
            N_way_set_associative_cache(
                const N_way_set_associative_cache&) [[cpu, hc]] = delete;
            N_way_set_associative_cache(
                N_way_set_associative_cache&&) [[cpu, hc]] = default;
            ~N_way_set_associative_cache() [[cpu, hc]] = default;

            // MANIPULATORS
            N_way_set_associative_cache& operator=(
                const N_way_set_associative_cache&) [[cpu, hc]] = delete;
            N_way_set_associative_cache& operator=(
                N_way_set_associative_cache&&) [[cpu, hc]] = default;

            constexpr
            // TODO: C++11 is odd with constexpr, if / when we move up revisit.
            iterator end() const noexcept [[cpu, hc]]
            {
                return nullptr;
            }

            iterator find(const void* ptr) noexcept [[cpu, hc]]
            {
                const auto idx = find_cache_entry_(ptr);

                if (idx == cache_.size()) return end();

                return &cache_[idx].second.second;
            }

            size_type erase(const void* ptr) noexcept [[cpu, hc]]
            {
                auto idx = find_cache_entry_(ptr);

                if (idx == cache_.size()) return 0u;

                cache_[idx].second = {};
                cache_[idx].first.clear();

                return 1u;
            }

            std::pair<iterator, bool> insert(
                const void* ptr, T x) noexcept [[cpu, hc]]
            {
                const auto idx = flat_set_idx_(ptr);

                for (auto i = 0u; i != set_size_; ++i) {
                    if (cache_[idx + i].first.test_and_set()) continue;

                    cache_[idx + i].second.first = ptr;
                    cache_[idx + i].second.second = std::move(x);

                    return {&cache_[idx + i].second.second, true};
                }

                return {end(), false};
            }

            // ACCESSORS
            constexpr
            const_iterator cend() const noexcept [[cpu, hc]]
            {
                return nullptr;
            }

            const_iterator find(const void* ptr) const noexcept [[cpu, hc]]
            {   // TODO: remove abusive usage of const_cast.
                return
                    const_cast<N_way_set_associative_cache&>(*this).find(ptr);
            }
        };
    } // Namespace hc::detail.
} // Namespace hc.